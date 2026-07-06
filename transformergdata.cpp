// SPDX-License-Identifier: MIT
//
// Training data generator for SurfaceFormer / Mesh Denoising Transformer.
//
// This program keeps the original command-line interface and binary output
// format of transformergdata.cpp:
//   ./transformergdata <profile.txt> <lsd_size> <patch_face_count> <axis_excluded_vertex>
//
// Profile format:
//   <number_of_meshes>
//   <clean_mesh_0>
//   ...
//   <clean_mesh_N-1>
//   <noisy_mesh_0>
//   ...
//   <noisy_mesh_N-1>
//   <output_name_template>
//
// The output_name_template follows the legacy naming convention used by the
// original code. Characters at offsets [-9, -8] are overwritten by the two-digit
// mesh index and characters at offsets [-6, -5] by the block id and '0'.
// Therefore a template such as "data/00_00.bin" produces:
//   data/00_00.bin  LSD normals,       float32 [F, lsd_size, lsd_size, 3]
//   data/00_10.bin  clean normals,     float32 [F, 3]
//   data/00_20.bin  face centroids,    float32 [F, 3]
//   data/00_30.bin  patch face ids,    int32   [F, patch_face_count]
//   data/00_40.bin  polar axes,        float32 [F, 3]
//   data/00_50.bin  noisy vertices,    float32 [F, 9]
//   data/00_60.bin  clean vertices,    float32 [F, 9]

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "Eigen/Dense"


#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

inline double square(double x) { return x * x; }

// -----------------------------------------------------------------------------
// Mesh type
// -----------------------------------------------------------------------------
struct MyTraits : OpenMesh::DefaultTraits {
  typedef OpenMesh::Vec3d Point;
  typedef OpenMesh::Vec3d Normal;
  typedef double TexCoord1D;
  typedef OpenMesh::Vec2d TexCoord2D;
  typedef OpenMesh::Vec3d TexCoord3D;

  VertexAttributes(OpenMesh::Attributes::Status | OpenMesh::Attributes::Normal |
                   OpenMesh::Attributes::Color);
  HalfedgeAttributes(OpenMesh::Attributes::Status |
                     OpenMesh::Attributes::PrevHalfedge);
  FaceAttributes(OpenMesh::Attributes::Status | OpenMesh::Attributes::Normal |
                 OpenMesh::Attributes::Color);
  EdgeAttributes(OpenMesh::Attributes::Status | OpenMesh::Attributes::Color);
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> TriMesh;

struct SegmentLine {
  TriMesh::Point v1;
  TriMesh::Point v2;
};

// Runtime parameters. They are intentionally kept global to mirror the original
// program while avoiding a large parameter list in inner-loop functions.
int g_lsd_size = 20;
int g_patch_face_count = 240;
int g_axis_excluded_vertex = 2;

// Virtual Cartesian grid used to derive the geodesic length and direction.
// entry = {i - lsd_size/2, j - lsd_size/2, dx^2 + dy^2}
std::vector<std::array<int, 3> > g_sampling_grid;

// -----------------------------------------------------------------------------
// Legacy 3D kd-tree for nearest-face fallback in patch generation.
//
// IMPORTANT: The *_30.bin files store the exact order of patch face ids.  The
// original program used a global comparison axis (iidx) and a global priority
// queue (nq) in kd-tree queries.  Even small "cleanups" such as changing the
// tie-break rule of the priority queue can alter fallback patches on symmetric
// meshes.  Since this generator is expected to reproduce the original training
// data, this implementation intentionally keeps the legacy behavior.
// -----------------------------------------------------------------------------
int iidx = 0;

struct point {
  double x[3];
  int index;
  bool operator<(const point& u) const { return x[iidx] < u.x[iidx]; }
};

typedef std::pair<double, point> tp;
std::priority_queue<tp> nq;

struct KDTree {
  int N;
  std::vector<point> po;
  std::vector<point> pt;
  std::vector<int> son;

  void init(int NN) {
    N = NN;
    po.resize(N);
    pt.resize((N << 2));
    son.resize((N << 2));
  }

  void build(int l, int r, int rt = 1, int dep = 0) {
    if (l > r) return;
    son[rt] = r - l;
    son[rt * 2] = son[rt * 2 + 1] = -1;
    iidx = dep % 3;
    int mid = (l + r) / 2;
    std::nth_element(po.begin() + l, po.begin() + mid, po.begin() + r + 1);
    pt[rt] = po[mid];
    build(l, mid - 1, rt * 2, dep + 1);
    build(mid + 1, r, rt * 2 + 1, dep + 1);
  }

  void query(point p, int m, int rt = 1, int dep = 0) const {
    if (son[rt] == -1) return;
    tp nd(0, pt[rt]);
    for (int i = 0; i < 3; i++) nd.first += square(nd.second.x[i] - p.x[i]);
    int dim = dep % 3, x = rt * 2, y = rt * 2 + 1, fg = 0;
    if (p.x[dim] >= pt[rt].x[dim]) std::swap(x, y);
    if (~son[x]) query(p, m, x, dep + 1);
    if (static_cast<int>(nq.size()) < m) {
      nq.push(nd);
      fg = 1;
    } else {
      if (nd.first < nq.top().first) {
        nq.pop();
        nq.push(nd);
      }
      if (square(p.x[dim] - pt[rt].x[dim]) < nq.top().first) fg = 1;
    }
    if (~son[y] && fg) query(p, m, y, dep + 1);
  }
};

struct MeshData {
  TriMesh clean_mesh;
  TriMesh noisy_mesh;
  std::vector<TriMesh::Normal> clean_normals;
  std::vector<TriMesh::Normal> noisy_normals;
  std::vector<TriMesh::Point> face_centroids;
  std::vector<SegmentLine> halfedge_segments;
  KDTree centroid_tree;
  double sigma_s = 0.0;
};

// -----------------------------------------------------------------------------
// Basic geometry preprocessing
// -----------------------------------------------------------------------------
void ComputeFaceNormals(TriMesh& mesh, std::vector<TriMesh::Normal>* normals) {
  mesh.request_face_normals();
  mesh.update_face_normals();

  normals->assign(mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));
  for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
    (*normals)[f_it->idx()] = mesh.normal(*f_it);
  }
}

void ComputeFaceCentroids(TriMesh& mesh, std::vector<TriMesh::Point>* centroids) {
  centroids->assign(mesh.n_faces(), TriMesh::Point(0.0, 0.0, 0.0));
  for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
    (*centroids)[f_it->idx()] = mesh.calc_face_centroid(*f_it);
  }
}

double ComputeSigmaS(double multiple, const std::vector<TriMesh::Point>& centroids,
                     TriMesh& mesh) {
  double sigma_s = 0.0;
  double count = 0.0;
  for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
    const TriMesh::Point& ci = centroids[f_it->idx()];
    for (TriMesh::FaceFaceIter ff_it = mesh.ff_iter(*f_it); ff_it.is_valid(); ++ff_it) {
      const TriMesh::Point& cj = centroids[ff_it->idx()];
      sigma_s += (cj - ci).length();
      count += 1.0;
    }
  }
  if (count == 0.0) return 0.0;
  return sigma_s * multiple / count;
}

void BuildHalfedgeSegments(TriMesh& mesh, std::vector<SegmentLine>* segments) {
  segments->resize(mesh.n_halfedges());
  for (TriMesh::HalfedgeIter h_it = mesh.halfedges_begin(); h_it != mesh.halfedges_end(); ++h_it) {
    const int h = h_it->idx();
    (*segments)[h].v1 = mesh.point(mesh.from_vertex_handle(*h_it));
    (*segments)[h].v2 = mesh.point(mesh.to_vertex_handle(*h_it));
  }
}

void BuildSamplingGrid() {
  if (g_lsd_size <= 0) throw std::runtime_error("lsd_size must be positive.");
  g_sampling_grid.assign(static_cast<size_t>(g_lsd_size * g_lsd_size),
                         std::array<int, 3>{{0, 0, 0}});
  for (int i = 0; i < g_lsd_size; ++i) {
    for (int j = 0; j < g_lsd_size; ++j) {
      const int dx = i - g_lsd_size / 2;
      const int dy = j - g_lsd_size / 2;
      g_sampling_grid[static_cast<size_t>(i * g_lsd_size + j)] =
          std::array<int, 3>{{dx, dy, dx * dx + dy * dy}};
    }
  }
}

// -----------------------------------------------------------------------------
// Intersection and geodesic propagation
// -----------------------------------------------------------------------------
bool CalculateLineLineIntersection(const TriMesh::Point& line1Point1,
                                   const TriMesh::Point& line1Point2,
                                   const TriMesh::Point& line2Point1,
                                   const TriMesh::Point& line2Point2,
                                   TriMesh::Point* resultSegmentPoint,
                                   TriMesh::Normal* rayDirection) {
  const TriMesh::Point p1 = line1Point1;
  const TriMesh::Point p2 = line1Point2;
  const TriMesh::Point p3 = line2Point1;
  const TriMesh::Point p4 = line2Point2;
  const TriMesh::Point p13 = p1 - p3;
  const TriMesh::Point p43 = p4 - p3;

  if (p43.length() < 1e-8) return false;
  const TriMesh::Point p21 = p2 - p1;
  if (p21.length() < 1e-8) return false;

  const double d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2];
  const double d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2];
  const double d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2];
  const double d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2];
  const double d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

  const double denom = d2121 * d4343 - d4321 * d4321;
  if (denom == 0.0) return false;
  const double numer = d1343 * d4321 - d1321 * d4343;

  const double mua = numer / denom;
  const double mub = (d1343 + d4321 * mua) / d4343;

  TriMesh::Point hit1(p1[0] + mua * p21[0], p1[1] + mua * p21[1],
                      p1[2] + mua * p21[2]);
  const TriMesh::Point hit2(p3[0] + mub * p43[0], p3[1] + mub * p43[1],
                            p3[2] + mub * p43[2]);

  if ((hit2 - hit1).length() < 1e-6 && mua >= -1e-6 && mua <= 1.0 + 1e-6 &&
      mub >= 0.0) {
    // Avoid landing exactly on a vertex/edge endpoint. This preserves the
    // original behavior and prevents immediate re-intersection with the same
    // edge due to numerical precision.
    if (mua > 1.0 - 1e-6) {
      hit1 = TriMesh::Point(p1[0] + 0.9999 * p21[0], p1[1] + 0.9999 * p21[1],
                            p1[2] + 0.9999 * p21[2]);
      *rayDirection = hit1 - line2Point1;
      rayDirection->normalize();
    }
    if (mua < 1e-6) {
      hit1 = TriMesh::Point(p1[0] + 0.0001 * p21[0], p1[1] + 0.0001 * p21[1],
                            p1[2] + 0.0001 * p21[2]);
      *rayDirection = hit1 - line2Point1;
      rayDirection->normalize();
    }
    *resultSegmentPoint = hit1;
    return true;
  }
  return false;
}

void StoreNormal(const std::vector<TriMesh::Normal>& normals, int face_id, float* out) {
  out[0] = static_cast<float>(normals[face_id][0]);
  out[1] = static_cast<float>(normals[face_id][1]);
  out[2] = static_cast<float>(normals[face_id][2]);
}

// Generate an unnormalized LSD for one face. Normalization is intentionally not
// performed here; the training loader can compute the patch-level transform on
// the fly, as in the original implementation.
void GenerateFaceLSD(int face_id, TriMesh& noisy_mesh, float* lsd_out, float* axis_out,
                     double sigma_s,
                     const std::vector<TriMesh::Normal>& clean_normals,
                     const std::vector<SegmentLine>& halfedge_segments,
                     const std::vector<TriMesh::Normal>& noisy_normals,
                     const std::vector<TriMesh::Point>& face_centroids) {
  (void)clean_normals;  // Clean normals are written outside this function.

  int excluded_vertex = g_axis_excluded_vertex;
  if (excluded_vertex < 0 || excluded_vertex > 2) {
    excluded_vertex = std::rand() % 3;  // legacy fallback path
  }

  // Polar axis: direction from face centroid to the midpoint of the two vertices
  // not excluded by axis_excluded_vertex. With the default value 2, this matches
  // the paper setting: use vertex 0 and 1 to define the axis.
  TriMesh::Point axis_point(0.0, 0.0, 0.0);
  int local_id = 0;
  for (TriMesh::FaceVertexIter fv_it = noisy_mesh.fv_begin(TriMesh::FaceHandle(face_id));
       local_id <= 2; ++local_id, ++fv_it) {
    if (local_id != excluded_vertex) axis_point += noisy_mesh.point(*fv_it);
  }
  axis_point /= 2.0;

  TriMesh::Normal start_dir = axis_point - face_centroids[face_id];
  start_dir.normalize();

  axis_out[0] = static_cast<float>(start_dir[0]);
  axis_out[1] = static_cast<float>(start_dir[1]);
  axis_out[2] = static_cast<float>(start_dir[2]);

  const Eigen::Vector3d source_face_normal(noisy_normals[face_id][0], noisy_normals[face_id][1],
                                           noisy_normals[face_id][2]);

  for (int i = 0; i < g_lsd_size; ++i) {
    for (int j = 0; j < g_lsd_size; ++j) {
      const std::array<int, 3>& grid = g_sampling_grid[static_cast<size_t>(i * g_lsd_size + j)];
      const int dx = grid[0];
      const int dy = grid[1];
      const int radius2 = grid[2];

      float* sample_out = lsd_out + static_cast<size_t>(i * g_lsd_size * 3 + j * 3);
      if (dx == 0 && dy == 0) {
        StoreNormal(noisy_normals, face_id, sample_out);
        continue;
      }

      const double target_length = sigma_s * std::sqrt(static_cast<double>(radius2));
      double current_length = 0.0;
      TriMesh::Point current_point = face_centroids[face_id];
      TriMesh::Normal ray_dir = start_dir;

      // Rotate the polar axis to the desired virtual Cartesian direction. The
      // atan2(dx, dy) order is legacy-compatible and should not be swapped.
      double angle = 0.0;
      if (dx == 0) {
        angle = (dy > 0) ? 0.0 : M_PI;
      } else {
        angle = std::atan2(static_cast<double>(dx), static_cast<double>(dy));
      }

      Eigen::Vector3d ray_vec(ray_dir[0], ray_dir[1], ray_dir[2]);
      ray_vec = Eigen::AngleAxisd(angle, source_face_normal).matrix() * ray_vec;
      ray_dir[0] = ray_vec[0];
      ray_dir[1] = ray_vec[1];
      ray_dir[2] = ray_vec[2];
      ray_dir.normalize();

      int previous_halfedge = -1;
      int end_state = 0;
      OpenMesh::FaceHandle current_face(face_id);
      std::unordered_set<int> visited_faces;
      visited_faces.reserve(32);

      while (end_state == 0) {
        visited_faces.insert(current_face.idx());
        int checked_edges = 0;
        int found_edge = 0;

        for (TriMesh::FaceHalfedgeIter h_it = noisy_mesh.fh_begin(current_face);
             h_it != noisy_mesh.fh_end(current_face); ++h_it) {
          const int halfedge_id = h_it->idx();
          if (halfedge_id == previous_halfedge) continue;

          ++checked_edges;
          TriMesh::Point next_point;
          const TriMesh::Point ray_end = current_point + ray_dir * sigma_s * 100.0;
          if (!CalculateLineLineIntersection(halfedge_segments[halfedge_id].v1,
                                             halfedge_segments[halfedge_id].v2,
                                             current_point, ray_end, &next_point, &ray_dir)) {
            if ((checked_edges == 2 && found_edge == 0 && previous_halfedge != -1) ||
                (checked_edges == 3 && found_edge == 0 && previous_halfedge == -1)) {
              // Legacy behavior: leave the initialized zero normal for this
              // sample if geodesic tracing fails.
              end_state = -4;
              break;
            }
            continue;
          }

          found_edge = 1;
          current_length += (next_point - current_point).length();

          if (current_length >= target_length) {
            StoreNormal(noisy_normals, current_face.idx(), sample_out);
            end_state = -1;
            break;
          }

          current_point = next_point;
          previous_halfedge = noisy_mesh.opposite_halfedge_handle(*h_it).idx();
          const OpenMesh::FaceHandle next_face =
              noisy_mesh.face_handle(noisy_mesh.opposite_halfedge_handle(*h_it));

          if (next_face.idx() == -1) {  // boundary reached
            end_state = -2;
            break;
          }
          if (visited_faces.find(next_face.idx()) != visited_faces.end()) {
            end_state = -3;
            break;
          }

          const Eigen::Vector3d normal_current(noisy_normals[current_face.idx()][0],
                                               noisy_normals[current_face.idx()][1],
                                               noisy_normals[current_face.idx()][2]);
          const Eigen::Vector3d normal_next(noisy_normals[next_face.idx()][0],
                                            noisy_normals[next_face.idx()][1],
                                            noisy_normals[next_face.idx()][2]);
          const Eigen::Matrix3d rot = Eigen::Quaterniond::FromTwoVectors(normal_current, normal_next).toRotationMatrix();

          Eigen::Vector3d dir(ray_dir[0], ray_dir[1], ray_dir[2]);
          dir = rot * dir;
          ray_dir[0] = dir[0];
          ray_dir[1] = dir[1];
          ray_dir[2] = dir[2];
          ray_dir.normalize();

          current_face = next_face;
          break;
        }
      }
    }
  }
}

// Build a fixed-size overlapping patch for one center face.
//
// Compatibility note:
// The legacy gfacelist() iterates over std::unordered_set directly when it
// expands the patch. That iteration order is implementation-dependent, and the
// first g_patch_face_count accepted faces are written to *_30.bin. Therefore,
// even seemingly harmless changes such as unordered_set::reserve() or changing
// the traversal to a sorted/BFS queue alter the patch-face-id file. This function
// deliberately keeps the same expansion structure as the original code.
void GeneratePatchFaceList(int center_face, TriMesh& mesh, int* patch_out,
                           const std::vector<TriMesh::Point>& face_centroids,
                           const KDTree& centroid_tree) {
  std::unordered_set<int> selected;  // Do NOT reserve: preserve legacy bucket layout.
  int found_new_face = 1;

  patch_out[0] = center_face;
  selected.insert(center_face);

  while (true) {
    found_new_face = 0;

    std::vector<int> current_faces;
    for (std::unordered_set<int>::const_iterator it = selected.begin(); it != selected.end(); ++it) {
      current_faces.push_back(*it);
    }

    for (std::vector<int>::const_iterator it = current_faces.begin(); it != current_faces.end(); ++it) {
      for (TriMesh::FaceVertexIter fv_it = mesh.fv_begin(TriMesh::FaceHandle(*it));
           fv_it.is_valid(); ++fv_it) {
        for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*fv_it);
             vf_it.is_valid(); ++vf_it) {
          const int candidate = vf_it->idx();
          if (selected.find(candidate) == selected.end()) {
            found_new_face = 1;
            selected.insert(candidate);
            patch_out[selected.size() - 1] = candidate;

            if (static_cast<int>(selected.size()) == g_patch_face_count) return;
          }
        }
      }
    }

    if (found_new_face == 0) break;
  }

  if (static_cast<int>(selected.size()) < g_patch_face_count) {
    std::printf("(fi==%d)||", center_face);
    point query;
    query.x[0] = face_centroids[center_face][0];
    query.x[1] = face_centroids[center_face][1];
    query.x[2] = face_centroids[center_face][2];
    query.index = center_face;

    while (!nq.empty()) nq.pop();
    centroid_tree.query(query, g_patch_face_count + 10);

    std::vector<tp> result;
    for (int cc = 0; !nq.empty(); cc++) {
      tp z = nq.top();
      result.push_back(z);
      nq.pop();
    }
    for (int cc = static_cast<int>(result.size()) - 1; cc >= 0; cc--) {
      const int candidate = result[cc].second.index;
      if (selected.find(candidate) == selected.end()) {
        selected.insert(candidate);
        patch_out[selected.size() - 1] = candidate;
        if (static_cast<int>(selected.size()) == g_patch_face_count) break;
      }
    }
  }
}

void StoreFaceVertices(TriMesh& mesh, int face_id, float* out, int* vertex_ids) {
  int local_id = 0;
  for (TriMesh::FaceVertexIter fv_it = mesh.fv_begin(TriMesh::FaceHandle(face_id));
       fv_it.is_valid(); ++fv_it) {
    const OpenMesh::VertexHandle vh = fv_it.handle();
    if (vertex_ids != NULL) vertex_ids[local_id] = vh.idx();
    const TriMesh::Point p = mesh.point(vh);
    out[local_id * 3 + 0] = static_cast<float>(p[0]);
    out[local_id * 3 + 1] = static_cast<float>(p[1]);
    out[local_id * 3 + 2] = static_cast<float>(p[2]);
    ++local_id;
  }
}

void StoreCleanFaceVerticesWithCheck(TriMesh& clean_mesh, int face_id, const int* noisy_vertex_ids,
                                     float* out) {
  int local_id = 0;
  for (TriMesh::FaceVertexIter fv_it = clean_mesh.fv_begin(TriMesh::FaceHandle(face_id));
       fv_it.is_valid(); ++fv_it) {
    const OpenMesh::VertexHandle vh = fv_it.handle();
    if (noisy_vertex_ids != NULL && noisy_vertex_ids[local_id] != vh.idx()) {
      std::cerr << "Warning: vertex order mismatch at face " << face_id << "\n";
    }
    const TriMesh::Point p = clean_mesh.point(vh);
    out[local_id * 3 + 0] = static_cast<float>(p[0]);
    out[local_id * 3 + 1] = static_cast<float>(p[1]);
    out[local_id * 3 + 2] = static_cast<float>(p[2]);
    ++local_id;
  }
}

// -----------------------------------------------------------------------------
// I/O helpers
// -----------------------------------------------------------------------------
template <typename T>
void WriteBinary(const std::string& filename, const std::vector<T>& data) {
  std::ofstream out(filename.c_str(), std::ios::binary);
  if (!out) throw std::runtime_error("Failed to open output file: " + filename);
  if (!data.empty()) {
    out.write(reinterpret_cast<const char*>(&data[0]),
              static_cast<std::streamsize>(sizeof(T) * data.size()));
  }
  if (!out) throw std::runtime_error("Failed while writing output file: " + filename);
}

std::string MakeOutputName(const std::string& template_name, int mesh_id, int block_id) {
  if (template_name.size() < 9) {
    throw std::runtime_error("Output name template is too short for the legacy naming rule.");
  }
  if (mesh_id < 0 || mesh_id > 99) {
    throw std::runtime_error("The legacy output naming rule supports only mesh ids in [0, 99].");
  }
  if (block_id < 0 || block_id > 9) {
    throw std::runtime_error("The legacy output naming rule supports only one-digit block ids.");
  }

  std::string out = template_name;
  const size_t n = out.size();
  out[n - 9] = static_cast<char>('0' + mesh_id / 10);
  out[n - 8] = static_cast<char>('0' + mesh_id % 10);
  out[n - 6] = static_cast<char>('0' + block_id);
  out[n - 5] = '0';
  return out;
}

std::vector<std::string> ReadProfile(const std::string& profile_name, int* number_of_meshes,
                                     std::string* output_template) {
  std::ifstream profile(profile_name.c_str());
  if (!profile) throw std::runtime_error("Failed to open profile: " + profile_name);

  if (!(profile >> *number_of_meshes) || *number_of_meshes <= 0) {
    throw std::runtime_error("Invalid mesh count in profile: " + profile_name);
  }

  std::vector<std::string> names(static_cast<size_t>(*number_of_meshes * 2));
  for (size_t i = 0; i < names.size(); ++i) {
    if (!(profile >> names[i])) {
      throw std::runtime_error("Profile ended before all mesh paths were read.");
    }
  }
  if (!(profile >> *output_template)) {
    throw std::runtime_error("Profile ended before output name template was read.");
  }
  return names;
}

void ReadMeshes(const std::vector<std::string>& mesh_names, int number_of_meshes,
                std::vector<MeshData>* meshes) {
  meshes->resize(static_cast<size_t>(number_of_meshes));

  std::cout << "Reading clean meshes..." << std::endl;
  for (int i = 0; i < number_of_meshes; ++i) {
    if (!OpenMesh::IO::read_mesh((*meshes)[i].clean_mesh, mesh_names[static_cast<size_t>(i)])) {
      throw std::runtime_error("Failed to read clean mesh: " + mesh_names[static_cast<size_t>(i)]);
    }
  }

  std::cout << "Reading noisy meshes..." << std::endl;
  for (int i = 0; i < number_of_meshes; ++i) {
    const std::string& noisy_name = mesh_names[static_cast<size_t>(number_of_meshes + i)];
    if (!OpenMesh::IO::read_mesh((*meshes)[i].noisy_mesh, noisy_name)) {
      throw std::runtime_error("Failed to read noisy mesh: " + noisy_name);
    }
    if ((*meshes)[i].noisy_mesh.n_faces() != (*meshes)[i].clean_mesh.n_faces()) {
      throw std::runtime_error("Clean/noisy face-count mismatch: " + noisy_name);
    }
  }
}

void PreprocessMeshes(std::vector<MeshData>* meshes) {
  for (size_t i = 0; i < meshes->size(); ++i) {
    MeshData& data = (*meshes)[i];
    std::cout << "Preprocessing mesh " << i << " (faces=" << data.noisy_mesh.n_faces() << ")..."
              << std::endl;

    ComputeFaceNormals(data.clean_mesh, &data.clean_normals);
    ComputeFaceNormals(data.noisy_mesh, &data.noisy_normals);
    ComputeFaceCentroids(data.noisy_mesh, &data.face_centroids);
    BuildHalfedgeSegments(data.noisy_mesh, &data.halfedge_segments);
    data.centroid_tree.init(static_cast<int>(data.face_centroids.size()));
    for (int k = 0; k < static_cast<int>(data.face_centroids.size()); ++k) {
      data.centroid_tree.po[k].x[0] = data.face_centroids[k][0];
      data.centroid_tree.po[k].x[1] = data.face_centroids[k][1];
      data.centroid_tree.po[k].x[2] = data.face_centroids[k][2];
      data.centroid_tree.po[k].index = k;
    }
    if (!data.face_centroids.empty()) {
      data.centroid_tree.build(0, static_cast<int>(data.face_centroids.size()) - 1);
    }

    // This matches the original code: getSigmaS(2, ...) / 8.
    data.sigma_s = ComputeSigmaS(2.0, data.face_centroids, data.noisy_mesh) / 8.0;
  }
}

void GenerateOneMesh(int mesh_id, MeshData* data, const std::string& output_template) {
  const int face_count = static_cast<int>(data->noisy_mesh.n_faces());
  const size_t lsd_values_per_face = static_cast<size_t>(g_lsd_size) * g_lsd_size * 3;

  std::cout << "Generating LSD for mesh " << mesh_id << " (faces=" << face_count << ")..."
            << std::endl;

  std::vector<float> lsd_cache(static_cast<size_t>(face_count) * lsd_values_per_face, 0.0f);
  std::vector<float> clean_normal_cache(static_cast<size_t>(face_count) * 3, 0.0f);
  std::vector<float> centroid_cache(static_cast<size_t>(face_count) * 3, 0.0f);
  std::vector<int> patch_index_cache(static_cast<size_t>(face_count) * g_patch_face_count, 0);
  std::vector<float> axis_cache(static_cast<size_t>(face_count) * 3, 0.0f);
  std::vector<float> noisy_vertex_cache(static_cast<size_t>(face_count) * 9, 0.0f);
  std::vector<float> clean_vertex_cache(static_cast<size_t>(face_count) * 9, 0.0f);

  // Serial generation. Keeping this loop single-threaded preserves the legacy
  // random-axis behavior when axis_excluded_vertex is outside [0, 2], because
  // GenerateFaceLSD then uses std::rand() in face order.
  for (int face_id = 0; face_id < face_count; ++face_id) {
    float* lsd_ptr = &lsd_cache[static_cast<size_t>(face_id) * lsd_values_per_face];
    GenerateFaceLSD(face_id, data->noisy_mesh, lsd_ptr,
                    &axis_cache[static_cast<size_t>(face_id) * 3], data->sigma_s,
                    data->clean_normals, data->halfedge_segments, data->noisy_normals,
                    data->face_centroids);

    clean_normal_cache[static_cast<size_t>(face_id) * 3 + 0] =
        static_cast<float>(data->clean_normals[face_id][0]);
    clean_normal_cache[static_cast<size_t>(face_id) * 3 + 1] =
        static_cast<float>(data->clean_normals[face_id][1]);
    clean_normal_cache[static_cast<size_t>(face_id) * 3 + 2] =
        static_cast<float>(data->clean_normals[face_id][2]);

    centroid_cache[static_cast<size_t>(face_id) * 3 + 0] =
        static_cast<float>(data->face_centroids[face_id][0]);
    centroid_cache[static_cast<size_t>(face_id) * 3 + 1] =
        static_cast<float>(data->face_centroids[face_id][1]);
    centroid_cache[static_cast<size_t>(face_id) * 3 + 2] =
        static_cast<float>(data->face_centroids[face_id][2]);

    GeneratePatchFaceList(face_id, data->noisy_mesh,
                          &patch_index_cache[static_cast<size_t>(face_id) *
                                             g_patch_face_count],
                          data->face_centroids, data->centroid_tree);

    int vertex_ids[3] = {-1, -1, -1};
    StoreFaceVertices(data->noisy_mesh, face_id,
                      &noisy_vertex_cache[static_cast<size_t>(face_id) * 9],
                      vertex_ids);
    StoreCleanFaceVerticesWithCheck(data->clean_mesh, face_id, vertex_ids,
                                    &clean_vertex_cache[static_cast<size_t>(face_id) * 9]);
  }

  const std::string lsd_file = MakeOutputName(output_template, mesh_id, 0);
  std::cout << "Writing " << lsd_file << std::endl;
  WriteBinary(lsd_file, lsd_cache);
  WriteBinary(MakeOutputName(output_template, mesh_id, 1), clean_normal_cache);
  WriteBinary(MakeOutputName(output_template, mesh_id, 2), centroid_cache);
  WriteBinary(MakeOutputName(output_template, mesh_id, 3), patch_index_cache);
  WriteBinary(MakeOutputName(output_template, mesh_id, 4), axis_cache);
  WriteBinary(MakeOutputName(output_template, mesh_id, 5), noisy_vertex_cache);
  WriteBinary(MakeOutputName(output_template, mesh_id, 6), clean_vertex_cache);
}

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program
            << " <profile.txt> <lsd_size> <patch_face_count> <axis_excluded_vertex>\n"
            << "Example: " << program << " train_profile.txt 20 240 2\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 5) {
      PrintUsage(argv[0]);
      return 1;
    }

    const std::string profile_name = argv[1];
    g_lsd_size = std::atoi(argv[2]);
    g_patch_face_count = std::atoi(argv[3]);
    g_axis_excluded_vertex = std::atoi(argv[4]);
    if (g_lsd_size <= 0) throw std::runtime_error("lsd_size must be positive.");
    if (g_patch_face_count <= 0) throw std::runtime_error("patch_face_count must be positive.");

    std::srand(static_cast<unsigned int>(std::time(NULL)));
    BuildSamplingGrid();

    std::cout << "profile=" << profile_name << ", lsd_size=" << g_lsd_size
              << ", patch_face_count=" << g_patch_face_count
              << ", axis_excluded_vertex=" << g_axis_excluded_vertex << std::endl;
    std::cout << "Generating data serially." << std::endl;

    int number_of_meshes = 0;
    std::string output_template;
    const std::vector<std::string> mesh_names =
        ReadProfile(profile_name, &number_of_meshes, &output_template);

    std::vector<MeshData> meshes;
    ReadMeshes(mesh_names, number_of_meshes, &meshes);
    PreprocessMeshes(&meshes);

    for (int mesh_id = 0; mesh_id < number_of_meshes; ++mesh_id) {
      GenerateOneMesh(mesh_id, &meshes[static_cast<size_t>(mesh_id)], output_template);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
