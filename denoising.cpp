// SPDX-License-Identifier: MIT
//
// SurfaceFormer / Mesh Denoising Transformer inference driver.
//
// This program keeps the original command-line interface and intermediate
// binary file format of denoising.cpp:
//   ./denoising <profile.txt> <lsd_size> <patch_face_count> <axis_excluded_vertex>
//
// Profile format:
//   <number_of_models>
//   <model_path_0>
//   ...
//   <model_path_N-1>
//   <number_of_denoising_stages> <vertex_refinement_iterations>
//   <number_of_meshes>
//   <mesh_path_0>
//   ...
//   <mesh_path_M-1>
//
// For each denoising stage, the C++ code writes the following files in the
// current working directory, calls `python denoising.py ...`, and reads the
// predicted results back:
//   lsd.bin     float32 [F, lsd_size, lsd_size, 3]
//   xyz.bin     float32 [F, 3]
//   index.bin   int32   [num_patches, patch_face_count]
//   dr.bin      float32 [F, 3]
//   vp.bin      float32 [F, 9]
//   normal.bin  float32 [num_patches, patch_face_count, 3]  (written by Python)
//   pos.bin     float32 [num_patches, patch_face_count, 9]  (written by Python)
//
// Implementation notes for reproducibility:
// - Patch face-id generation intentionally preserves the legacy unordered_set
//   traversal and kd-tree fallback behavior. This is required to keep index.bin
//   identical to the original implementation.
// - The angle convention of the original LSD generation, atan2(dx, dy), is kept
//   unchanged even though atan2(dy, dx) is more common in Cartesian coordinates.
// - No OpenMP is used in this version.

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
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

// Runtime parameters. They are global to keep inner-loop code close to the
// legacy implementation while still making the program structure explicit.
int g_lsd_size = 20;
int g_patch_face_count = 240;
int g_axis_excluded_vertex = 2;

// Virtual Cartesian grid used to derive geodesic lengths and directions.
// entry = {i - lsd_size/2, j - lsd_size/2, dx^2 + dy^2}
std::vector<std::array<int, 3> > g_sampling_grid;

// -----------------------------------------------------------------------------
// Legacy kd-tree for nearest-face fallback during patch generation.
//
// IMPORTANT: index.bin depends on the exact ordering of the fallback faces. The
// original program used a global comparison axis and a global priority queue.
// They are intentionally kept here. Do not replace this with std::set, a modern
// kd-tree, or a custom tie-break unless you also accept changed patch ids.
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
    po.clear();
    pt.clear();
    son.clear();
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

struct RuntimeData {
  TriMesh mesh;
  std::vector<SegmentLine> halfedge_segments;
  std::vector<TriMesh::Normal> noisy_normals;
  std::vector<TriMesh::Normal> filtered_normals;
  std::vector<TriMesh::Point> face_centroids;
  KDTree centroid_tree;
  double sigma = 0.0;
};

struct BinaryInputs {
  std::vector<float> lsd;       // [F, lsd_size, lsd_size, 3]
  std::vector<float> xyz;       // [F, 3]
  std::vector<int> index;       // [num_patches, patch_face_count]
  std::vector<float> direction; // [F, 3]
  std::vector<float> vertices;  // [F, 9]
  std::vector<int> face_vertices; // [F, 3]
  int num_patches = 0;
};

// -----------------------------------------------------------------------------
// File helpers
// -----------------------------------------------------------------------------
template <typename T>
void WriteBinaryFile(const std::string& filename, const std::vector<T>& data,
                     size_t count) {
  FILE* fp = std::fopen(filename.c_str(), "wb");
  if (!fp) {
    throw std::runtime_error("Failed to open " + filename + " for writing: " +
                             std::strerror(errno));
  }
  const size_t written = std::fwrite(data.data(), sizeof(T), count, fp);
  if (written != count) {
    std::fclose(fp);
    throw std::runtime_error("Failed to write complete file: " + filename);
  }
  std::fclose(fp);
}

template <typename T>
void ReadBinaryFile(const std::string& filename, std::vector<T>* data,
                    size_t count) {
  data->assign(count, T());
  FILE* fp = std::fopen(filename.c_str(), "rb");
  if (!fp) {
    throw std::runtime_error("Failed to open " + filename + " for reading: " +
                             std::strerror(errno));
  }
  const size_t read = std::fread(data->data(), sizeof(T), count, fp);
  if (read != count) {
    std::fclose(fp);
    throw std::runtime_error("Failed to read complete file: " + filename);
  }
  std::fclose(fp);
}

// Minimal shell quoting so model paths containing spaces do not break the
// Python call. The command name and script name are preserved as in the legacy
// code: `python denoising.py ...`.
std::string ShellQuote(const std::string& s) {
  std::string out = "'";
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\'') {
      out += "'\\''";
    } else {
      out += s[i];
    }
  }
  out += "'";
  return out;
}

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

double ComputeSigma(double multiple, const std::vector<TriMesh::Point>& centroids,
                    TriMesh& mesh) {
  double sigma = 0.0;
  double count = 0.0;
  for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
    const TriMesh::Point& ci = centroids[f_it->idx()];
    for (TriMesh::FaceFaceIter ff_it = mesh.ff_iter(*f_it); ff_it.is_valid(); ++ff_it) {
      const TriMesh::Point& cj = centroids[ff_it->idx()];
      sigma += (cj - ci).length();
      count += 1.0;
    }
  }
  if (count == 0.0) return 0.0;
  return sigma * multiple / count;
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

void Preprocess(RuntimeData* data) {
  RuntimeData& d = *data;
  d.halfedge_segments.clear();
  d.noisy_normals.clear();
  d.filtered_normals.clear();
  d.face_centroids.clear();

  BuildHalfedgeSegments(d.mesh, &d.halfedge_segments);
  ComputeFaceNormals(d.mesh, &d.noisy_normals);
  ComputeFaceCentroids(d.mesh, &d.face_centroids);
  d.filtered_normals.assign(d.mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));

  // Keep the original scale rule: getSigmaS(2)/8.
  d.sigma = ComputeSigma(2.0, d.face_centroids, d.mesh) / 8.0;

  d.centroid_tree.init(static_cast<int>(d.mesh.n_faces()));
  for (int f = 0; f < static_cast<int>(d.mesh.n_faces()); ++f) {
    d.centroid_tree.po[f].x[0] = d.face_centroids[f][0];
    d.centroid_tree.po[f].x[1] = d.face_centroids[f][1];
    d.centroid_tree.po[f].x[2] = d.face_centroids[f][2];
    d.centroid_tree.po[f].index = f;
  }
  if (d.mesh.n_faces() > 0) {
    d.centroid_tree.build(0, static_cast<int>(d.mesh.n_faces()) - 1);
  }
}

// -----------------------------------------------------------------------------
// Intersection and LSD generation
// -----------------------------------------------------------------------------
bool CalculateLineLineIntersection(const TriMesh::Point& line1Point1,
                                   const TriMesh::Point& line1Point2,
                                   const TriMesh::Point& line2Point1,
                                   const TriMesh::Point& line2Point2,
                                   TriMesh::Point* resultSegmentPoint,
                                   TriMesh::Normal* nowNormal) {
  TriMesh::Point p1 = line1Point1;
  TriMesh::Point p2 = line1Point2;
  TriMesh::Point p3 = line2Point1;
  TriMesh::Point p4 = line2Point2;
  TriMesh::Point p13 = p1 - p3;
  TriMesh::Point p43 = p4 - p3;

  if (p43.length() < 1e-8) return false;
  TriMesh::Point p21 = p2 - p1;
  if (p21.length() < 1e-8) return false;

  const double d1343 = p13.data()[0] * static_cast<double>(p43.data()[0]) +
                       p13.data()[1] * p43.data()[1] +
                       p13.data()[2] * p43.data()[2];
  const double d4321 = p43.data()[0] * static_cast<double>(p21.data()[0]) +
                       p43.data()[1] * p21.data()[1] +
                       p43.data()[2] * p21.data()[2];
  const double d1321 = p13.data()[0] * static_cast<double>(p21.data()[0]) +
                       p13.data()[1] * p21.data()[1] +
                       p13.data()[2] * p21.data()[2];
  const double d4343 = p43.data()[0] * static_cast<double>(p43.data()[0]) +
                       p43.data()[1] * p43.data()[1] +
                       p43.data()[2] * p43.data()[2];
  const double d2121 = p21.data()[0] * static_cast<double>(p21.data()[0]) +
                       p21.data()[1] * p21.data()[1] +
                       p21.data()[2] * p21.data()[2];

  const double denom = d2121 * d4343 - d4321 * d4321;
  if (denom == 0.0) return false;
  const double numer = d1343 * d4321 - d1321 * d4343;

  const double mua = numer / denom;
  const double mub = (d1343 + d4321 * mua) / d4343;

  TriMesh::Point result1(p1.data()[0] + mua * p21.data()[0],
                         p1.data()[1] + mua * p21.data()[1],
                         p1.data()[2] + mua * p21.data()[2]);
  TriMesh::Point result2(p3.data()[0] + mub * p43.data()[0],
                         p3.data()[1] + mub * p43.data()[1],
                         p3.data()[2] + mub * p43.data()[2]);

  if ((result2 - result1).length() < 1e-6 && mua >= -1e-6 &&
      mua <= 1.0 + 1e-6 && mub >= 0.0) {
    if (mua > 1.0 - 1e-6) {
      result1 = TriMesh::Point(p1.data()[0] + 0.9999 * p21.data()[0],
                               p1.data()[1] + 0.9999 * p21.data()[1],
                               p1.data()[2] + 0.9999 * p21.data()[2]);
      *nowNormal = result1 - line2Point1;
      nowNormal->normalize();
    }
    if (mua < 1e-6) {
      result1 = TriMesh::Point(p1.data()[0] + 0.0001 * p21.data()[0],
                               p1.data()[1] + 0.0001 * p21.data()[1],
                               p1.data()[2] + 0.0001 * p21.data()[2]);
      *nowNormal = result1 - line2Point1;
      nowNormal->normalize();
    }
    *resultSegmentPoint = result1;
    return true;
  }
  return false;
}

void GenerateLSDForFace(int face_index, RuntimeData* data, float* output,
                        float* direction) {
  RuntimeData& d = *data;

  // The polar axis is the vector from face centroid to the midpoint of two
  // vertices. By default vertex #2 is excluded, matching the paper/code setting.
  TriMesh::Point start_point(0.0, 0.0, 0.0);
  int cc = 0;
  int excluded_vertex = 0;
  if (g_axis_excluded_vertex > 2 || g_axis_excluded_vertex < 0) {
    excluded_vertex = std::rand() % 3;
  } else {
    excluded_vertex = g_axis_excluded_vertex;
  }

  for (TriMesh::FaceVertexIter it = d.mesh.fv_begin(TriMesh::FaceHandle(face_index));
       cc <= 2; ++cc, ++it) {
    if (cc != excluded_vertex) start_point += d.mesh.point(*it);
  }
  start_point /= 2.0;
  TriMesh::Normal start_normal = start_point - d.face_centroids[face_index];
  start_normal.normalize();

  direction[0] = static_cast<float>(start_normal.data()[0]);
  direction[1] = static_cast<float>(start_normal.data()[1]);
  direction[2] = static_cast<float>(start_normal.data()[2]);

  for (int i = 0; i < g_lsd_size; ++i) {
    for (int j = 0; j < g_lsd_size; ++j) {
      const std::array<int, 3>& grid =
          g_sampling_grid[static_cast<size_t>(i * g_lsd_size + j)];
      const int dx = grid[0];
      const int dy = grid[1];
      const int r2 = grid[2];

      const double target_length = d.sigma * std::sqrt(static_cast<double>(r2));
      double current_length = 0.0;

      TriMesh::Point current_point = d.face_centroids[face_index];
      TriMesh::Normal current_dir = start_normal;

      // Preserve the legacy angle convention: angle = atan2(dx, dy).
      if (dx == 0) {
        const double angle = (dy > 0) ? 0.0 : M_PI;
        Eigen::AngleAxisd rotation(angle,
                                   Eigen::Vector3d(d.noisy_normals[face_index].data()[0],
                                                   d.noisy_normals[face_index].data()[1],
                                                   d.noisy_normals[face_index].data()[2]));
        Eigen::Vector3d tmp(current_dir.data()[0], current_dir.data()[1], current_dir.data()[2]);
        tmp = rotation.matrix() * tmp;
        current_dir.data()[0] = tmp[0];
        current_dir.data()[1] = tmp[1];
        current_dir.data()[2] = tmp[2];
      } else {
        Eigen::AngleAxisd rotation(std::atan2(dx, dy),
                                   Eigen::Vector3d(d.noisy_normals[face_index].data()[0],
                                                   d.noisy_normals[face_index].data()[1],
                                                   d.noisy_normals[face_index].data()[2]));
        Eigen::Vector3d tmp(current_dir.data()[0], current_dir.data()[1], current_dir.data()[2]);
        tmp = rotation.matrix() * tmp;
        current_dir.data()[0] = tmp[0];
        current_dir.data()[1] = tmp[1];
        current_dir.data()[2] = tmp[2];
      }
      current_dir.normalize();

      const int offset = i * g_lsd_size * 3 + j * 3;
      OpenMesh::FaceHandle current_face(face_index);

      if (dx == 0 && dy == 0) {
        output[offset + 0] = static_cast<float>(d.noisy_normals[current_face.idx()].data()[0]);
        output[offset + 1] = static_cast<float>(d.noisy_normals[current_face.idx()].data()[1]);
        output[offset + 2] = static_cast<float>(d.noisy_normals[current_face.idx()].data()[2]);
        continue;
      }

      std::unordered_set<int> visited_faces;
      int previous_halfedge = -1;
      int end_flag = 0;

      while (end_flag == 0) {
        visited_faces.insert(current_face.idx());
        int edge_count = 0;
        int go_flag = 0;

        for (TriMesh::FaceHalfedgeIter it = d.mesh.fh_begin(current_face);
             it != d.mesh.fh_end(current_face); ++it) {
          TriMesh::Point next_point;
          const int current_halfedge = it->idx();

          if (current_halfedge != previous_halfedge) {
            edge_count++;
            TriMesh::Point far_point = current_point + current_dir * d.sigma * 100.0;

            if (CalculateLineLineIntersection(d.halfedge_segments[current_halfedge].v1,
                                              d.halfedge_segments[current_halfedge].v2,
                                              current_point, far_point, &next_point,
                                              &current_dir)) {
              go_flag = 1;
              current_length += (next_point - current_point).length();

              if (current_length >= target_length) {
                output[offset + 0] = static_cast<float>(d.noisy_normals[current_face.idx()].data()[0]);
                output[offset + 1] = static_cast<float>(d.noisy_normals[current_face.idx()].data()[1]);
                output[offset + 2] = static_cast<float>(d.noisy_normals[current_face.idx()].data()[2]);
                end_flag = -1;
                break;
              }

              current_point = next_point;
              previous_halfedge = d.mesh.opposite_halfedge_handle(*it).idx();
              OpenMesh::FaceHandle next_face =
                  d.mesh.face_handle(d.mesh.opposite_halfedge_handle(*it));

              if (next_face.idx() == -1) {  // boundary
                end_flag = -2;
                break;
              }
              if (visited_faces.find(next_face.idx()) != visited_faces.end()) {
                end_flag = -3;
                break;
              }

              Eigen::Matrix3d rotate_to_next(
                  Eigen::Quaterniond::FromTwoVectors(
                      Eigen::Vector3d(d.noisy_normals[current_face.idx()].data()[0],
                                      d.noisy_normals[current_face.idx()].data()[1],
                                      d.noisy_normals[current_face.idx()].data()[2]),
                      Eigen::Vector3d(d.noisy_normals[next_face.idx()].data()[0],
                                      d.noisy_normals[next_face.idx()].data()[1],
                                      d.noisy_normals[next_face.idx()].data()[2])));
              Eigen::Vector3d tmp(current_dir.data()[0], current_dir.data()[1], current_dir.data()[2]);
              tmp = rotate_to_next * tmp;
              current_dir.data()[0] = tmp[0];
              current_dir.data()[1] = tmp[1];
              current_dir.data()[2] = tmp[2];
              current_dir.normalize();

              current_face = next_face;
              break;
            }
          }

          if ((edge_count == 2 && go_flag == 0 && previous_halfedge != -1) ||
              (edge_count == 3 && go_flag == 0 && previous_halfedge == -1)) {
            std::printf("!");
            end_flag = -4;
            break;
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Patch generation. This is intentionally close to the original implementation
// because patch id ordering affects all later aggregation steps.
// -----------------------------------------------------------------------------
void GeneratePatchFaceList(int center_face, RuntimeData* data, int* face_list) {
  RuntimeData& d = *data;
  std::unordered_set<int> neighbor_face_index;
  int findflag = 1;

  face_list[0] = center_face;
  neighbor_face_index.insert(center_face);
  while (true) {
    findflag = 0;

    std::vector<int> cc;
    for (std::unordered_set<int>::iterator it = neighbor_face_index.begin();
         it != neighbor_face_index.end(); ++it) {
      cc.push_back(*it);
    }

    for (std::vector<int>::iterator it = cc.begin(); it != cc.end(); ++it) {
      for (TriMesh::FaceVertexIter fv_it = d.mesh.fv_begin(TriMesh::FaceHandle(*it));
           fv_it.is_valid(); ++fv_it) {
        for (TriMesh::VertexFaceIter vf_it = d.mesh.vf_iter(*fv_it); vf_it.is_valid(); ++vf_it) {
          if (neighbor_face_index.find(vf_it->idx()) == neighbor_face_index.end()) {
            findflag = 1;
            neighbor_face_index.insert(vf_it->idx());
            face_list[neighbor_face_index.size() - 1] = vf_it->idx();

            if (static_cast<int>(neighbor_face_index.size()) == g_patch_face_count) goto lab;
          }
        }
      }
    }
    if (findflag == 0) break;
  }

lab:;
  if (static_cast<int>(neighbor_face_index.size()) < g_patch_face_count) {
    std::printf("!");
    point ask;
    ask.x[0] = d.face_centroids[center_face][0];
    ask.x[1] = d.face_centroids[center_face][1];
    ask.x[2] = d.face_centroids[center_face][2];
    while (!nq.empty()) nq.pop();
    d.centroid_tree.query(ask, g_patch_face_count + 10);

    std::vector<tp> result;
    for (int cc = 0; !nq.empty(); cc++) {
      tp z = nq.top();
      result.push_back(z);
      nq.pop();
    }
    for (int cc = static_cast<int>(result.size()) - 1; cc >= 0; cc--) {
      if (neighbor_face_index.find(result[cc].second.index) == neighbor_face_index.end()) {
        neighbor_face_index.insert(result[cc].second.index);
        face_list[neighbor_face_index.size() - 1] = result[cc].second.index;
        if (static_cast<int>(neighbor_face_index.size()) == g_patch_face_count) break;
      }
    }
  }
}

static bool CompareDistanceOnly(const std::pair<float, int>& a,
                                const std::pair<float, int>& b) {
  return a.first < b.first;
}

int GeneratePatchIndexList(RuntimeData* data, std::vector<int>* index_list) {
  RuntimeData& d = *data;
  const int face_count = static_cast<int>(d.mesh.n_faces());

  std::vector<std::pair<float, int> > length_cache(face_count);
  std::vector<int> visit(face_count, 0);
  std::vector<int> mapping(face_count, 0);
  std::vector<float> mindist(face_count, 9999999999.0f);  // kept for legacy parity
  (void)mindist;

  for (int i = 0; i < face_count; ++i) {
    length_cache[i].second = i;
    length_cache[i].first =
        static_cast<float>(square(d.face_centroids[i][0] - d.face_centroids[0][0]) +
                           square(d.face_centroids[i][1] - d.face_centroids[0][1]) +
                           square(d.face_centroids[i][2] - d.face_centroids[0][2]));
  }
  std::sort(length_cache.begin(), length_cache.end(), CompareDistanceOnly);
  for (int i = 0; i < face_count; ++i) {
    mapping[length_cache[i].second] = i;
  }

  index_list->assign(static_cast<size_t>(face_count * g_patch_face_count), 0);
  int patch_count = 0;
  for (int i = 0; i < face_count; ++i) {
    if (visit[i] == 0) {
      GeneratePatchFaceList(length_cache[i].second, data,
                            &(*index_list)[static_cast<size_t>(patch_count * g_patch_face_count)]);
      for (int j = 0; j < g_patch_face_count; ++j) {
        visit[mapping[(*index_list)[static_cast<size_t>(patch_count * g_patch_face_count + j)]]] = 1;
      }
      patch_count++;
    }
  }
  return patch_count;
}

// -----------------------------------------------------------------------------
// Network input generation and output aggregation
// -----------------------------------------------------------------------------
void BuildNetworkInputs(RuntimeData* data, BinaryInputs* inputs) {
  RuntimeData& d = *data;
  const int face_count = static_cast<int>(d.mesh.n_faces());
  const int vertex_count = static_cast<int>(d.mesh.n_vertices());

  const size_t lsd_stride = static_cast<size_t>(g_lsd_size * g_lsd_size * 3);
  inputs->lsd.assign(static_cast<size_t>(face_count) * lsd_stride, 0.0f);
  inputs->xyz.assign(static_cast<size_t>(face_count) * 3, 0.0f);
  inputs->direction.assign(static_cast<size_t>(face_count) * 3, 0.0f);
  inputs->vertices.assign(static_cast<size_t>(face_count) * 9, 0.0f);
  inputs->face_vertices.assign(static_cast<size_t>(face_count) * 3, 0);

  inputs->num_patches = GeneratePatchIndexList(data, &inputs->index);

  for (int f = 0; f < face_count; ++f) {
    GenerateLSDForFace(f, data, &inputs->lsd[static_cast<size_t>(f) * lsd_stride],
                       &inputs->direction[static_cast<size_t>(f) * 3]);

    inputs->xyz[static_cast<size_t>(f) * 3 + 0] = static_cast<float>(d.face_centroids[f][0]);
    inputs->xyz[static_cast<size_t>(f) * 3 + 1] = static_cast<float>(d.face_centroids[f][1]);
    inputs->xyz[static_cast<size_t>(f) * 3 + 2] = static_cast<float>(d.face_centroids[f][2]);

    int count = 0;
    for (TriMesh::FaceVertexIter fv_it = d.mesh.fv_begin(TriMesh::FaceHandle(f));
         fv_it.is_valid(); ++fv_it) {
      const OpenMesh::VertexHandle vh = fv_it.handle();
      inputs->face_vertices[static_cast<size_t>(f) * 3 + count] = vh.idx();
      const TriMesh::Point p = d.mesh.point(vh);
      inputs->vertices[static_cast<size_t>(f) * 9 + count * 3 + 0] = static_cast<float>(p.data()[0]);
      inputs->vertices[static_cast<size_t>(f) * 9 + count * 3 + 1] = static_cast<float>(p.data()[1]);
      inputs->vertices[static_cast<size_t>(f) * 9 + count * 3 + 2] = static_cast<float>(p.data()[2]);
      count++;
    }

    // The program assumes triangular meshes. Keep a clear error message for
    // invalid inputs instead of silently writing partially initialized data.
    if (count != 3) {
      throw std::runtime_error("Input mesh contains a non-triangular face. Please triangulate it first.");
    }
  }

  (void)vertex_count;
}

void WriteNetworkInputs(const BinaryInputs& inputs, int face_count) {
  const size_t lsd_count = static_cast<size_t>(face_count) * g_lsd_size * g_lsd_size * 3;
  WriteBinaryFile("lsd.bin", inputs.lsd, lsd_count);
  WriteBinaryFile("xyz.bin", inputs.xyz, static_cast<size_t>(face_count) * 3);
  WriteBinaryFile("index.bin", inputs.index,
                  static_cast<size_t>(inputs.num_patches) * g_patch_face_count);
  WriteBinaryFile("dr.bin", inputs.direction, static_cast<size_t>(face_count) * 3);
  WriteBinaryFile("vp.bin", inputs.vertices, static_cast<size_t>(face_count) * 9);
}

void RunPythonDenoiser(const std::string& model_path) {
  std::string command = "python denoising.py ";
  command += ShellQuote(model_path);
  command += " ";
  command += std::to_string(g_lsd_size);
  command += " ";
  command += std::to_string(g_patch_face_count);

  std::printf("%s\n", command.c_str());
  const int ret = std::system(command.c_str());
  if (ret != 0) {
    throw std::runtime_error("Python denoising command failed with code " +
                             std::to_string(ret));
  }
}

void AggregatePredictedNormals(RuntimeData* data, const BinaryInputs& inputs) {
  RuntimeData& d = *data;
  const int face_count = static_cast<int>(d.mesh.n_faces());
  const size_t prediction_count = static_cast<size_t>(inputs.num_patches) *
                                  g_patch_face_count * 3;

  std::vector<float> normal_cache;
  ReadBinaryFile("normal.bin", &normal_cache, prediction_count);

  for (int f = 0; f < face_count; ++f) {
    d.filtered_normals[f][0] = 0.0;
    d.filtered_normals[f][1] = 0.0;
    d.filtered_normals[f][2] = 0.0;
  }

  const int total_patch_faces = inputs.num_patches * g_patch_face_count;
  for (int i = 0; i < total_patch_faces; ++i) {
    const int face_id = inputs.index[static_cast<size_t>(i)];
    d.filtered_normals[face_id][0] += normal_cache[static_cast<size_t>(i) * 3 + 0];
    d.filtered_normals[face_id][1] += normal_cache[static_cast<size_t>(i) * 3 + 1];
    d.filtered_normals[face_id][2] += normal_cache[static_cast<size_t>(i) * 3 + 2];
  }

  for (int f = 0; f < face_count; ++f) {
    d.filtered_normals[f].normalize();
  }
}

void AggregatePredictedVertices(RuntimeData* data, const BinaryInputs& inputs) {
  RuntimeData& d = *data;
  const int vertex_count = static_cast<int>(d.mesh.n_vertices());
  const size_t prediction_count = static_cast<size_t>(inputs.num_patches) *
                                  g_patch_face_count * 9;

  std::vector<float> pos_cache;
  ReadBinaryFile("pos.bin", &pos_cache, prediction_count);

  std::vector<float> pos_sum(static_cast<size_t>(vertex_count) * 3, 0.0f);
  std::vector<int> count(vertex_count, 0);

  const int total_patch_faces = inputs.num_patches * g_patch_face_count;
  for (int i = 0; i < total_patch_faces; ++i) {
    const int face_id = inputs.index[static_cast<size_t>(i)];
    for (int j = 0; j < 3; ++j) {
      const int vertex_id = inputs.face_vertices[static_cast<size_t>(face_id) * 3 + j];
      count[vertex_id]++;
      pos_sum[static_cast<size_t>(vertex_id) * 3 + 0] += pos_cache[static_cast<size_t>(i) * 9 + j * 3 + 0];
      pos_sum[static_cast<size_t>(vertex_id) * 3 + 1] += pos_cache[static_cast<size_t>(i) * 9 + j * 3 + 1];
      pos_sum[static_cast<size_t>(vertex_id) * 3 + 2] += pos_cache[static_cast<size_t>(i) * 9 + j * 3 + 2];
    }
  }

  for (int v = 0; v < vertex_count; ++v) {
    if (count[v] > 0) {
      pos_sum[static_cast<size_t>(v) * 3 + 0] /= count[v];
      pos_sum[static_cast<size_t>(v) * 3 + 1] /= count[v];
      pos_sum[static_cast<size_t>(v) * 3 + 2] /= count[v];
    } else {
      // Degenerate meshes with isolated vertices would divide by zero in the
      // legacy code. Preserve a sane position for such invalid input.
      const TriMesh::Point p = d.mesh.point(TriMesh::VertexHandle(v));
      pos_sum[static_cast<size_t>(v) * 3 + 0] = static_cast<float>(p[0]);
      pos_sum[static_cast<size_t>(v) * 3 + 1] = static_cast<float>(p[1]);
      pos_sum[static_cast<size_t>(v) * 3 + 2] = static_cast<float>(p[2]);
    }
  }

  for (TriMesh::VertexIter v_it = d.mesh.vertices_begin(); v_it != d.mesh.vertices_end(); ++v_it) {
    const int v = v_it->idx();
    d.mesh.set_point(*v_it, TriMesh::Point(pos_sum[static_cast<size_t>(v) * 3 + 0],
                                           pos_sum[static_cast<size_t>(v) * 3 + 1],
                                           pos_sum[static_cast<size_t>(v) * 3 + 2]));
  }
}

// Iterative vertex refinement that aligns vertex positions with predicted face
// normals. This is the same update rule as the original implementation.
void UpdateVertexPosition(TriMesh& mesh,
                          const std::vector<TriMesh::Normal>& filtered_normals,
                          int iteration_number, bool fixed_boundary) {
  std::vector<TriMesh::Point> new_points(mesh.n_vertices());
  std::vector<TriMesh::Point> centroids;

  for (int iter = 0; iter < iteration_number; ++iter) {
    ComputeFaceCentroids(mesh, &centroids);

    for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
      TriMesh::Point p = mesh.point(*v_it);
      if (fixed_boundary && mesh.is_boundary(*v_it)) {
        new_points[v_it->idx()] = p;
      } else {
        double face_num = 0.0;
        TriMesh::Point offset(0.0, 0.0, 0.0);
        for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
          const TriMesh::Normal n = filtered_normals[vf_it->idx()];
          const TriMesh::Point c = centroids[vf_it->idx()];
          offset += n * (n | (c - p));
          face_num += 1.0;
        }
        if (face_num > 0.0) p += offset / face_num;
        new_points[v_it->idx()] = p;
      }
    }

    for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
      mesh.set_point(*v_it, new_points[v_it->idx()]);
    }
  }
}

// Generate the denoised mesh filename. This preserves the legacy convention:
// `xxx.off` -> `xxx_01.off`, and `xxx_03.off` with stage 0 -> `xxx_04.off`.
std::string GenerateOutputFilename(const std::string& input, int stage_index) {
  const int len = static_cast<int>(input.length());
  if (len < 4) {
    throw std::runtime_error("Input mesh name is too short: " + input);
  }

  const std::string stem = input.substr(0, len - 4);
  if (len < 7) {
    std::string suffix = "_00.off";
    const int id = stage_index + 1;
    suffix[1] = static_cast<char>(id / 10 + '0');
    suffix[2] = static_cast<char>(id % 10 + '0');
    return stem + suffix;
  }

  std::string tag = input.substr(len - 4 - 3, 3);
  if (tag[0] == '_' && tag[1] >= '0' && tag[1] <= '9' && tag[2] >= '0' && tag[2] <= '9') {
    int id = (tag[1] - '0') * 10 + (tag[2] - '0');
    id += stage_index + 1;
    tag[1] = static_cast<char>(id / 10 + '0');
    tag[2] = static_cast<char>(id % 10 + '0');
    return input.substr(0, len - 7) + tag + ".off";
  }

  std::string suffix = "_00.off";
  const int id = stage_index + 1;
  suffix[1] = static_cast<char>(id / 10 + '0');
  suffix[2] = static_cast<char>(id % 10 + '0');
  return stem + suffix;
}

struct Profile {
  std::vector<std::string> model_paths;
  int denoising_stages = 0;
  int vertex_refinement_iterations = 0;
  std::vector<std::string> mesh_paths;
};

Profile ReadProfile(const std::string& filename) {
  std::ifstream in(filename.c_str());
  if (!in) {
    throw std::runtime_error("Failed to open profile: " + filename);
  }

  Profile profile;
  int model_count = 0;
  if (!(in >> model_count) || model_count <= 0) {
    throw std::runtime_error("Invalid model count in profile.");
  }
  profile.model_paths.resize(model_count);
  for (int i = 0; i < model_count; ++i) {
    if (!(in >> profile.model_paths[i])) {
      throw std::runtime_error("Failed to read model path from profile.");
    }
  }

  if (!(in >> profile.denoising_stages >> profile.vertex_refinement_iterations)) {
    throw std::runtime_error("Failed to read denoising stage count and vertex refinement iterations.");
  }
  if (profile.denoising_stages > model_count || profile.denoising_stages <= 0 ||
      profile.vertex_refinement_iterations <= 0) {
    throw std::runtime_error("iter number error");
  }

  int mesh_count = 0;
  if (!(in >> mesh_count) || mesh_count < 0) {
    throw std::runtime_error("Invalid mesh count in profile.");
  }
  profile.mesh_paths.resize(mesh_count);
  for (int i = 0; i < mesh_count; ++i) {
    if (!(in >> profile.mesh_paths[i])) {
      throw std::runtime_error("Failed to read mesh path from profile.");
    }
  }
  return profile;
}

void ProcessOneStage(RuntimeData* data, const std::string& model_path,
                     int vertex_refinement_iterations, const std::string& mesh_path,
                     int stage_index) {
  std::clock_t t1 = std::clock();

  Preprocess(data);
  const int face_count = static_cast<int>(data->mesh.n_faces());
  if (face_count <= 0) {
    throw std::runtime_error("Input mesh has no faces: " + mesh_path);
  }

  BinaryInputs inputs;
  BuildNetworkInputs(data, &inputs);
  WriteNetworkInputs(inputs, face_count);

  std::clock_t t2 = std::clock();
  RunPythonDenoiser(model_path);
  std::clock_t t3 = std::clock();

  AggregatePredictedNormals(data, inputs);
  AggregatePredictedVertices(data, inputs);
  UpdateVertexPosition(data->mesh, data->filtered_normals,
                       vertex_refinement_iterations, false);

  const std::string output_mesh = GenerateOutputFilename(mesh_path, stage_index);
  if (!OpenMesh::IO::write_mesh(data->mesh, output_mesh)) {
    throw std::runtime_error("Failed to write denoised mesh: " + output_mesh);
  }

  std::clock_t t4 = std::clock();
  std::cout << "LSD time = " << double(t2 - t1) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "denoising time = " << double(t3 - t2) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "Update time = " << double(t4 - t3) / CLOCKS_PER_SEC << "s" << std::endl;
}

int MainImpl(int argc, char* argv[]) {
  if (argc != 5) {
    std::printf("profile error\n");
    return 0;
  }

  std::srand(0);
  const std::string profile_path = argv[1];
  g_lsd_size = std::atoi(argv[2]);
  g_patch_face_count = std::atoi(argv[3]);
  g_axis_excluded_vertex = std::atoi(argv[4]);
  if (g_lsd_size <= 0 || g_patch_face_count <= 0) {
    throw std::runtime_error("lsd_size and patch_face_count must be positive.");
  }

  std::printf("%d %d %d\n", g_lsd_size, g_patch_face_count, g_axis_excluded_vertex);
  const Profile profile = ReadProfile(profile_path);
  BuildSamplingGrid();

  std::printf("read mesh\n");
  RuntimeData data;

  for (size_t mesh_id = 0; mesh_id < profile.mesh_paths.size(); ++mesh_id) {
    const std::string& mesh_path = profile.mesh_paths[mesh_id];
    std::printf("processing: ");
    std::printf("%s\n", mesh_path.c_str());

    data.mesh.clean();
    if (!OpenMesh::IO::read_mesh(data.mesh, mesh_path)) {
      throw std::runtime_error("data error: failed to read mesh " + mesh_path);
    }

    for (int stage = 0; stage < profile.denoising_stages; ++stage) {
      ProcessOneStage(&data, profile.model_paths[stage],
                      profile.vertex_refinement_iterations, mesh_path, stage);
    }
    data.mesh.clean();
  }
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return MainImpl(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
