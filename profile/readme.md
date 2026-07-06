# The structure of s_i1/train.txt 
```
63                      <- Number of the ground truth meshes and noisy meshes
strain/bumpy_sphere.obj         <- Path of GT mesh
strain/bumpy_sphere.obj
strain/bumpy_sphere.obj
strain/bunny.obj
strain/bunny.obj
strain/bunny.obj
...
strain/bunny_n1.obj    <- Path of noisy mesh
strain/bunny_n2.obj
strain/bunny_n3.obj
strain/bust_n1.obj
strain/bust_n2.obj
strain/bust_n3.obj
...
train/02t00t00.bin     <- Path and name of LSD files


# The structure of s_i1/test.txt
```
1                      <- Number of models (not used)
out/s-re               <- Path of model
1 60                   <- Number of updates for face normals (not used) and vertex positions
87                     <- Number of noisy meshes
stest/armadillo_n1.obj <- Path of noisy mesh
stest/armadillo_n2.obj
stest/armadillo_n3.obj
stest/block_n1.obj
stest/block_n2.obj
stest/block_n3.obj
...

```

