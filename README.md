# MDT

Mesh denoising transformer

## Environment

```
openmesh 8.1
eigen
tqdm
numpy
pyyaml
tensorboardX
einops
```

## Compilation

Run the following command to compiling transformergdata.cpp and denoising.cpp, you may need to manually modify the path of OpenMesh-8.1 and Eigen.

```
g++ -std=c++11 transformergdata.cpp -O2 -o TF-Gdata -lOpenMeshCore -lOpenMeshTools
g++ -std=c++11 denoising.cpp -O2 -o TF-denoising -lOpenMeshCore -lOpenMeshTools
```


## Training and Evaluation 

We take the synthetic dataset as an example to illustrate the training process and evaluation.

1, Download the dataset from [CNR](https://wang-ps.github.io/denoising.html), move the files as follow:

```
Synthetic\train\noisy -> strain
Synthetic\train\original -> strain
Synthetic\test\noisy -> stest
```

2, create folder "train", and run the following commands for building training data.

```
./TF-Gdata profile/s_i1/train.txt 20 240 2
```

3, create folder "out/s", and train with the dataset to obtain the model: 

```
python train.py 20 240 80
```

4, evaluation:

```
./TF-denoising profile/s_i1/test.txt 20 240 2
```

## Pretrained models

The models have been converted for single-GPU inference:
[https://1drv.ms/f/c/1345fa17d336f6c3/IgDmdh7vfjrTS4AzLi3lMsCwAS6tQRaHgN7d3-HDbe58NkU?e=XmRL0G]

## Acknowledgement
The code is based on [GNF](https://github.com/bldeng/GuidedDenoising), If you use any of this code, please make sure to cite these works.

