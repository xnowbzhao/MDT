import yaml

import torch
from torchvision import transforms
import torch.nn as nn
import coder

# General config
# Datasets

def get_model(device, patchsize_, dim_,depth_, heads_):

    predictor = coder.ViT2(
    patchsize=patchsize_, 
    dim = dim_,
    depth = depth_,
    heads = heads_,
    mlp_dim = dim_*4,
    dropout = 0.1,
    emb_dropout = 0.1).to(device)

    model = coder.Network(predictor, device=device)
    return model
