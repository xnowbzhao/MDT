import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

class ResUnit(nn.Module):
    def __init__(self, channel):
        super(ResUnit,self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = residual+out
        out = self.relu(out)
        
        return out
def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)

    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

    feature = x.view(batch_size*num_points, -1)[idx, :]

    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()

        self.layer1 = self.make_layer(in_channel=3,  channel = 32, block=6)
        self.avgpool = nn.AvgPool2d(9)
        

    def make_layer(self, in_channel, channel, block):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=channel,kernel_size=3,stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU()))
        for i in range(1, block):
            layers.append(ResUnit(channel))
        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.layer1(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FeedForward_adj(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        qkv = self.to_qkv(x)

        qkv=qkv.chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx
def cos_sim(x1, x2):
    scores = torch.acos(torch.cosine_similarity(x1, x2, dim=1))/np.pi*180
    return scores.mean()


class DGCNN_cls(nn.Module):
    def __init__(self, k, dim):
        super(DGCNN_cls, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim*2)
        self.bn4 = nn.BatchNorm2d(dim*4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv1 = nn.Sequential(nn.Conv2d(30, dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(dim*2, dim*2, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(dim*2*2, dim*4, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))


    def forward(self, x):
        batch_size = x.size(0)
        x=x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class ViT2(nn.Module):
    def __init__(self, *, patchsize, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        patch_dim = channels * patchsize * patchsize

        self.to_patch_embedding =  nn.Conv2d(3,dim,kernel_size=patchsize,stride=patchsize)
        self.dgcnn_embedding = DGCNN_cls(20,dim//8)
        self.mix_embedding1 = nn.Linear(dim+32, dim)
        self.mix_embedding2 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(emb_dropout)
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.Res = ResNet()
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,3)
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,9)
        )

    def forward(self, d2, d3, d4, d5):
        s2=d2.shape
        s3=d3.shape   
        d2=d2.reshape((-1,s2[2],s2[3],s2[4]))
        d2=d2.permute(0,3,1,2)
        cnn_r=self.Res(d2)
        cnn_r= cnn_r.reshape((s2[0],s2[1],-1))       
        x1 = self.to_patch_embedding(d2)
        x1= x1.reshape((s2[0],s2[1],-1))

        x1=torch.cat((x1,cnn_r),2)        
        x1=self.mix_embedding1(x1)


        x2=torch.cat((d3,d4,d5),2)
        x2=self.dgcnn_embedding(x2)
        x2=x2.permute(0, 2, 1)
        x2=self.mix_embedding2(x2)
        
        x1=x1+x2
        x1 = self.dropout1(x1)
        x1 = self.transformer1(x1)

        nomf = self.mlp_head1(x1)
        nomf = nn.functional.normalize(nomf,dim=2)

        v3 = self.mlp_head2(x1)

        return nomf, v3


class Network(nn.Module):

    def __init__(self, predictor, device):
        super().__init__()
        self.predictor = predictor.to(device)
        self._device = device

    def compute_loss(self, batch):
        lab =batch.get('gt').to(self._device).float()
        output = self.pred(batch)

        loss = torch.nn.functional.cosine_similarity(output, lab, dim=2)

        loss2=torch.acos(loss)*180.0/np.pi
        isnan=torch.isnan(loss2)
        for i in range(isnan.shape[0]):
            for j in range(isnan.shape[1]):
                if isnan[i][j]==True:
                    loss2[i][j]=0

        return loss2

    def pred(self, batch):
        lsd =batch.get('lsd').to(self._device).float()
        xyz =batch.get('xyz').to(self._device).float()
        dr =batch.get('dr').to(self._device).float()
        vp =batch.get('vp').to(self._device).float()
        ret1, ret2= self.predictor(lsd, xyz, dr, vp)
        return ret1, ret2


