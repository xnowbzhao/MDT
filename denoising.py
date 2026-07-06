import torch
import torch.optim as optim
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tensorboardX import SummaryWriter
import argparse
import time
import config
from trainer import Trainer
from checkpoints import CheckpointIO
from fileloader import Loader2
import pickle
import sys
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)
if __name__ == '__main__':  
# Arguments

    is_cuda = (torch.cuda.is_available() )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r=sys.argv
    patchsize=int(r[2])
    patchnumber=int(r[3])
    print(patchsize,patchnumber)

    t1=time.time()
    model = config.get_model(device, patchsize, 512,12,12)
    
    model.eval()


    
    checkpoint_io = CheckpointIO(sys.argv[1], model=model)
    try:
        load_dict = checkpoint_io.load('model_100000.pt')
    except FileExistsError:
        print("error")
        input()
    
    test_dataset=Loader2(patchsize,patchnumber)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, num_workers=10, shuffle=False)

    output1=[]
    output2=[]

    for batch in test_loader:
        with torch.no_grad():
            c,v = model.pred(batch)
        
        temp1=c.detach().cpu().numpy()
        temp2=v.detach().cpu().numpy()
        temp2=(temp2+batch.get('vp').detach().cpu().numpy()).astype(np.float32)
        print(temp1.shape, temp2.shape)       
        temp2=temp2.reshape(temp2.shape[0], -1, 3);

        rmat=batch.get('rmat').detach().cpu().numpy()
        scale=batch.get('scale').detach().cpu().numpy()
        zeropoint=batch.get('zeropoint').detach().cpu().numpy()

        temp1 = np.einsum('nij,nmj->nmi', rmat, temp1).astype(np.float32); 

        scale_expanded = scale[:, np.newaxis, np.newaxis]  
        temp2 = temp2 * scale_expanded
        temp2 = np.einsum('nij,nmj->nmi', rmat, temp2)
        zeropoint_expanded = zeropoint[:, np.newaxis, :]  
        temp2 = temp2 + zeropoint_expanded


        temp2=temp2.reshape((temp2.shape[0], -1, 9)).astype(np.float32); 

        output1.append(temp1)
        output2.append(temp2)

    output1=np.concatenate(output1,axis=0)
    output1=output1.reshape(-1)
    output1.tofile("normal.bin")

    output2=np.concatenate(output2,axis=0)
    output2=output2.reshape(-1)
    output2.tofile("pos.bin")




