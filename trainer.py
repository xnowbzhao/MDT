import os
from tqdm import tqdm
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
import numpy


def angle_loss(x1,x2):
    batchsize, hidden_size=x1

class Trainer():
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''



    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, batch):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()

        loss= self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        val=0.0
        num=0
        for batch in tqdm(val_loader):
            loss = self.eval_step(batch)
            val=val+torch.mean(loss).float()
            num=num+1
        return val/num
    
    def eval_step(self, batch):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        with torch.no_grad():
            loss = self.model.compute_loss(batch)
        return  loss

    def compute_loss(self, batch):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        output1,output2, = self.model.pred(batch)
        output1 = output1.float()
        output2 = output2.float()

        label1=batch.get('gt').to(self.device).float()
        label2=batch.get('gtvp').to(self.device).float()

        loss_fn1 = torch.nn.L1Loss()
        loss1 = loss_fn1(output1, label1)

        loss_fn2 = torch.nn.L1Loss()
        loss2 = loss_fn2(output2, label2)
        #print(loss1, loss2)
        return loss1.float()+ loss2.float() 



    
