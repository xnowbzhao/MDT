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
from fileloader import Loader
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
    print(device)
    np.random.seed(0)
    # Set t0
    t0 = time.time()
    r=sys.argv
    patchsize=int(r[1])
    patchnumber=int(r[2])
    batch_size=int(r[3])
    print(patchsize,patchnumber,batch_size)
    
    model = config.get_model(device, patchsize, 512,12,12)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, device=device)

    out_dir = 'out/s/'
    logfile = open('out/s/log.txt','a')
    
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    try:
        load_dict = checkpoint_io.load('model_70000.pt')
        epoch_it = 7
        it = 70000
    except FileExistsError:
        load_dict = dict()
        epoch_it = 0
        it = 0

    metric_val_best = np.inf

    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    nparameters = sum(p.numel() for p in model.parameters())

    logfile.write('Total number of parameters: %d\n' % nparameters)

    train_dataset=Loader("/home/xnozero/Desktop/mdt/train-re/02t00t00.bin",batch_size, 63, 3000000,patchsize,patchnumber, True)
    #train_dataset=Loader("/media/xnozero/d/mit/train/02t00t00.bin",batch_size, 71, 2600000,patchsize,patchnumber, True)
    #train_dataset=Loader("/media/xnozero/d/mit/train/02t00t00.bin",batch_size, 72, 900000,patchsize,patchnumber, True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)


    for epoch_it in range(0,30):
        logfile.flush()
        for batch in train_loader:
            loss = trainer.train_step(batch)
            logger.add_scalar('train/loss', loss, it)
            if it%1000==0:
                print('[Epoch %02d] it=%02d: loss=%.6f' % (epoch_it, it, loss))
                logfile.write('[Epoch %02d] it=%02d: loss=%.6f\n' % (epoch_it, it, loss))
            it+=1
            if it % 10000 ==0:           
                logfile.write('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, loss_val_best=metric_val_best)
                checkpoint_io.save('model_'+str(it)+'.pt', epoch_it=epoch_it, loss_val_best=metric_val_best)
                logfile.flush()
    logger.close()

