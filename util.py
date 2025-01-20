
import os
import numpy as np
from scipy.stats import poisson
import torch
import torch.nn as nn

def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    ckpt_lst = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if len(ckpt_lst) == 0:
        print("No checkpoint files found in the directory.")
        epoch = 0
        return net, optim, epoch
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch