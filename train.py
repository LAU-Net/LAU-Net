import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from preprocessing import data_preprocessing_train
from sklearn.model_selection import train_test_split
import logging
import time
from torchinfo import summary
import random
from model import *
from dataset import *
from util import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("test.log"),
    logging.StreamHandler()
])

parser = argparse.ArgumentParser(description="Lightweight Accelerometer-Assisted U-Net",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=200, type=int, dest="num_epoch")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--nch", default=2, type=int, dest="nch") 
parser.add_argument("--nker", default=4, type=int, dest="nker") 
parser.add_argument("--network", default="LAUNet", choices=["LAUNet"], type=str, dest="network")
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
ckpt_dir = args.ckpt_dir
nch = args.nch
nker = args.nker
network = args.network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)


dataset_dir = './datasets/training'
acc_seg_data, mic_seg_data, noisy_seg_data = data_preprocessing_train(dataset_dir=dataset_dir)

acc_seg_data = (acc_seg_data - np.min(acc_seg_data)) / (np.max(acc_seg_data) - np.min(acc_seg_data))
mic_seg_data = (mic_seg_data - np.min(mic_seg_data)) / (np.max(mic_seg_data) - np.min(mic_seg_data))
noisy_seg_data = (noisy_seg_data - np.min(noisy_seg_data)) / (np.max(noisy_seg_data) - np.min(noisy_seg_data))

harmonic_block = HarmonicEstimation()
harmonic_masks = []
for i in range(acc_seg_data.shape[0]):
    x = torch.tensor(acc_seg_data[i], dtype=torch.float32)
    harmonic_mask = harmonic_block.harmonic_estimation(x, max_power_input = 0.5)
    harmonic_masks.append(harmonic_mask.numpy())
harmonic_masks = np.array(harmonic_masks)

acc_train = acc_seg_data
mic_train = mic_seg_data
noisy_train = noisy_seg_data
harmonic_train = harmonic_masks

acc_train = torch.tensor(np.array(acc_train), dtype=torch.float32)
mic_train = torch.tensor(np.array(mic_train), dtype=torch.float32)
noisy_train = torch.tensor(np.array(noisy_train), dtype=torch.float32)
harmonic_train = torch.tensor(np.array(harmonic_train), dtype=torch.float32)

train_dataset = SpectrogramDatasetWithHarmonicMask(acc_train, mic_train, noisy_train, harmonic_train)
loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
num_data_train = len(train_dataset)
num_batch_train = np.ceil(num_data_train / batch_size)

net = LAUNet(nch=nch, nker=nker).to(device)

fn_loss = nn.L1Loss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)


st_epoch = 0
for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_log = []
    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        label = label.unsqueeze(1)
        input = data['input'].to(device)

        output = net(input)

        optim.zero_grad()

        harmonic_mask = input[:, -1:, :, :] 
        loss_magnitude = fn_loss(output, label)
        loss_harmonic = fn_loss(output * harmonic_mask, label * harmonic_mask)
        loss = 0.5*loss_magnitude + 0.5*loss_harmonic

        loss.backward()
        optim.step()
        loss_log += [loss.item()]

        logging.info("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                (epoch, num_epoch, batch, num_batch_train, np.mean(loss_log)))

    if epoch % 10 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
