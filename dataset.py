import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from util import *
from preprocessing import data_preprocessing

class SpectrogramDatasetWithHarmonicMask(Dataset):
    def __init__(self, acc, mic, noisy, harmonic_masks):
        self.acc = acc
        self.mic = mic
        self.noisy = noisy
        self.harmonic_masks = harmonic_masks

    def __len__(self):
        return len(self.acc)

    def __getitem__(self, idx):
        acc = self.acc[idx]
        mic = self.mic[idx]
        noisy = self.noisy[idx]
        harmonic_mask = self.harmonic_masks[idx]
        input_data = torch.cat((acc.unsqueeze(0), noisy.unsqueeze(0), harmonic_mask.unsqueeze(0)), dim=0)
        sample = {'input': input_data, 'label': mic}
        return sample