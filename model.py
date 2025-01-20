import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa.display
import matplotlib.pyplot as plt
import time

class CBR2d_HarmonicAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)
        
        self.norm = None
        if not norm is None:
            if norm == "bnorm":
                self.norm = nn.BatchNorm2d(num_features=out_channels)
            elif norm == "inorm":
                self.norm = nn.InstanceNorm2d(num_features=out_channels)

        self.scale_factor = nn.Parameter(torch.tensor(1.0)) 
        self.alpha = nn.Parameter(torch.tensor(0.5))       
        self.channel_scale = nn.Parameter(torch.ones(out_channels))  
        self.channel_alpha = nn.Parameter(torch.ones(out_channels))

        self.activation = None
        if relu is not None and relu >= 0.0:
            self.activation = nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)

    def forward(self, x, harmonic_mask):
        x = self.conv(x)
        x = self.norm(x)
        harmonic_mask = torch.sigmoid(harmonic_mask) 
        harmonic_mask = harmonic_mask.expand(-1, x.shape[1], -1, -1)  
        x_HA = x * harmonic_mask * self.channel_scale.view(1, -1, 1, 1) * self.scale_factor
        x = x + self.channel_alpha.view(1, -1, 1, 1) * x_HA
        x = self.activation(x)
        return x


class LAUNet(nn.Module):
    def __init__(self, nch, nker=4, norm="bnorm"):
        super(LAUNet, self).__init__()

        self.enc1_1 = CBR2d_HarmonicAttention(in_channels=nch, out_channels=1 * nker, norm=norm)
        self.enc1_2 = CBR2d_HarmonicAttention(in_channels=1 * nker, out_channels=1 * nker, norm=norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2_1 = CBR2d_HarmonicAttention(in_channels=nker, out_channels=2 * nker, norm=norm)
        self.enc2_2 = CBR2d_HarmonicAttention(in_channels=2 * nker, out_channels=2 * nker, norm=norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3_1 = CBR2d_HarmonicAttention(in_channels=2 * nker, out_channels=4 * nker, norm=norm)
        self.enc3_2 = CBR2d_HarmonicAttention(in_channels=4 * nker, out_channels=4 * nker, norm=norm)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc4_1 = CBR2d_HarmonicAttention(in_channels=4 * nker, out_channels=8 * nker, norm=norm)
        self.enc4_2 = CBR2d_HarmonicAttention(in_channels=8 * nker, out_channels=8 * nker, norm=norm)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.enc5_1 = CBR2d_HarmonicAttention(in_channels=8 * nker, out_channels=16 * nker, norm=norm)

        self.dec5_1 = CBR2d_HarmonicAttention(in_channels=16 * nker, out_channels=8 * nker, norm=norm)
        self.unpool4 = nn.ConvTranspose2d(in_channels=8 * nker, out_channels=8 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = CBR2d_HarmonicAttention(in_channels=2 * 8 * nker, out_channels=8 * nker, norm=norm)
        self.dec4_1 = CBR2d_HarmonicAttention(in_channels=8 * nker, out_channels=4 * nker, norm=norm)
        self.unpool3 = nn.ConvTranspose2d(in_channels=4 * nker, out_channels=4 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = CBR2d_HarmonicAttention(in_channels=2 * 4 * nker, out_channels=4 * nker, norm=norm)
        self.dec3_1 = CBR2d_HarmonicAttention(in_channels=4 * nker, out_channels=2 * nker, norm=norm)
        self.unpool2 = nn.ConvTranspose2d(in_channels=2 * nker, out_channels=2 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d_HarmonicAttention(in_channels=2 * 2 * nker, out_channels=2 * nker, norm=norm)
        self.dec2_1 = CBR2d_HarmonicAttention(in_channels=2 * nker, out_channels=1 * nker, norm=norm)
        self.unpool1 = nn.ConvTranspose2d(in_channels=1 * nker, out_channels=1 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = CBR2d_HarmonicAttention(in_channels=2 * 1 * nker, out_channels=1 * nker, norm=norm)
        self.dec1_1 = CBR2d_HarmonicAttention(in_channels=1 * nker, out_channels=1 * nker, norm=norm)
        self.fc = nn.Conv2d(in_channels=1 * nker, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        input_data = x[:, :2, :, :]
        harmonic_mask = x[:, 2, :, :].unsqueeze(1)

        enc1_1 = self.enc1_1(input_data, harmonic_mask)
        enc1_2 = self.enc1_2(enc1_1, harmonic_mask)
        pool1 = self.pool1(enc1_2)
        pool1_harmonic_mask = self.pool1(harmonic_mask)
        enc2_1 = self.enc2_1(pool1, pool1_harmonic_mask)
        enc2_2 = self.enc2_2(enc2_1, pool1_harmonic_mask)
        pool2 = self.pool2(enc2_2)
        pool2_harmonic_mask = self.pool2(pool1_harmonic_mask)
        enc3_1 = self.enc3_1(pool2, pool2_harmonic_mask)
        enc3_2 = self.enc3_2(enc3_1, pool2_harmonic_mask)
        pool3 = self.pool3(enc3_2)
        pool3_harmonic_mask = self.pool3(pool2_harmonic_mask)
        enc4_1 = self.enc4_1(pool3, pool3_harmonic_mask)
        enc4_2 = self.enc4_2(enc4_1, pool3_harmonic_mask)
        pool4 = self.pool4(enc4_2)
        pool4_harmonic_mask = self.pool4(pool3_harmonic_mask)
        enc5_1 = self.enc5_1(pool4, pool4_harmonic_mask)
        
        dec5_1 = self.dec5_1(enc5_1, pool4_harmonic_mask)
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) 
        dec4_2 = self.dec4_2(cat4, pool3_harmonic_mask)
        dec4_1 = self.dec4_1(dec4_2, pool3_harmonic_mask)
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3, pool2_harmonic_mask)
        dec3_1 = self.dec3_1(dec3_2, pool2_harmonic_mask)
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2, pool1_harmonic_mask)
        dec2_1 = self.dec2_1(dec2_2, pool1_harmonic_mask)
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1, harmonic_mask)
        dec1_1 = self.dec1_1(dec1_2, harmonic_mask)

        x = self.fc(dec1_1)
        return x
    

class HarmonicEstimation(nn.Module):
    def __init__(self, peak_distance=2, max_peaks=5, freq_margin=3, max_power=0.1):
        super().__init__()
        self.peak_distance = peak_distance
        self.max_peaks = max_peaks
        self.freq_margin = freq_margin
        self.max_power = max_power

    def extract_f0(self, magnitudes, max_power_input):
        self.max_power = max_power_input
        z = magnitudes.clone()
        min_peak_indices = []
        for col in range(z.shape[1]):
            column = z[:, col] 
            peak_indices = []
            for _ in range(self.max_peaks):
                max_value, max_index = torch.max(column[1:], dim=0)
                max_index += 1

                if max_value.item() > self.max_power:
                    peak_indices.append(max_index.item())
                    column[max_index] = -float('inf')
                else:
                    break
            if peak_indices:
                min_peak_indices.append(min(peak_indices))
            else:
                min_peak_indices.append(0)
        return min_peak_indices 

    def harmonic_estimation(self, magnitudes, max_power_input):
        binmask = torch.full_like(magnitudes, 0.5, dtype=torch.float32)
        f0_indices = self.extract_f0(magnitudes, max_power_input)
        for col, f0_index in enumerate(f0_indices):
            if f0_index > 0: 
                pitch_index = f0_index
                for i in range(pitch_index, magnitudes.shape[0] - (self.freq_margin + 1), pitch_index):
                    for k in range(i - self.freq_margin, i + self.freq_margin + 1):
                        if 0 <= k < magnitudes.shape[0]:
                            distance = abs(k - i)
                            binmask[k, col] = max(1 - 0.5 * (distance / self.freq_margin), 0.5)
        return binmask
    
    def forward(self, x):
        batch_size, channels, freq_bins, time_frames = x.shape
        binmasks = torch.zeros(batch_size, 1, freq_bins, time_frames, dtype=torch.float32, device=x.device)
        for b in range(batch_size):
            for t in range(time_frames):
                binmasks[b, 0, :, t] = self.harmonic_estimation(x[b, 0, :, t])
        binmasks = torch.where(binmasks == 0, torch.tensor(0.5, device=x.device, dtype=torch.float32), binmasks)
        return binmasks
