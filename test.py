import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from preprocessing import data_preprocessing_test
from sklearn.model_selection import train_test_split
import logging
import time
from torchinfo import summary
import librosa
import soundfile as sf
from compute_metric import *
import pandas as pd
from pystoi import stoi 
from pesq import pesq
import scipy.signal as signal
import xlsxwriter

from model import *
from dataset import *
from util import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("test.log"),
    logging.StreamHandler()
])


parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=1, type=int, dest="num_epoch")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
parser.add_argument("--nch", default=2, type=int, dest="nch") 
parser.add_argument("--nker", default=4, type=int, dest="nker")
parser.add_argument("--network", default="LAUNet", choices=["LAUNet"], type=str, dest="network")
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir
nch = args.nch
nker = args.nker
network = args.network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = './datasets/test'
sound_file_name, acc_seg_data, mic_seg_data, noisy_seg_data, acc_seg_data_phase, mic_seg_data_phase, noisy_seg_data_phase = data_preprocessing_test(dataset_dir=dataset_dir)

acc_seg_data = acc_seg_data.astype(np.float32)
mic_seg_data = mic_seg_data.astype(np.float32)
noisy_seg_data = noisy_seg_data.astype(np.float32)
acc_seg_data_phase = acc_seg_data_phase.astype(np.float32)
mic_seg_data_phase = mic_seg_data_phase.astype(np.float32)
noisy_seg_data_phase = noisy_seg_data_phase.astype(np.float32)

acc_min, acc_max = np.min(acc_seg_data), np.max(acc_seg_data)
mic_min, mic_max = np.min(mic_seg_data), np.max(mic_seg_data)
noisy_min, noisy_max = np.min(noisy_seg_data), np.max(noisy_seg_data)

acc_seg_data = (acc_seg_data - acc_min) / (acc_max - acc_min)
mic_seg_data = (mic_seg_data - mic_min) / (mic_max - mic_min)
noisy_seg_data = (noisy_seg_data - noisy_min) / (noisy_max - noisy_min)


harmonic_block = HarmonicEstimation()
harmonic_masks = []
for i in range(acc_seg_data.shape[0]):
    x = torch.tensor(acc_seg_data[i], dtype=torch.float32)
    harmonic_mask = harmonic_block.harmonic_estimation(x, max_power_input = 0.5)  
    harmonic_masks.append(harmonic_mask.numpy())
harmonic_masks = np.array(harmonic_masks)

acc_test = torch.tensor(np.array(acc_seg_data), dtype=torch.float32)
mic_test = torch.tensor(np.array(mic_seg_data), dtype=torch.float32)
noisy_test = torch.tensor(np.array(noisy_seg_data), dtype=torch.float32)
harmonic_test = torch.tensor(np.array(harmonic_masks), dtype=torch.float32)

test_dataset = SpectrogramDatasetWithHarmonicMask(acc_test, mic_test, noisy_test, harmonic_test)
loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
num_data_test = len(test_dataset)
num_batch_test = np.ceil(num_data_test / batch_size)

net = LAUNet(nch=nch, nker=nker).to(device)

fn_loss = nn.L1Loss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)


st_epoch = 0
sr = 8000  

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
    net.eval()
    loss_log = []

    combined_sound_data = {}
    count = 0

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        label = label.unsqueeze(1)
        input = data['input'].to(device) 
        
        output = net(input)

        harmonic_mask = input[:, -1:, :, :]
        loss_magnitude = fn_loss(output, label)
        loss_harmonic = fn_loss(output * harmonic_mask, label * harmonic_mask)
        loss = 0.5*loss_magnitude + 0.5*loss_harmonic

        loss_log += [loss.item()]
        logging.info("TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_log)))

        label = fn_tonumpy(label)
        input = fn_tonumpy(input)
        output = fn_tonumpy(output)
        
        input_acc = input[..., 0].squeeze()  
        input_noisy = input[..., 1].squeeze()  
        harmonic_mask = harmonic_mask.detach().cpu().squeeze().numpy()
        label = label[0].squeeze()
        output = output[0].squeeze()
        
        acc_segment = acc_seg_data_phase[batch - 1]
        mic_segment = mic_seg_data_phase[batch - 1]
        noisy_segment = noisy_seg_data_phase[batch - 1] 

        input_acc = input_acc * (acc_max - acc_min) + acc_min
        input_noisy = input_noisy * (noisy_max - noisy_min) + noisy_min
        output = np.clip(output, 0, 1)
        output = output * (noisy_max - noisy_min) + noisy_min
        label = label * (mic_max - mic_min) + mic_min
        
        
        current_sound_name = sound_file_name[batch - 1]
        if current_sound_name not in combined_sound_data:
            combined_sound_data[current_sound_name] = {
                'input_acc': [],
                'input_noisy': [],
                'output': [],
                'label': [],
                'acc_phase': [],
                'mic_phase': [],
                'noisy_phase': [],
                'count': []
            }
     
        combined_sound_data[current_sound_name]['input_acc'].append(input_acc) 
        combined_sound_data[current_sound_name]['input_noisy'].append(input_noisy) 
        combined_sound_data[current_sound_name]['output'].append(output)  
        combined_sound_data[current_sound_name]['label'].append(label)  
        combined_sound_data[current_sound_name]['acc_phase'].append(acc_segment)  
        combined_sound_data[current_sound_name]['mic_phase'].append(mic_segment)  
        combined_sound_data[current_sound_name]['noisy_phase'].append(noisy_segment) 
        combined_sound_data[current_sound_name]['count'].append(count)       
       
        count = count + 1
            
            
    results = []
    test_dataset_dir = './datasets/test'
    subject_folders = [folder for folder in os.listdir(test_dataset_dir) if os.path.isdir(os.path.join(test_dataset_dir, folder))]


    for sound_name, sound_data in combined_sound_data.items():
        combined_input_acc = np.concatenate(sound_data['input_acc'], axis=1)
        combined_input_noisy = np.concatenate(sound_data['input_noisy'], axis=1)
        combined_output = np.concatenate(sound_data['output'], axis=1)
        combined_label = np.concatenate(sound_data['label'], axis=1)
        combined_acc_phase = np.concatenate(sound_data['acc_phase'], axis=1)
        combined_mic_phase = np.concatenate(sound_data['mic_phase'], axis=1)
        combined_noisy_phase = np.concatenate(sound_data['noisy_phase'], axis=1)

        combined_input_acc_amplitude = librosa.db_to_amplitude(combined_input_acc, ref=1.0)
        combined_input_noisy_amplitude = librosa.db_to_amplitude(combined_input_noisy, ref=1.0)
        combined_output_amplitude = librosa.db_to_amplitude(combined_output, ref=1.0)
        combined_label_amplitude = librosa.db_to_amplitude(combined_label, ref=1.0)


        combined_input_acc_amplitude = np.vstack([np.zeros((1, combined_input_acc_amplitude.shape[1])), combined_input_acc_amplitude])
        combined_acc_phase = np.vstack([np.zeros((1, combined_acc_phase.shape[1])), combined_acc_phase])
        
        combined_input_noisy_amplitude = np.vstack([np.zeros((1, combined_input_noisy_amplitude.shape[1])), combined_input_noisy_amplitude])
        combined_noisy_phase = np.vstack([np.zeros((1, combined_noisy_phase.shape[1])), combined_noisy_phase])
        
        combined_output_amplitude = np.vstack([np.zeros((1, combined_output_amplitude.shape[1])), combined_output_amplitude])
    
        combined_label_amplitude = np.vstack([np.zeros((1, combined_label_amplitude.shape[1])), combined_label_amplitude])
        combined_mic_phase = np.vstack([np.zeros((1, combined_mic_phase.shape[1])), combined_mic_phase])


        output_noisy_similarity = combined_input_noisy_amplitude - combined_output_amplitude
        output_similarity = combined_output_amplitude
        combined_phase = np.where(output_noisy_similarity < output_similarity, combined_noisy_phase, combined_acc_phase)
        complex_signal = combined_output_amplitude * np.exp(1j * combined_phase) 
        audio_signal_output = librosa.istft(complex_signal, n_fft=512, hop_length=128, win_length=512, window='hann')


        subject = sound_name.split('_')[0]
        base_path = os.path.join(test_dataset_dir, subject)

        acc_path = os.path.join(base_path, f"{sound_name}_acc.wav")
        mic_path = os.path.join(base_path, f"{sound_name}_mic.wav")
        noisy_mic_path = os.path.join(base_path, f"{sound_name}_noisy_mic.wav")

        acc_original, sr_acc_original = librosa.load(acc_path, sr=8000)
        mic_original, sr_mic_original = librosa.load(mic_path, sr=8000)
        noisy_mic_original, sr_noisy_original = librosa.load(noisy_mic_path, sr=8000)

        min_len = min(len(mic_original), len(audio_signal_output))
        mic_original_trimmed = mic_original[:min_len]
        audio_signal_output_trimmed = audio_signal_output[:min_len]

        output_pesq_score = pesq(sr, mic_original_trimmed, audio_signal_output_trimmed, 'nb')
        output_stoi_score = stoi(mic_original_trimmed, audio_signal_output_trimmed, sr, extended=False)
        output_scig_score = compute_csig(mic_original_trimmed, audio_signal_output_trimmed, sr, norm=False)
        output_cbak_score = compute_cbak(mic_original_trimmed, audio_signal_output_trimmed, sr, norm=False)
        output_covl_score = compute_covl(mic_original_trimmed, audio_signal_output_trimmed, sr, norm=False)
        
        results.append({
            'sound_name': sound_name,
            'output_PESQ': output_pesq_score,
            'output_STOI': output_stoi_score,
            'output_CSIG': output_scig_score,
            'output_CBAK': output_cbak_score,
            'output_COVL': output_covl_score
        })

    df = pd.DataFrame(results)
    df.to_excel('test/sound_metrics.xlsx', index=False)