
## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import random
from scipy.fft import fft, fftfreq
from scipy.signal import stft
import pysptk
import soundfile as sf
import time
import torch.nn.functional as F
import torch
import pandas as pd


def create_spectrogram(waveform):
    D = librosa.stft(waveform, n_fft=512, win_length=512, hop_length=128, window='hann')
    D_amplitude = np.abs(D)
    D_phase = np.angle(D)
    return D_amplitude, D_phase

def split_spectrogram_fixed_size(spectrogram, segment_size=64):
    num_segments = spectrogram.shape[1] // segment_size 
    segments = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment = spectrogram[:, start_idx:end_idx]
        segments.append(segment)
    return segments

def pad_to_multiple_of_x(spectrogram, segment_size=64):
    time_length = spectrogram.shape[1]
    remainder = time_length % segment_size
    if remainder != 0:
        padding_amount = segment_size - remainder
        padding = np.zeros((spectrogram.shape[0], padding_amount))
        spectrogram = np.hstack((spectrogram, padding))
    return spectrogram


def data_preprocessing_train(dataset_dir=None):
    dataset_dir = dataset_dir
    subfolders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]

    acc_seg_data = []
    mic_seg_data = []
    noisy_seg_data = []

    for subfolder in subfolders:
        print(f"Exploring folder: {subfolder}")
        acc_files = []
        mic_files = []
        noisy_files = []

        for file in os.listdir(subfolder):
            if file.endswith(".wav"):
                if "acc" in file:
                    acc_files.append(file)
                elif "noisy_mic" in file:
                    noisy_files.append(file)
                elif "mic" in file:
                    mic_files.append(file)
        
        mic_number = 0

        for acc_file in acc_files:
            acc_number = acc_file.split('p')[-1].split('u')[0]
            matching_mic_file = [mic_file for mic_file in mic_files if
                                 mic_file.split('p')[-1].split('u')[0] == acc_number]
            matching_noisy_file = [noisy_file for noisy_file in noisy_files if
                                 noisy_file.split('p')[-1].split('u')[0] == acc_number]

            if matching_mic_file:
                acc_path = os.path.join(subfolder, acc_file)
                mic_path = os.path.join(subfolder, matching_mic_file[mic_number])
                noisy_path = os.path.join(subfolder, matching_noisy_file[mic_number])
                mic_number = mic_number + 1
                
                acc_data, acc_sr = librosa.load(acc_path, sr=8000)
                mic_data, mic_sr = librosa.load(mic_path, sr=8000)
                noisy_data, noisy_sr = librosa.load(noisy_path, sr=8000)
                
                if len(acc_data) > 511:
                    acc_spectrogram, acc_phase = create_spectrogram(acc_data)
                    mic_spectrogram, mic_phase = create_spectrogram(mic_data)
                    noisy_spectrogram, noisy_phase = create_spectrogram(noisy_data)

                    acc_spectrogram = acc_spectrogram[1:, :]
                    mic_spectrogram = mic_spectrogram[1:, :]
                    noisy_spectrogram = noisy_spectrogram[1:, :]

                    acc_spectrogram = librosa.amplitude_to_db(acc_spectrogram, ref=np.max)
                    mic_spectrogram = librosa.amplitude_to_db(mic_spectrogram, ref=np.max)
                    noisy_spectrogram = librosa.amplitude_to_db(noisy_spectrogram, ref=np.max)

                    acc_segments = split_spectrogram_fixed_size(acc_spectrogram, segment_size=16)
                    mic_segments = split_spectrogram_fixed_size(mic_spectrogram, segment_size=16)
                    noisy_segments = split_spectrogram_fixed_size(noisy_spectrogram, segment_size=16)

                    acc_seg_data.extend(acc_segments)
                    mic_seg_data.extend(mic_segments)
                    noisy_seg_data.extend(noisy_segments)

    return acc_seg_data, mic_seg_data, noisy_seg_data


def data_preprocessing_test(dataset_dir=None):
    dataset_dir = dataset_dir
    subfolders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]

    sound_file_name = []
    acc_seg_data = []
    mic_seg_data = []
    noisy_seg_data = []
    acc_seg_data_phase = []
    mic_seg_data_phase = []
    noisy_seg_data_phase = []

    for idx, subfolder in enumerate(subfolders):
        print(f"Exploring folder: {subfolder}")
        acc_files = []
        mic_files = []
        noisy_files = []

        for file in os.listdir(subfolder):
            if file.endswith(".wav"):
                if "acc" in file:
                    acc_files.append(file)
                elif "noisy_mic" in file:
                    noisy_files.append(file)
                elif "mic" in file:
                    mic_files.append(file)

        mic_number = 0
        for acc_file in acc_files:
            acc_number = acc_file.split('p')[-1].split('u')[0]
            subject_sentences = acc_file.split('_acc')[0]
            
            matching_mic_file = [mic_file for mic_file in mic_files if
                                 mic_file.split('p')[-1].split('u')[0] == acc_number]
            matching_noisy_file = [noisy_file for noisy_file in noisy_files if
                                 noisy_file.split('p')[-1].split('u')[0] == acc_number]

            if matching_mic_file:
                acc_path = os.path.join(subfolder, acc_file)
                mic_path = os.path.join(subfolder, matching_mic_file[mic_number])
                noisy_path = os.path.join(subfolder, matching_noisy_file[mic_number])
                mic_number = mic_number + 1

                acc_data, acc_sr = librosa.load(acc_path, sr=8000)
                mic_data, mic_sr = librosa.load(mic_path, sr=8000)
                noisy_data, noisy_sr = librosa.load(noisy_path, sr=8000)

                if len(acc_data) > 511:
                    acc_spectrogram, acc_phase = create_spectrogram(acc_data)
                    mic_spectrogram, mic_phase = create_spectrogram(mic_data)
                    noisy_spectrogram, noisy_phase = create_spectrogram(noisy_data)

                    acc_spectrogram = pad_to_multiple_of_x(acc_spectrogram[1:, :], segment_size=16)
                    mic_spectrogram = pad_to_multiple_of_x(mic_spectrogram[1:, :], segment_size=16)
                    noisy_spectrogram = pad_to_multiple_of_x(noisy_spectrogram[1:, :], segment_size=16)
                    acc_phase = pad_to_multiple_of_x(acc_phase[1:, :], segment_size=16)
                    mic_phase = pad_to_multiple_of_x(mic_phase[1:, :], segment_size=16)
                    noisy_phase = pad_to_multiple_of_x(noisy_phase[1:, :], segment_size=16)

                    acc_spectrogram = librosa.amplitude_to_db(acc_spectrogram, ref=np.max)
                    mic_spectrogram = librosa.amplitude_to_db(mic_spectrogram, ref=np.max)
                    noisy_spectrogram = librosa.amplitude_to_db(noisy_spectrogram, ref=np.max)
        
                    acc_segments = split_spectrogram_fixed_size(acc_spectrogram, segment_size=16)
                    mic_segments = split_spectrogram_fixed_size(mic_spectrogram, segment_size=16)
                    noisy_segments = split_spectrogram_fixed_size(noisy_spectrogram, segment_size=16)
                    acc_phase = split_spectrogram_fixed_size(acc_phase, segment_size=16)
                    mic_phase = split_spectrogram_fixed_size(mic_phase, segment_size=16)
                    noisy_phase = split_spectrogram_fixed_size(noisy_phase, segment_size=16)

                    acc_segments_array = np.array(acc_segments)
                    segment_counts = acc_segments_array.shape[0] 

                    for _ in range(segment_counts):
                        sound_file_name.append(subject_sentences)

                    acc_seg_data.extend(acc_segments)
                    mic_seg_data.extend(mic_segments)
                    noisy_seg_data.extend(noisy_segments)
                    acc_seg_data_phase.extend(acc_phase)
                    mic_seg_data_phase.extend(mic_phase)
                    noisy_seg_data_phase.extend(noisy_phase)
    
    return sound_file_name, acc_seg_data, mic_seg_data, noisy_seg_data, acc_seg_data_phase, mic_seg_data_phase, noisy_seg_data_phase
