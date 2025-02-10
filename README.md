# LAU-Net
This repository contains the implementation of LAU-Net, a novel approach for speech enhancement in extremely noisy environments. This study was submitted to Interspeech 2025.

# Requirements
The specific versions of the required modules are listed in the **requirements.txt** file.

    
    pip install -r requirements.txt
    

# LAU-Net Overview
<p align="center" >
	<img src="https://github.com/yonghunsong/LAU-Net/blob/main/figure.PNG" width="700">
</p>

# Project Strcuture
```
.
├── dataset/
│   ├── Training/
│   │   ├── p00
│   │   │   ├── p00_u00_acc.wav
│   │   │   ├── p00_u00_mic.wav
│   │   │   ├── p00_u01_acc.wav
│   │   │   ├── p00_u01_mic.wav
│   │   │   ├── p00_u02_acc.wav
│   │   │   ├── ...
│   │   ├── p01
│   ├── Test/
│   │   ├── ...
├── checkpoint/
├── homepage/ # about audio samples and homepage
│
├── preprocessing.py        
├── dataset.py
├── model.py
├── train.py     
├── test.py 
├── quantizationTFLite.py 
├── compute_metric.py 
├── metric_helper.py 
├── util.py
├── index.html # about homepage (URL: "https://lau-net.github.io/LAU-Net/")
```
# Dataset
### Throat and acoustic microphone paired dataset
* <a href="https://huggingface.co/datasets/yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset"> 60 people and 6,000 paired recordings </a>

# Quick Start

* Download the TAPS dataset and place it in the **dataset folder**.
    
* Run Training or Testing Scripts
    ```
    python train.py
    python test.py
    ```

# Citation
TBD.
