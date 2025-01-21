# LAU-Net
This repository contains the implementation of LAU-Net, a novel approach for speech enhancement in extremely noisy environments. This study was submitted to Interspeech 2025.

# Requirements
The specific versions of the required modules are listed in the **requirements.txt** file.

    ```
    pip install -r requirements.txt
    ```

# LAU-Net Overview
<p align="center" >
	<img src="https://github.com/yonghunsong/LAU-Net/blob/main/figure.PNG" width="700">
</p>

# Project Strcuture
```
.
├── dataset
│   ├── Training
│   │   ├── p00
│   │   │   ├── u00
│   │   │   ├── u01
│   │   │   ├── ...
│   │   ├── p01
│   ├── Test
│   │   ├── ...
├── checkpoint
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
```
# Dataset
### Throat and acoustic microphone paired dataset
* <a href="https://hina3271.github.io/taps-dataset/"> 60 people, 12.7 hours of training data with 5,000 paired utterances and 2.6 hours of evaluation data </a>

# Quick Start

* Download the TAPS dataset and place it in the **dataset folder**.
    
* Run Training or Testing Scripts
    ```
    python train.py
    python test.py
    ```

# Citation
TBD.
