import torch
import torch.onnx
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from model import *
from dataset import *
from util import *
from preprocessing import *
import tensorflow as tf
import onnx
import random
import numpy as np

class QunatizationTFLite():
    def __init__(self, model_path, data_dir, onnx_path, quantized_path):
        super().__init__()
        self.model_path = model_path
        self.cali_input_path = data_dir
        self.onnx_path = onnx_path
        self.quantized_path = quantized_path

        self.model = LAUNet(nch=2, nker=4)  
        self.model.load_state_dict(torch.load(self.model_path)['net'])


    def ToONNX(self):
        self.model.eval()

        dummy_input =  torch.randn(1,3,256,16)
        torch.onnx.export(
            self.model,
            dummy_input,
            self.onnx_path, 
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
    
    def ToquantizedTFlite(self):
        
        def Calibration(self):
            dataset_dir = './datasets/test'
            _, acc_seg_data, _, noisy_seg_data, _, _, _ = data_preprocessing_test(dataset_dir=dataset_dir)

            harmonic_block = HarmonicEstimation()
            harmonic_masks = []
            for i in range(acc_seg_data.shape[0]):
                x = torch.tensor(acc_seg_data[i], dtype=torch.float32)
                harmonic_mask = harmonic_block.harmonic_estimation(x, max_power_input = 0.5)  
                harmonic_masks.append(harmonic_mask.numpy())
            harmonic_masks = np.array(harmonic_masks)

            acc_min, acc_max = np.min(acc_seg_data), np.max(acc_seg_data)
            noisy_min, noisy_max = np.min(noisy_seg_data), np.max(noisy_seg_data)

            acc_seg_data = (acc_seg_data - acc_min) / (acc_max - acc_min)
            noisy_seg_data = (noisy_seg_data - noisy_min) / (noisy_max - noisy_min)

            acc_tensor= torch.tensor(np.array(acc_seg_data), dtype=torch.float32)
            noisy_tensor = torch.tensor(np.array(noisy_seg_data), dtype=torch.float32)
            harmonic_tensor = torch.tensor(harmonic_masks, dtype=torch.float32)
            random_integers = [random.randint(0, len(acc_tensor) + 1) for _ in range(1000)]

            for i in random_integers:
                acc = acc_tensor[i].numpy()
                noisy = noisy_tensor[i].numpy()
                harmonic = harmonic_tensor[i].numpy()
                input_data = np.stack((acc, noisy, harmonic), axis=0).astype(np.float32) 
                input_data = np.expand_dims(input_data, axis=0)
                yield [input_data]

        onnx_model = onnx.load(self.onnx_path)
        tf_rep = prepare(onnx_model)
        tf_model_dir = "./tf_model_temp"
        tf_rep.export_graph(tf_model_dir)

        quant_converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
        quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_converter.representative_dataset = Calibration

        quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        quant_converter.inference_input_type = tf.uint8
        quant_converter.inference_output_type = tf.uint8
        tflite_model_quant = quant_converter.convert()

        with open(self.quantized_path, "wb") as f:
            f.write(tflite_model_quant)
            
model_path = './checkpoing'
data_dir = './datasets/test'
onnx_path = './user_defined'
quantized_path = './user_defined'

stm_File = QunatizationTFLite(model_path, data_dir, onnx_path, quantized_path)
stm_File.ToONNX()
stm_File.ToquantizedTFlite()