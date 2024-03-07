# Edge-Compatible OCR Model

## Overview

This repository contains an Optical Character Recognition (OCR) model designed for deployment on common edge devices. The model utilizes a simple architecture based on convolutional neural networks (CNNs), making it highly compatible with a wide range of edge computing platforms. It employs Connectionist Temporal Classification (CTC) loss for efficient sequence learning.

## Key Features

1. **Edge-Friendly Architecture**: The model is built using only convolutional operations, which are widely supported on edge devices and accelerators.

2. **Lightweight Design**: The architecture is optimized for efficiency without compromising accuracy, making it suitable for resource-constrained environments.

3. **Flexible Input**: The model can handle various input sizes, with a default size of 75x240 pixels.

4. **Custom ResNet-style Blocks**: The architecture incorporates custom residual blocks for improved feature extraction.

5. **Multi-stage Processing**: The model uses a multi-stage approach with skip connections for better information flow.

6. **Configurable Output**: The number of classes can be easily adjusted to accommodate different character sets or languages.

7. **CTC Loss**: The model uses Connectionist Temporal Classification loss, which allows for efficient learning of sequence-to-sequence tasks without the need for explicit alignment between input and output sequences.


The output of the model is designed to be used with CTC loss during training and CTC decoding during inference.

4. Train the model on your dataset.

## Training

The model is trained using CTC loss, which is particularly suited for OCR tasks as it doesn't require pre-segmented training data. 

## Inference

For inference on edge devices, we recommend converting the PyTorch model to ONNX format for wider compatibility. Note that during inference, you'll need to implement CTC decoding to convert the model's output into readable text.