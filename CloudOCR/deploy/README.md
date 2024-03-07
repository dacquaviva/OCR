# Cloud OCR System

## Overview

This repository contains a cloud-based Optical Character Recognition (OCR) system using pre-trained models from PaddlePaddleOCR. The system is deployed using Triton Inference Server and exposed via a Flask API. This setup allows for efficient, scalable OCR processing in the cloud.

## Key Features

1. **Pre-trained Models**: Utilizes state-of-the-art PaddlePaddleOCR models for text detection and recognition.
2. **ONNX Conversion**: Models are converted to ONNX format for improved compatibility and performance.
3. **Triton Inference Server**: Leverages NVIDIA's Triton Inference Server for efficient model serving.
4. **Flask API**: Provides a simple, RESTful API for easy integration with various applications.
5. **Cloud-Ready**: Designed for deployment in cloud environments for scalability.

## System Architecture

The system consists of two main components:

1. **Text Detector**: Identifies areas of text within an image.
2. **Text Recognizer**: Reads and interprets the detected text.

Both models are served using Triton Inference Server, with a Flask application acting as the front-end API.

## Models

### Text Detector

- **Purpose**: Detects text areas in images (text in the wild).
- **Type**: Segmentation model
- **Version**: 1
- **Source**: PaddlePaddleOCR English Detection Model
- **Original URL**: https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar

### Text Recognizer

- **Purpose**: Recognizes text from detected areas.
- **Type**: CNN + LSTM with CTC decoding
- **Version**: 1
- **Source**: PaddlePaddleOCR English Recognition Model
- **Original URL**: https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar

## ONNX Conversion

The PaddlePaddle models have been converted to ONNX format for deployment. For instructions on how to convert PaddlePaddle models to ONNX, please refer to the official PaddleOCR documentation:

[PaddleOCR Paddle2ONNX Conversion Guide](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/paddle2onnx/readme.md)

This guide provides detailed steps for converting both the text detection and text recognition models to ONNX format.

## Further Optimization

For even better performance, especially on NVIDIA GPUs, the ONNX models can be further optimized by converting them to TensorRT format. TensorRT can significantly accelerate inference by using optimized CUDA kernels and mixed precision arithmetic.