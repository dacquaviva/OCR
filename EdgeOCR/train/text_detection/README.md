# Edge OCR: Text Detection

## Overview

This section demonstrates how to implement efficient text detection for edge devices. We use a YOLO (You Only Look Once) model from the Ultralytics family, specifically YOLOv8, for fast and accurate text detection.

## Text Detection with YOLOv8

For the text detection stage of our Edge OCR pipeline, we utilize YOLOv8, a state-of-the-art object detection model that offers an excellent balance between speed and accuracy, making it suitable for edge devices.

### Why YOLOv8?

1. **Efficiency**: YOLOv8 is designed to be fast and lightweight, making it ideal for edge computing scenarios.
2. **Accuracy**: Despite its efficiency, YOLOv8 maintains high detection accuracy.
3. **Flexibility**: It can be easily fine-tuned for specific tasks like text detection.
4. **Community Support**: Being part of the Ultralytics family, YOLOv8 has strong community support and regular updates.

## Training Data Preparation

To train an effective YOLOv8 model for text detection, we need a large dataset of annotated images. Here's how you can create such a dataset:

1. **Collect Raw Images**: Gather a diverse set of images containing text in various scenarios.

2. **Use CRAFT for Initial Annotation**: We recommend using the CRAFT (Character Region Awareness for Text Detection) model to generate initial annotations. CRAFT is a more complex and accurate text detection model, making it suitable for creating high-quality annotations.

   - CRAFT GitHub Repository: [https://github.com/clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch)

3. **Generate YOLO Format Annotations**: Convert the CRAFT output to YOLO format annotations.

4. **Manual Verification**: Review and adjust the annotations as necessary to ensure quality.

## Training and Exporting to ONNX

### 1. Training Your Model

Once your Docker environment is set up, you can start training your model:

a. Prepare your custom dataset:
   - Create a `dataset.yaml` configuration file
   - Prepare labels in YOLO format (a corresponding .txt file for each image)
   - Include the number of classes in the yaml file, e.g., `nc: 80`

b. Start training with the following command (example for training a YOLOv8s model):

```bash
yolo detect train data=coco128.yaml model=yolov8s.pt name=retrain_yolov8s epochs=100 batch=16
```

- `yolov8s.pt`: Pretrained weights. Weights for YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x will be automatically downloaded.
- `coco128.yaml`: Example data.yaml file (found in ultralytics/ultralytics/datasets)
- `retrain_yolov8s`: Name of the output directory (weights will be saved in ultralytics/runs/detect/retrain_yolov8s)
- `epochs`: Number of training epochs (default: 100)
- `batch`: Batch size (default: 16)

**Note:** For more configurable parameters, visit [Ultralytics Documentation](https://docs.ultralytics.com/modes/train/)

### 2. Exporting to ONNX

To export your trained YOLOv8 model to ONNX format, use the following command:

```bash
yolo export model=/path/to/trained/best.pt imgsz=640 format=onnx opset=11  # export at 640x640
```

**Note:** For more export options, refer to [Ultralytics Export Documentation](https://docs.ultralytics.com/modes/export/)

## Inference

After exporting to ONNX, you can use your YOLOv8 model for text detection on edge devices. The exact implementation will depend on your specific edge computing platform.


## Next Steps

After detecting text regions with YOLOv8, the next step in the OCR pipeline is text recognition. Refer to our text recognition model for the complete Edge OCR solution.
