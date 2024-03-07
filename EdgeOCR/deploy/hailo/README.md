# Hailo Model Compilation Guide

This guide provides instructions on how to compile ONNX models, including YOLOv8 for text detection, for use with Hailo-8 devices using the Hailo Model Zoo.

## Prerequisites

- A working Hailo Model Zoo environment
- ONNX model file(s) you want to compile
- Calibration images for your model
- Hailo Model Zoo YAML configuration file for your model

## Compilation Process

To generate an HEF (Hailo Executable Format) file for inference on Hailo-8 from your trained ONNX model, follow these steps:

1. Ensure you have a working Hailo Model Zoo environment set up.

2. Locate or create the appropriate YAML configuration file for your model in the Hailo Model Zoo networks configuration directory. For example:
   - For an OCRNET model: `./OCRNET.yaml`
   - For a YOLOv8 model: `hailo_model_zoo/cfg/networks/yolov8s.yaml`

3. Prepare a directory containing calibration images in JPEG or PNG format. These images should be representative of the data your model will process.

4. Run the compilation command using the Hailo Model Zoo CLI:

   For OCR_net or general models:
   ```bash
   hailomz compile --ckpt your_model.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/your_model.yaml --start-node-names name1 name2 --end-node-names name1
   ```

   For YOLOv8 models (e.g., for text detection):
   ```bash
   hailomz compile --ckpt yolov8s.onnx --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolov8s.yaml --start-node-names name1 name2 --end-node-names name1 --classes 80
   ```

   Replace the placeholders with your specific paths and node names:

   - `your_model.onnx` or `yolov8s.onnx`: Path to your ONNX file
   - `/path/to/calibration/imgs/dir/`: Path to the directory containing your calibration images
   - `path/to/your_model.yaml` or `path/to/yolov8s.yaml`: Path to your model's YAML configuration file
   - `name1`, `name2`: Optional start and end node names for customizing parsing behavior

## Important Notes

- The `--ckpt` argument specifies the path to your ONNX file.
- The `--calib-path` argument should point to a directory containing your calibration images in JPEG or PNG format.
- The `--yaml` argument specifies the path to your configuration YAML file.
- The `--start-node-names` and `--end-node-names` arguments are optional and can be used to customize parsing behavior.
- For YOLOv8 models, the `--classes` argument is used to adjust the number of classes in post-processing configuration (optional).
- The Hailo Model Zoo will automatically handle adding input normalization as part of the model.
- For Hailo models, you must manually supply a calibration set.
- For YOLOv8 models, make sure to update the `preprocessing.input_shape` field in the YAML file if it was changed during retraining.

## YAML Configuration

The YAML configuration file is crucial for the compilation process. It defines various aspects of your model and how it should be compiled for Hailo hardware. For more detailed information about YAML files and their structure, please refer to the Hailo Model Zoo documentation.

## Troubleshooting

If you encounter issues during the compilation process:

1. Ensure all paths in your command are correct and the files/directories exist.
2. Check that your ONNX model is compatible with the Hailo compiler.
3. Verify that your calibration images are in the correct format and are representative of your model's input data.
4. Review the YAML configuration file to ensure it's correctly set up for your model.
5. For YOLOv8 models, double-check that the `preprocessing.input_shape` in the YAML file matches your model's input shape.

For further assistance, please consult the Hailo Model Zoo documentation or contact Hailo support.

## Additional Resources

- [Hailo Model Zoo Documentation](https://hailo.ai/developer-zone/documentation/model-zoo/)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)

## License

Please refer to the Hailo Model Zoo license for usage terms and conditions.
