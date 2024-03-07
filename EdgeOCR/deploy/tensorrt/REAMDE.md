# ONNX to TensorRT Conversion

This project provides a Docker-based solution for converting ONNX models to TensorRT format. It includes a Dockerfile for setting up the necessary environment and a Python script for performing the conversion.

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (for Docker GPU support)

## Project Structure

- `Dockerfile`: Defines the Docker image with CUDA, TensorRT, and other necessary dependencies.
- `convert_to_trt.py`: Python script that converts an ONNX model to TensorRT format.
- `model.onnx`: Your ONNX model file (you need to provide this).

## Setup and Usage

1. Clone this repository or copy the `Dockerfile` and `convert_to_trt.py` to your project directory.

2. Place your ONNX model file (named `model.onnx`) in the same directory.

3. Build the Docker image:
   ```
   docker build -t onnx_to_trt .
   ```

4. Run the Docker container to convert your model:
   ```
   docker run --gpus all -v $(pwd):/workspace onnx_to_trt
   ```

   This command mounts your current directory to `/workspace` in the container and runs the conversion script.

5. After the conversion is complete, you should find a `model.trt` file in your directory. This is the TensorRT version of your model.

## Customization

- If your ONNX model has a different name, update the `onnx_file_path` in `convert_to_trt.py`.
- You can modify the TensorRT configuration in `convert_to_trt.py` to adjust parameters like workspace size or precision.

## Troubleshooting

- Ensure you have the NVIDIA Container Toolkit installed and configured correctly.
- If you encounter CUDA version incompatibilities, you may need to adjust the base image in the Dockerfile.
- For TensorRT-specific issues, refer to the [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html).
