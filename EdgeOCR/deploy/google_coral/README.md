# Multi-Model ONNX to Google Coral Conversion

This project provides a Docker-based solution for converting multiple ONNX models to TensorFlow Lite format optimized for Google Coral Edge TPU. It's designed to handle multiple ONNX models, such as a text detector and a text recognizer.

## Prerequisites

- Docker
- Google Coral device (for running the converted models)

## Project Structure

- `Dockerfile`: Defines the Docker image with TensorFlow and Edge TPU compiler.
- `convert_to_coral.py`: Python script that converts ONNX models to TFLite format and compiles them for Edge TPU.
- Your ONNX model files (e.g., `text_detector.onnx`, `text_recognizer.onnx`)

## Setup and Usage

1. Clone this repository or copy the `Dockerfile` and `convert_to_coral.py` to your project directory.

2. Place your ONNX model files (with `.onnx` extension) in the same directory.

3. Build the Docker image:
   ```
   docker build -t onnx_to_coral .
   ```

4. Run the Docker container to convert your models:
   ```
   docker run -v $(pwd):/workspace onnx_to_coral
   ```

   This command mounts your current directory to `/workspace` in the container and runs the conversion script for all `.onnx` files in the directory.

5. After the conversion is complete, you should find new `.tflite` and `_edgetpu.tflite` files for each of your ONNX models in your directory. The `_edgetpu.tflite` files are compiled for use with Google Coral Edge TPU.

## Customization

- The script will automatically convert all `.onnx` files in the directory.
- You may need to modify the `representative_dataset_gen()` function in the Python script to provide representative data for your specific models.
- Adjust the input and output data types in the TFLite converter if your models require different types.

## Troubleshooting

- Ensure you have Docker installed and configured correctly.
- If you encounter issues with the Edge TPU compiler, make sure you're using a compatible version.
- For TensorFlow Lite or Edge TPU-specific issues, refer to the [TensorFlow Lite](https://www.tensorflow.org/lite) and [Google Coral](https://coral.ai/docs/) documentation.

## Contributing

Contributions to improve the conversion process or extend functionality are welcome. Please feel free to submit issues or pull requests.

## License

This project is open-source and available under the MIT License.
