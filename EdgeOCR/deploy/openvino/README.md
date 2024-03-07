# Multi-Model ONNX to OpenVINO Conversion

This project provides a Docker-based solution for converting multiple ONNX models to OpenVINO Intermediate Representation (IR) format. It's designed to handle multiple ONNX models, such as a text detector and a text recognizer.

## Prerequisites

- Docker
- Intel CPU (for optimal performance with OpenVINO)

## Project Structure

- `Dockerfile`: Defines the Docker image based on Intel's OpenVINO image.
- `convert_to_openvino.py`: Python script that converts ONNX models to OpenVINO IR format.
- Your ONNX model files (e.g., `text_detector.onnx`, `text_recognizer.onnx`)

## Setup and Usage

1. Clone this repository or copy the `Dockerfile` and `convert_to_openvino.py` to your project directory.

2. Place your ONNX model files (with `.onnx` extension) in the same directory.

3. Build the Docker image:
   ```
   docker build -t onnx_to_openvino .
   ```

4. Run the Docker container to convert your models:
   ```
   docker run -v $(pwd):/workspace onnx_to_openvino
   ```

   This command mounts your current directory to `/workspace` in the container and runs the conversion script for all `.onnx` files in the directory.

5. After the conversion is complete, you should find new directories named `*_openvino` for each of your ONNX models in your directory. These directories contain the OpenVINO IR versions of your models (.xml and .bin files).

## Customization

- The script will automatically convert all `.onnx` files in the directory.
- You can modify the conversion parameters in `convert_to_openvino.py` to adjust for specific needs or target hardware.
- If you need to compile for specific hardware, uncomment and adjust the relevant lines in the Python script.

## Troubleshooting

- Ensure you have Docker installed and configured correctly.
- If you need a specific version of OpenVINO, you can change the tag in the Dockerfile (e.g., `FROM openvino/ubuntu20_dev:2021.4`).
- For OpenVINO-specific issues, refer to the [OpenVINO Documentation](https://docs.openvino.ai/).

## Contributing

Contributions to improve the conversion process or extend functionality are welcome. Please feel free to submit issues or pull requests.

## License

This project is open-source and available under the MIT License.
