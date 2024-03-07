# Use Intel's OpenVINO image as the base
FROM openvino/ubuntu20_dev:latest

# Set the working directory
WORKDIR /workspace

# Install additional Python packages
RUN pip install onnx

# Copy the ONNX models and conversion script
COPY *.onnx /workspace/
COPY onnx-to-openvino-script.py /workspace/

# Set the entrypoint
ENTRYPOINT ["python3", "onnx-to-openvino-script.py"]
