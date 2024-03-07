# Use NVIDIA's TensorRT image
FROM nvcr.io/nvidia/tensorrt:22.08-py3

# Set the working directory
WORKDIR /workspace

# Install any additional Python packages you might need
RUN pip install onnx onnxruntime

# Copy the ONNX models and conversion script
COPY *.onnx /workspace/
COPY onnx-to-trt-script.py /workspace/

# Set the entrypoint
ENTRYPOINT ["python3", "onnx-to-trt-script.py"]