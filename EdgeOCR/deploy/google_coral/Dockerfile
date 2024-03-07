# Use TensorFlow as the base image
FROM tensorflow/tensorflow:2.8.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install onnx onnx-tf

# Install Edge TPU Compiler
RUN wget https://github.com/google-coral/edgetpu/releases/download/release-grouper/edgetpu_compiler_linux_x86_64_15.0.340273113.zip \
    && unzip edgetpu_compiler_linux_x86_64_15.0.340273113.zip \
    && mv edgetpu_compiler /usr/local/bin/edgetpu_compiler \
    && rm edgetpu_compiler_linux_x86_64_15.0.340273113.zip

# Set the working directory
WORKDIR /workspace

# Copy the ONNX models and conversion script
COPY *.onnx /workspace/
COPY onnx-to-coral-script.py /workspace/

# Set the entrypoint
ENTRYPOINT ["python", "onnx-to-coral-script.py"]
