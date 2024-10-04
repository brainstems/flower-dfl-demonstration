# The NVIDIA CUDA base image. CUDA 12.5.0 images fail on Akash.
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    nvidia-cuda-toolkit

# Install Python dependencies
RUN pip install --upgrade pip


# Install additional Python dependencies
COPY requirements.txt /app/
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install boto3 flwr datasets transformers pandas pickle

# Copy the entrypoint script into the container.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy project files (including model operations and other necessary files)
COPY certificates /app/certificates
COPY server.py /app/server.py
COPY util.py /app/util.py
# Expose the WebSocket and HTTP ports
EXPOSE 8000 8080


# Set environment variables
ENV AWS_ACCESS_KEY_ID=''
ENV AWS_SECRET_ACCESS_KEY=''
ENV AWS_DEFAULT_REGION=''


# Set the entry point for the container.
ENTRYPOINT ["/app/entrypoint.sh"]