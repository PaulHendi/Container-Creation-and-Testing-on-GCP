# Use an NVIDIA base image with CUDA support
FROM nvidia/cuda:12.1.0-base-ubuntu22.04


# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip


# Set the working directory in the container
WORKDIR /app


# Copy only the requirements file first (for caching optimization)
COPY requirements.txt /app

# Set the environment variable for Hugging Face cache
# Note : the dir is set so that weights are saved to a volume (check launch_training.sh)
# This is to ensure that the weights are not lost when the container is stopped
ENV HF_HOME=/volume/.cache/huggingface


# Install Hugging Face libraries for training
RUN pip3 install --no-cache-dir -r requirements.txt


# (Optional) Copy project files
COPY . /app

