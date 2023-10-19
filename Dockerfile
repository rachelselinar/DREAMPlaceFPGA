# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# Set the working directory in the container to /app
# WORKDIR /app

# Add metadata to an image
LABEL maintainer="zx.jiang@utexas.edu"
LABEL version="1.0"
LABEL description="Docker image for dreamplacefpga"

#
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    flex \
    vim \
    libcairo2-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Installs system dependencies from conda.
RUN conda install -y -c conda-forge bison

ADD https://cmake.org/files/v3.21/cmake-3.21.0-linux-x86_64.sh /cmake-3.21.0-linux-x86_64.sh
RUN mkdir /opt/cmake \
        && sh /cmake-3.21.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
        && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
        && cmake --version

COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

