FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# pip needed obviously, egl1, gl1 and gomp1 for open3d
RUN apt update && apt install -y \
    python3-pip \
    python-is-python3 \
    libegl1 \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/**

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0"

# TODO: need to copy
COPY . /radegs
WORKDIR /radegs

RUN pip install -r requirements.txt 
# Needed for headless
RUN pip install opencv-python-headless 

RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn

RUN pip install fire
