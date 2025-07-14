# Base image with CUDA support
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# Arguments for user permissions
ARG USER_NAME=genwarp
ARG GROUP_NAME=genwarp
ARG UID=1000
ARG GID=1000

# CUDA env
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/${USER_NAME}/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# System dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install -y tzdata libnvidia-gl-535 libgl1-mesa-dev libglib2.0-0 libglm-dev mesa-utils \
    git curl ffmpeg software-properties-common

# Install Python 3.10 and pip
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.10 python3.10-dev && \
    curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    ln -s $(which python3.10) /usr/bin/python

# Create non-root user
RUN groupadd -g ${GID} ${GROUP_NAME} && \
    useradd -ms /bin/bash -u ${UID} -g ${GID} ${USER_NAME}
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/app

# Copy project files
COPY --chown=${USER_NAME}:${GROUP_NAME} . .

# Install Python packages
RUN pip install --upgrade pip setuptools==69.5.1 ninja
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118

RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt

# Clone ZoeDepth if not already in repo
RUN if [ ! -d "extern/ZoeDepth" ]; then \
    git clone https://github.com/isl-org/ZoeDepth.git extern/ZoeDepth; fi

# Download GenWarp models (script must exist)
RUN mkdir -p ./checkpoints && ./scripts/download_models.sh

# Default entrypoint
ENTRYPOINT ["python", "main.py"]
