# Base image with CUDA support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04


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

# --- System dependencies ---
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata \
    libnvidia-gl-535 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglm-dev \
    mesa-utils \
    git \
    curl \
    ffmpeg \
    software-properties-common \
    python3.10 \
    python3.10-dev

RUN curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    ln -s $(which python3.10) /usr/bin/python

# --- Create non-root user ---
RUN groupadd -g ${GID} ${GROUP_NAME} && \
    useradd -ms /bin/bash -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/app

# --- Copy only dependency lists first for caching ---
COPY --chown=${USER_NAME}:${GROUP_NAME} requirements.txt ./

# --- Install dependencies early (cached) ---
RUN pip install --upgrade pip setuptools==69.5.1 ninja && \
    pip install -r requirements.txt

# -- ensure splatting is compiled against the right torch version
RUN pip install --no-cache-dir --no-binary :all: git+https://github.com/African-Pelagic/splatting

# --- Clone ZoeDepth before copying the rest (to isolate caching) ---
RUN mkdir -p extern && \
    git clone https://github.com/isl-org/ZoeDepth.git extern/ZoeDepth

# --- Copy project code ---
COPY --chown=${USER_NAME}:${GROUP_NAME} . .

# --- Download GenWarp models ---
RUN mkdir -p ./checkpoints && ./scripts/download_models.sh

# --- Entrypoint ---
ENTRYPOINT ["python", "main.py"]
