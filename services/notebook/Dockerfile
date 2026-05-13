# Jupyter Datascience с поддержкой CUDA
# Базовый образ: CUDA runtime + Ubuntu (без полного toolkit, чтобы образ был меньше)
ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python, Jupyter и системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    fonts-dejavu-core \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && pip3 install --no-cache-dir --upgrade pip

# PyTorch (CUDA 12.1) для проверки/работы с GPU из ноутбука
RUN pip3 install --no-cache-dir --retries 10 --timeout 300 \
    "torch>=2.6" "torchvision>=0.21" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu118

# PaddlePaddle GPU (cu118 wheel совместим с CUDA 12.x драйверами)
RUN pip3 install --no-cache-dir paddlepaddle-gpu==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Jupyter и datascience-стек + NLP/Embeddings
COPY services/notebook/requirements.txt /tmp/requirements.notebook.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.notebook.txt

WORKDIR /workspace

EXPOSE 8888

# Для доступа к GPU в контейнере нужен nvidia-container-toolkit на хосте и deploy.resources в compose
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
