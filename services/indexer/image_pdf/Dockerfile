FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 ccache \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN python -m pip install --no-cache-dir --upgrade pip

# PaddlePaddle GPU build (совместим с CUDA 12.x драйверами через cu118 wheel)
RUN python -m pip install --no-cache-dir paddlepaddle-gpu==3.0.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

COPY image_pdf/requirements.txt /app/image_pdf/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/image_pdf/requirements.txt

COPY common/ /app/common/
COPY router.py /app/router.py
COPY image_pdf/ /app/image_pdf/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/image_pdf/build_index.py"]
