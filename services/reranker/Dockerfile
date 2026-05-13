FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    "transformers>=4.41,<4.45" \
    "sentence-transformers>=3.0,<3.1"

COPY app.py .

CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8080"]
