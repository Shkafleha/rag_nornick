FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY text_pdf/requirements.txt /app/text_pdf/requirements.txt
RUN pip install --no-cache-dir -r /app/text_pdf/requirements.txt

# common/ и router.py подтягиваются через volume в compose,
# либо копируются для self-contained сборки:
COPY common/ /app/common/
COPY router.py /app/router.py
COPY text_pdf/ /app/text_pdf/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/text_pdf/build_index.py"]
