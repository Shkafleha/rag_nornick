FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir --timeout=300 -r requirements.txt
COPY server.py .
COPY lab_dict_search_clean.json .
CMD ["python", "server.py"]
