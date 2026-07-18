FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.deploy.txt .
RUN pip install --no-cache-dir -r requirements.deploy.txt

COPY . .
RUN mkdir -p data/models/perch && \
    curl -L --fail --retry 3 \
      https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx \
      -o data/models/perch/perch_v2.onnx && \
    test "$(wc -c < data/models/perch/perch_v2.onnx | tr -d ' ')" = "409148616"

EXPOSE 8080
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 300 app:app"]
