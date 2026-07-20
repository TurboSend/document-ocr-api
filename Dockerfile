FROM python:3.12-slim

# Install system dependencies:
#   tesseract-ocr   -> pytesseract
#   libgl1 / libglib2.0-0 -> opencv-python-headless
#   libgomp1        -> onnxruntime (used by rapidocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5005

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5005", "--timeout", "120", "app:app"]
