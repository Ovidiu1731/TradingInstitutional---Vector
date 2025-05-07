# Use official Python base image
FROM python:3.10-slim

# Install Tesseract OCR and system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements.txt first for better layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose the port your app runs on
EXPOSE 8080

# Start the FastAPI app with Uvicorn
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"
