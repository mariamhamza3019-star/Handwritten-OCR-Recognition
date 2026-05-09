FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only version first (no CUDA = much smaller download)
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn==0.46.0 \
    python-multipart==0.0.27 \
    Pillow==12.2.0 \
    opencv-python-headless==4.13.0.92 \
    numpy==2.4.4

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]