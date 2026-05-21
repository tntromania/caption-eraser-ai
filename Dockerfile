FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# FFmpeg + OpenCV runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download LaMa weights in image (cold start rapid)
RUN python -c "from simple_lama_inpainting import SimpleLama; SimpleLama(); print('LaMa weights OK')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
