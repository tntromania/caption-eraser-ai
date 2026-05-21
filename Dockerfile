FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --no-deps = nu reinstala torch/torchvision (deja in imaginea de baza)
RUN pip install --no-cache-dir --no-deps simple-lama-inpainting

# Restul dependintelor simple-lama-inpainting (fara torch)
RUN pip install --no-cache-dir \
    runpod \
    requests \
    opencv-python-headless \
    Pillow \
    huggingface_hub \
    omegaconf \
    kornia

# Pre-download weights LaMa in imagine
RUN python -c "from simple_lama_inpainting import SimpleLama; SimpleLama(); print('LaMa weights OK')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
