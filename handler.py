#!/usr/bin/env python3
"""
Caption Eraser — RunPod AI Worker
LaMa inpainting per frame pe GPU.
"""

import os
import base64
import tempfile
import subprocess
import runpod
import requests
import numpy as np
import cv2
from PIL import Image

print("[INIT] Incarcare LaMa...", flush=True)
from simple_lama_inpainting import SimpleLama
LAMA = SimpleLama()
print("[INIT] LaMa OK", flush=True)


def build_mask(boxes, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    PAD = 4
    for b in boxes:
        x1 = max(0, int(b['x']) - PAD)
        y1 = max(0, int(b['y']) - PAD)
        x2 = min(width,  int(b['x']) + int(b['w']) + PAD)
        y2 = min(height, int(b['y']) + int(b['h']) + PAD)
        mask[y1:y2, x1:x2] = 255
    return mask


def process_video(input_path, boxes, width, height, fps):
    mask_pil = Image.fromarray(build_mask(boxes, width, height))

    cap = cv2.VideoCapture(input_path)
    raw_out = input_path + "_raw.mp4"
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = LAMA(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mask_pil)
        writer.write(cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR))
        n += 1
        if n % 30 == 0:
            print(f"[PROC] {n} frames...", flush=True)

    cap.release()
    writer.release()
    print(f"[PROC] Done: {n} frames", flush=True)

    # Remux cu audio
    final = input_path + "_final.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-nostats', '-loglevel', 'error',
        '-i', raw_out, '-i', input_path,
        '-map', '0:v:0', '-map', '1:a?',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-c:a', 'copy', '-movflags', '+faststart',
        final,
    ], check=True)

    os.remove(raw_out)
    return final


def handler(job):
    inp = job.get('input', {})
    boxes  = inp.get('boxes', [])
    width  = int(inp.get('width', 0))
    height = int(inp.get('height', 0))
    fps    = float(inp.get('fps', 30.0))

    if not boxes or width == 0 or height == 0:
        return {'error': 'Input invalid: boxes/width/height lipsesc'}

    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    input_path = tmp.name
    tmp.close()

    try:
        if 'video_url' in inp:
            print(f"[DL] {inp['video_url']}", flush=True)
            r = requests.get(inp['video_url'], timeout=300, stream=True)
            r.raise_for_status()
            with open(input_path, 'wb') as f:
                for chunk in r.iter_content(8 * 1024 * 1024):
                    f.write(chunk)
        elif 'video_base64' in inp:
            with open(input_path, 'wb') as f:
                f.write(base64.b64decode(inp['video_base64']))
        else:
            return {'error': 'Niciun video (video_url sau video_base64)'}

        output_path = process_video(input_path, boxes, width, height, fps)
        with open(output_path, 'rb') as f:
            return {'video_base64': base64.b64encode(f.read()).decode()}

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return {'error': str(e)}
    finally:
        for p in [input_path, input_path + '_raw.mp4', input_path + '_final.mp4']:
            try: os.remove(p)
            except: pass


runpod.serverless.start({'handler': handler})
