#!/usr/bin/env python3
import os, sys, subprocess, traceback

print("[INIT] Python start...", flush=True)

# ── Detectam Blackwell INAINTE de import torch ────────────────
# Daca GPU e sm_120+ (Blackwell) si PyTorch e vechi, fortam CPU.
def _maybe_force_cpu():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return
        cap = float(r.stdout.strip().split('\n')[0].strip())  # ex: "12.0"
        if cap >= 12.0:
            print(f"[INIT] Blackwell GPU (sm_{int(cap*10)}) detectat → CPU fallback activat", flush=True)
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            print(f"[INIT] GPU sm_{int(cap*10)} OK pentru CUDA", flush=True)
    except Exception as e:
        print(f"[INIT] nvidia-smi check skip: {e}", flush=True)

_maybe_force_cpu()

# ── Importuri (dupa ce am setat CUDA_VISIBLE_DEVICES) ─────────
try:
    import base64, tempfile
    import requests
    import numpy as np
    import cv2
    from PIL import Image
    import runpod
    print("[INIT] Importuri OK", flush=True)
except Exception as e:
    print(f"[FATAL] Import failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    from simple_lama_inpainting import SimpleLama
    print("[INIT] Incarcare LaMa...", flush=True)
    LAMA = SimpleLama()
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INIT] LaMa gata pe {device.upper()}!", flush=True)
except Exception as e:
    print(f"[FATAL] LaMa load failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


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
    print(f"[JOB] {job.get('id', '?')}", flush=True)
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
            print(f"[DL] {os.path.getsize(input_path)/1024/1024:.1f} MB", flush=True)
        elif 'video_base64' in inp:
            with open(input_path, 'wb') as f:
                f.write(base64.b64decode(inp['video_base64']))
        else:
            return {'error': 'Niciun video furnizat'}

        output_path = process_video(input_path, boxes, width, height, fps)
        with open(output_path, 'rb') as f:
            return {'video_base64': base64.b64encode(f.read()).decode()}

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        return {'error': str(e)}
    finally:
        for p in [input_path, input_path + '_raw.mp4', input_path + '_final.mp4']:
            try: os.remove(p)
            except: pass


print("[INIT] Pornire worker...", flush=True)
runpod.serverless.start({'handler': handler})
