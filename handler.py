#!/usr/bin/env python3
import os, sys, subprocess, traceback

print("[INIT] Python start...", flush=True)

# Detectam Blackwell INAINTE de import torch
def _maybe_force_cpu():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            cap = float(r.stdout.strip().split('\n')[0].strip())
            if cap >= 12.0:
                print(f"[INIT] Blackwell sm_{int(cap*10)} → CPU fallback", flush=True)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            else:
                print(f"[INIT] GPU sm_{int(cap*10)} OK", flush=True)
    except Exception as e:
        print(f"[INIT] GPU check skip: {e}", flush=True)

_maybe_force_cpu()

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
    print(f"[FATAL] LaMa failed: {e}", flush=True)
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
    """
    Procesează frame cu frame prin LaMa, trimite direct la FFmpeg via pipe.
    Zero fișiere intermediare corupte — același pattern ca app-ul principal.
    """
    mask_pil = Image.fromarray(build_mask(boxes, width, height))
    cap = cv2.VideoCapture(input_path)

    final = input_path + "_final.mp4"

    # FFmpeg primește frame-uri raw BGR pe stdin, muxează cu audio din original
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pixel_format', 'bgr24',
        '-video_size', f'{width}x{height}',
        '-framerate', str(fps),
        '-i', 'pipe:0',        # frame-uri procesate
        '-i', input_path,      # audio original
        '-map', '0:v:0',
        '-map', '1:a?',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        final,
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = LAMA(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                mask_pil
            )
            encoder.stdin.write(
                cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR).tobytes()
            )
            n += 1
            if n % 30 == 0:
                print(f"[PROC] {n} frames...", flush=True)
    finally:
        cap.release()
        encoder.stdin.close()

    _, stderr_data = encoder.communicate()
    if encoder.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed: {stderr_data.decode()[:500]}")

    print(f"[PROC] Done: {n} frames → {final}", flush=True)
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
            print(f"[DL] {os.path.getsize(input_path)/1024/1024:.1f} MB OK", flush=True)
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
        for p in [input_path, input_path + '_final.mp4']:
            try: os.remove(p)
            except: pass


print("[INIT] Worker pornit.", flush=True)
runpod.serverless.start({'handler': handler})
