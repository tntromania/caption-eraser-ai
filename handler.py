#!/usr/bin/env python3
import os, sys, subprocess, traceback

print("[INIT] Python start...", flush=True)

def _maybe_force_cpu():
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            cap = float(r.stdout.strip().split('\n')[0].strip())
            if cap >= 12.0:
                print(f"[INIT] Blackwell → CPU fallback", flush=True)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
    except: pass

_maybe_force_cpu()

try:
    import base64, tempfile, requests
    import numpy as np, cv2, runpod
    from PIL import Image
    print("[INIT] Importuri OK", flush=True)
except Exception as e:
    print(f"[FATAL] {e}", flush=True); traceback.print_exc(); sys.exit(1)

try:
    from simple_lama_inpainting import SimpleLama
    print("[INIT] Incarcare LaMa...", flush=True)
    LAMA = SimpleLama()
    import torch
    print(f"[INIT] LaMa pe {'CUDA' if torch.cuda.is_available() else 'CPU'}!", flush=True)
except Exception as e:
    print(f"[FATAL] {e}", flush=True); traceback.print_exc(); sys.exit(1)

def auto_detect_boxes(video_path, width, height):
    """Detectează text în frame-ul de la 30% din video cu EasyOCR (GPU)."""
    try:
        import easyocr
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(int(total * 0.3), total - 1)))
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            return []
        reader = easyocr.Reader(['en', 'ro'], gpu=True, verbose=False)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = reader.readtext(frame_rgb, detail=1)
        PAD = 8
        boxes = []
        for (bbox, text, conf) in results:
            if conf < 0.2 or not str(text).strip():
                continue
            pts = np.array(bbox, dtype=np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            boxes.append({
                'x': int(max(0, x - PAD)),
                'y': int(max(0, y - PAD)),
                'w': int(min(width - max(0, x - PAD), bw + 2 * PAD)),
                'h': int(min(height - max(0, y - PAD), bh + 2 * PAD)),
            })
        print(f"[AUTO-DETECT] EasyOCR: {len(boxes)} zone → {boxes}", flush=True)
        return boxes
    except Exception as e:
        print(f"[AUTO-DETECT] Err: {e}", flush=True)
        traceback.print_exc()
        return []


def build_mask(boxes, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in boxes:
        x1=max(0,int(b['x'])-4); y1=max(0,int(b['y'])-4)
        x2=min(width,int(b['x'])+int(b['w'])+4); y2=min(height,int(b['y'])+int(b['h'])+4)
        mask[y1:y2, x1:x2] = 255
    return mask

def process_video(input_path, boxes, width, height, fps):
    import json as _json
    probe = subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0',
         '-show_entries','stream=width,height,r_frame_rate','-of','json',input_path],
        capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe: {probe.stderr[:200]}")
    meta = _json.loads(probe.stdout)['streams'][0]
    actual_w, actual_h = int(meta['width']), int(meta['height'])
    num, den = map(int, meta['r_frame_rate'].split('/'))
    actual_fps = num / den
    print(f"[PROC] {actual_w}x{actual_h} @ {actual_fps:.4f}fps", flush=True)

    if not boxes:
        print("[PROC] Boxes goale — auto-detectare text...", flush=True)
        boxes = auto_detect_boxes(input_path, actual_w, actual_h)
        if not boxes:
            raise RuntimeError("Auto-detectare eșuată: niciun text detectat. Furnizează boxes manual.")
        print(f"[PROC] Auto-detect: {len(boxes)} zone găsite", flush=True)

    # Precompute ROI (masked region + context) — process only this area with LAMA
    CONTEXT = 30
    mask_np = build_mask(boxes, actual_w, actual_h)
    ys, xs = np.where(mask_np > 0)
    ry1 = max(0, int(ys.min()) - CONTEXT)
    ry2 = min(actual_h, int(ys.max()) + CONTEXT + 1)
    rx1 = max(0, int(xs.min()) - CONTEXT)
    rx2 = min(actual_w, int(xs.max()) + CONTEXT + 1)
    roi_h, roi_w = ry2 - ry1, rx2 - rx1
    roi_mask_pil = Image.fromarray(mask_np[ry1:ry2, rx1:rx2])
    print(f"[PROC] ROI: {roi_w}x{roi_h} @ ({rx1},{ry1})", flush=True)

    frame_bytes = actual_w * actual_h * 3
    final = input_path + "_final.mp4"

    dec = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error',
        '-i',input_path,
        '-map','0:v:0',
        '-vf',f'scale={actual_w}:{actual_h},format=bgr24',
        '-f','rawvideo','pipe:1'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    enc = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error',
        '-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{actual_w}x{actual_h}','-framerate',str(actual_fps),
        '-i','pipe:0','-i',input_path,
        '-map','0:v:0','-map','1:a?',
        '-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
        '-c:a','copy','-movflags','+faststart',final,
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    n = 0
    write_err = None
    try:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes: break
            bgr_in = np.frombuffer(raw, dtype=np.uint8).reshape((actual_h, actual_w, 3)).copy()

            # Extract ROI, run LAMA only on caption region
            roi_bgr = bgr_in[ry1:ry2, rx1:rx2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            result_roi = LAMA(Image.fromarray(roi_rgb), roi_mask_pil)
            roi_out = np.array(result_roi.convert('RGB'))
            if roi_out.dtype != np.uint8:
                roi_out = np.clip(roi_out * 255, 0, 255).astype(np.uint8)
            # Ensure correct size before pasting back
            if roi_out.shape[:2] != (roi_h, roi_w):
                roi_out = cv2.resize(roi_out, (roi_w, roi_h))

            if n == 0:
                print(f"[DBG] roi_in_max={roi_bgr.max()} roi_out_max={roi_out.max()} roi_out_shape={roi_out.shape}", flush=True)

            # Paste inpainted ROI back into original frame
            bgr_out = bgr_in.copy()
            bgr_out[ry1:ry2, rx1:rx2] = cv2.cvtColor(roi_out, cv2.COLOR_RGB2BGR)

            try:
                enc.stdin.write(bgr_out.tobytes())
            except (BrokenPipeError, ValueError, OSError) as e:
                write_err = e; break
            n += 1
            if n % 30 == 0: print(f"[PROC] {n} frames...", flush=True)
    finally:
        try: dec.stdout.close()
        except: pass
        try: dec.kill()
        except: pass
        try: enc.stdin.close()
        except: pass

    enc_stderr = enc.stderr.read()
    enc.wait()
    if enc.returncode != 0 or write_err:
        msg = enc_stderr.decode()[:500] if enc_stderr else str(write_err)
        raise RuntimeError(f"FFmpeg enc: {msg}")
    print(f"[PROC] Done: {n} frames", flush=True)
    return final

def handler(job):
    inp = job.get('input', {})
    boxes=inp.get('boxes',[]); width=int(inp.get('width',0))
    height=int(inp.get('height',0)); fps=float(inp.get('fps',30.0))
    tmp=tempfile.NamedTemporaryFile(suffix='.mp4',delete=False); input_path=tmp.name; tmp.close()
    try:
        if 'video_url' in inp:
            print(f"[DL] {inp['video_url']}", flush=True)
            r=requests.get(inp['video_url'],timeout=300,stream=True); r.raise_for_status()
            with open(input_path,'wb') as f:
                for chunk in r.iter_content(8*1024*1024): f.write(chunk)
            print(f"[DL] {os.path.getsize(input_path)/1024/1024:.1f} MB", flush=True)
        elif 'video_base64' in inp:
            with open(input_path,'wb') as f: f.write(base64.b64decode(inp['video_base64']))
        else: return {'error':'Niciun video'}
        out = process_video(input_path, boxes, width, height, fps)
        with open(out,'rb') as f: return {'video_base64': base64.b64encode(f.read()).decode()}
    except Exception as e:
        print(f"[ERROR] {e}", flush=True); traceback.print_exc(); return {'error':str(e)}
    finally:
        for p in [input_path, input_path+'_final.mp4']:
            try: os.remove(p)
            except: pass

print("[INIT] Worker pornit.", flush=True)
runpod.serverless.start({'handler': handler})