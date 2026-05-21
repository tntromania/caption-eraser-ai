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
    actual_w, actual_h = meta['width'], meta['height']
    num, den = map(int, meta['r_frame_rate'].split('/'))
    actual_fps = num / den
    print(f"[PROC] {actual_w}x{actual_h} @ {actual_fps:.4f}fps", flush=True)

    mask_pil = Image.fromarray(build_mask(boxes, actual_w, actual_h))
    frame_bytes = actual_w * actual_h * 3
    final = input_path + "_final.mp4"

    dec = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error',
        '-i',input_path,
        '-f','rawvideo','-pixel_format','bgr24','pipe:1'
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
            rgb_in = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
            result = LAMA(Image.fromarray(rgb_in), mask_pil)
            rgb_out = np.array(result.convert('RGB'))
            if rgb_out.dtype != np.uint8:
                rgb_out = np.clip(rgb_out * 255, 0, 255).astype(np.uint8)
            if n == 0:
                print(f"[DBG] in_max={bgr_in.max()} out_max={rgb_out.max()} dtype={rgb_out.dtype}", flush=True)
            bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
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
    if not boxes or width==0 or height==0:
        return {'error':'Input invalid'}
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