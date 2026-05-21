require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { exec, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const { authenticate, hubAPI } = require('./hub-auth');

const app = express();
const PORT = process.env.PORT || 3000;
const DOWNLOAD_DIR = path.join(__dirname, 'downloads');

if (!fs.existsSync(DOWNLOAD_DIR)) fs.mkdirSync(DOWNLOAD_DIR);

// ── Credit costs per tier ─────────────────────────────────────
const CREDIT_COSTS = { fast: 0.5, standard: 0.75, ai: 1.5 };

// ── File cleanup ──────────────────────────────────────────────
const FILE_TTL_MS = 30 * 60 * 1000;

function cleanDownloads() {
    try {
        const now = Date.now();
        let deleted = 0;
        fs.readdirSync(DOWNLOAD_DIR).forEach(file => {
            const fp = path.join(DOWNLOAD_DIR, file);
            try { if (now - fs.statSync(fp).mtimeMs > FILE_TTL_MS) { fs.unlinkSync(fp); deleted++; } } catch (_) {}
        });
        if (deleted > 0) console.log(`🧹 Curățare: ${deleted} fișiere șterse din downloads/`);
    } catch (e) { console.error('Eroare curățare:', e.message); }
}

try {
    const files = fs.readdirSync(DOWNLOAD_DIR);
    files.forEach(f => { try { fs.unlinkSync(path.join(DOWNLOAD_DIR, f)); } catch (_) {} });
    console.log(`🧹 Startup: ${files.length} fișiere vechi șterse`);
} catch (_) {}

setInterval(cleanDownloads, 10 * 60 * 1000);

const upload = multer({ dest: DOWNLOAD_DIR, limits: { fileSize: 200 * 1024 * 1024 } });
app.use(cors({ origin: '*' }));
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Auth routes ───────────────────────────────────────────────
app.post('/api/auth/google', async (req, res) => {
    try {
        const response = await fetch(`${process.env.HUB_URL}/api/auth/google`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body),
        });
        const data = await response.json();
        res.status(response.status).json(data);
    } catch (e) {
        res.status(500).json({ error: 'Nu pot comunica cu serverul principal.' });
    }
});

app.get('/api/auth/me', authenticate, async (req, res) => {
    res.json({ user: req.user });
});

// ── Helper: ffprobe metadata ──────────────────────────────────
function getVideoMeta(inputPath) {
    return new Promise((resolve, reject) => {
        exec(
            `ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of json "${inputPath}"`,
            (err, out) => {
                if (err) return reject(err);
                try {
                    const meta = JSON.parse(out).streams[0];
                    const [num, den] = meta.r_frame_rate.split('/').map(Number);
                    resolve({ width: meta.width, height: meta.height, fps: (num / den).toFixed(4) });
                } catch (e) { reject(e); }
            }
        );
    });
}

// ── Helper: parse + clamp boxes ──────────────────────────────
function parseAndClampBoxes(body, width, height) {
    const MARGIN = 2, MIN_SIZE = 4;
    let raw = [];

    if (body.boxes) {
        try {
            const parsed = JSON.parse(body.boxes);
            if (Array.isArray(parsed) && parsed.length > 0) {
                raw = parsed.slice(0, 5).map(b => ({
                    x: parseInt(b.x), y: parseInt(b.y),
                    w: parseInt(b.w), h: parseInt(b.h),
                }));
            }
        } catch (_) {}
    }

    if (raw.length === 0) {
        raw = [{
            x: body.boxX !== undefined ? parseInt(body.boxX) : 10,
            y: body.boxY !== undefined ? parseInt(body.boxY) : 70,
            w: body.boxW !== undefined ? parseInt(body.boxW) : 80,
            h: body.boxH !== undefined ? parseInt(body.boxH) : 20,
        }];
    }

    raw = raw.filter(b =>
        Number.isFinite(b.x) && Number.isFinite(b.y) &&
        Number.isFinite(b.w) && Number.isFinite(b.h) &&
        b.w > 0 && b.h > 0 && b.x >= 0 && b.y >= 0 && b.x < 100 && b.y < 100
    );

    const clamped = [];
    for (const b of raw) {
        let px = Math.floor((b.x / 100) * width);
        let py = Math.floor((b.y / 100) * height);
        let pw = Math.floor((b.w / 100) * width);
        let ph = Math.floor((b.h / 100) * height);

        if (px < MARGIN) px = MARGIN;
        if (py < MARGIN) py = MARGIN;
        if (px + pw > width  - MARGIN) pw = width  - MARGIN - px;
        if (py + ph > height - MARGIN) ph = height - MARGIN - py;
        if (pw < MIN_SIZE) { pw = MIN_SIZE; if (px + pw > width  - MARGIN) px = width  - MARGIN - pw; }
        if (ph < MIN_SIZE) { ph = MIN_SIZE; if (py + ph > height - MARGIN) py = height - MARGIN - ph; }
        if (px < MARGIN) px = MARGIN;
        if (py < MARGIN) py = MARGIN;

        if (px + pw >= width || py + ph >= height || pw < MIN_SIZE || ph < MIN_SIZE) continue;
        clamped.push({ x: px, y: py, w: pw, h: ph });
    }
    return clamped;
}

// ══════════════════════════════════════════════════════════════
// ██ TIER 1: FAST — FFmpeg delogo (1-3 secunde, 0.5 credite)
// ══════════════════════════════════════════════════════════════
function processFast(inputPath, outputPath, boxes) {
    return new Promise((resolve, reject) => {
        // delogo: extrapolare rapidă a pixelilor din jurul zonei
        const delogoFilter = boxes.map(b =>
            `delogo=x=${b.x}:y=${b.y}:w=${b.w}:h=${b.h}`
        ).join(',');

        let errBuf = '';
        const enc = spawn('ffmpeg', [
            '-y', '-nostats', '-loglevel', 'error',
            '-i', inputPath,
            '-vf', delogoFilter,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            outputPath,
        ], { stdio: ['ignore', 'ignore', 'pipe'] });

        enc.stderr.on('data', d => { errBuf += d; if (errBuf.length > 2048) errBuf = errBuf.slice(-2048); });
        enc.on('close', code => code === 0 ? resolve() : reject(new Error(errBuf.slice(-400) || `FFmpeg exit ${code}`)));
        enc.on('error', reject);
    });
}

// ══════════════════════════════════════════════════════════════
// ██ TIER 2: STANDARD — TELEA Inpainting Python (0.75 credite)
// ══════════════════════════════════════════════════════════════
function processStandard(inputPath, outputPath, boxes, width, height, fps, videoId) {
    return new Promise((resolve, reject) => {
        const configJson = JSON.stringify({ width, height, boxes });
        const PYTHON = process.env.PYTHON_BIN || 'python3';
        const WORKER = path.join(__dirname, 'inpaint_worker.py');

        const decoder = spawn('ffmpeg', [
            '-y', '-nostats', '-loglevel', 'error',
            '-i', inputPath,
            '-map', '0:v:0',
            '-vf', 'format=bgr24',
            '-f', 'rawvideo',
            'pipe:1',
        ], { stdio: ['ignore', 'pipe', 'pipe'] });

        const worker = spawn(PYTHON, [WORKER], { stdio: ['pipe', 'pipe', 'pipe'] });
        worker.stdin.write(configJson + '\n');
        decoder.stdout.pipe(worker.stdin);

        const encoder = spawn('ffmpeg', [
            '-y', '-nostats', '-loglevel', 'error',
            '-f', 'rawvideo',
            '-pixel_format', 'bgr24',
            '-video_size', `${width}x${height}`,
            '-framerate', fps,
            '-i', 'pipe:0',
            '-i', inputPath,
            '-map', '0:v:0',
            '-map', '1:a?',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            outputPath,
        ], { stdio: ['pipe', 'pipe', 'pipe'] });

        worker.stdout.pipe(encoder.stdin);

        let decErr = '', encErr = '';
        decoder.stderr.on('data', d => { decErr += d; if (decErr.length > 2048) decErr = decErr.slice(-2048); });
        worker.stderr.on('data', d => process.stdout.write(`[PY  ${videoId}] ${d}`));
        encoder.stderr.on('data', d => { encErr += d; if (encErr.length > 2048) encErr = encErr.slice(-2048); });

        let pipelineError = null;
        decoder.on('error', e => { pipelineError = pipelineError || e; });
        worker.on('error',  e => { pipelineError = pipelineError || e; });
        encoder.on('error', e => { pipelineError = pipelineError || e; });

        encoder.on('close', code => {
            if (code === 0 && !pipelineError) {
                resolve();
            } else {
                if (decErr.trim()) console.error(`[DEC ${videoId}] stderr:`, decErr.trim());
                if (encErr.trim()) console.error(`[ENC ${videoId}] stderr:`, encErr.trim());
                reject(pipelineError || new Error(`Pipeline exit ${code}`));
            }
        });
    });
}

// ══════════════════════════════════════════════════════════════
// ██ TIER 3: AI — RunPod LaMa Inpainting (1.5 credite)
// ══════════════════════════════════════════════════════════════
async function processAI(inputPath, outputPath, boxes, width, height, fps, videoId) {
    const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
    const RUNPOD_ENDPOINT = process.env.RUNPOD_ENDPOINT_ID;
    const SERVER_URL = (process.env.SERVER_PUBLIC_URL || '').replace(/\/$/, '');

    if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT) {
        throw new Error('RunPod nu este configurat. Adaugă RUNPOD_API_KEY și RUNPOD_ENDPOINT_ID în .env');
    }

    // Build input payload — prefer URL (no size limit), fallback base64 pentru videouri mici
    let inputPayload;

    if (SERVER_URL) {
        // Remuxăm fișierul multer (fără extensie) într-un MP4 accesibil
        const inputName = `ai_input_${videoId}.mp4`;
        const inputPublicPath = path.join(DOWNLOAD_DIR, inputName);

        await new Promise((resolve, reject) => {
            const cp = spawn('ffmpeg', [
                '-y', '-nostats', '-loglevel', 'error',
                '-i', inputPath, '-c', 'copy', inputPublicPath,
            ], { stdio: ['ignore', 'ignore', 'pipe'] });
            cp.on('close', code => code === 0 ? resolve() : reject(new Error(`Remux exit ${code}`)));
            cp.on('error', reject);
        });

        inputPayload = {
            video_url: `${SERVER_URL}/download/${inputName}`,
            _inputPublicPath: inputPublicPath,
        };
        console.log(`[AI ${videoId}] Video URL: ${inputPayload.video_url}`);
    } else {
        const fileSize = fs.statSync(inputPath).size;
        if (fileSize > 50 * 1024 * 1024) {
            throw new Error('Videoul este prea mare pentru RunPod fără SERVER_PUBLIC_URL. Adaugă SERVER_PUBLIC_URL în .env');
        }
        inputPayload = { video_base64: fs.readFileSync(inputPath).toString('base64') };
        console.log(`[AI ${videoId}] Trimit video ca base64 (${(fileSize / 1024 / 1024).toFixed(1)} MB)`);
    }

    // Start RunPod async job
    const jobRes = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT}/run`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${RUNPOD_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
            input: {
                video_url:    inputPayload.video_url    || undefined,
                video_base64: inputPayload.video_base64 || undefined,
                boxes,
                width,
                height,
                fps: parseFloat(fps),
            }
        }),
    });

    if (!jobRes.ok) {
        const txt = await jobRes.text();
        throw new Error(`RunPod start failed (${jobRes.status}): ${txt}`);
    }

    const { id: jobId } = await jobRes.json();
    console.log(`[AI ${videoId}] Job ID: ${jobId}`);

    // Poll until COMPLETED / FAILED (max 30 minute)
    const AI_TIMEOUT_MS = 30 * 60 * 1000;
    const pollStart = Date.now();

    try {
        while (true) {
            if (Date.now() - pollStart > AI_TIMEOUT_MS) {
                throw new Error('RunPod job timeout (30 minute depășite)');
            }

            await new Promise(r => setTimeout(r, 3000));

            const statusRes = await fetch(
                `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT}/status/${jobId}`,
                { headers: { 'Authorization': `Bearer ${RUNPOD_API_KEY}` } }
            );
            const status = await statusRes.json();
            console.log(`[AI ${videoId}] Status: ${status.status}`);

            if (status.status === 'COMPLETED') {
                const videoB64 = status.output?.video_base64;
                if (!videoB64) throw new Error('RunPod a terminat dar nu a returnat video');
                fs.writeFileSync(outputPath, Buffer.from(videoB64, 'base64'));
                return;
            }

            if (status.status === 'FAILED') {
                throw new Error(`RunPod job eșuat: ${status.error || 'eroare necunoscută'}`);
            }
            // IN_QUEUE, IN_PROGRESS — continuăm polling
        }
    } finally {
        // Curățăm copia publică
        if (inputPayload._inputPublicPath) {
            try { fs.unlinkSync(inputPayload._inputPublicPath); } catch (_) {}
        }
    }
}

// ══════════════════════════════════════════════════════════════
// ██ DETECT CAPTIONS — frame → EasyOCR/pytesseract → boxes %
// ══════════════════════════════════════════════════════════════
function runDetectWorker(framePath) {
    return new Promise((resolve) => {
        const PYTHON = process.env.PYTHON_BIN || 'python3';
        const WORKER = path.join(__dirname, 'detect_worker.py');
        let stdout = '';
        const worker = spawn(PYTHON, [WORKER, framePath], { stdio: ['ignore', 'pipe', 'pipe'] });
        worker.stdout.on('data', d => { stdout += d; });
        worker.stderr.on('data', d => process.stdout.write(`[DETECT] ${d}`));
        worker.on('close', () => {
            try { resolve(JSON.parse(stdout.trim())); }
            catch (_) { resolve({ width: 0, height: 0, boxes: [] }); }
        });
        worker.on('error', () => resolve({ width: 0, height: 0, boxes: [] }));
    });
}

app.post('/api/detect-captions', authenticate, upload.single('frame'), async (req, res) => {
    const framePath = req.file?.path;
    try {
        if (!req.file) return res.status(400).json({ error: 'Frame lipsă.' });

        const result = await runDetectWorker(framePath);

        if (!result.width || !result.height || !result.boxes.length) {
            return res.json({ boxes: [], count: 0 });
        }

        const boxesPct = result.boxes.map(b => ({
            x: Math.round((b.x / result.width)  * 100),
            y: Math.round((b.y / result.height) * 100),
            w: Math.round((b.w / result.width)  * 100),
            h: Math.round((b.h / result.height) * 100),
        })).filter(b => b.w > 1 && b.h > 1 && b.w <= 100 && b.h <= 100);

        console.log(`[DETECT] ✅ ${boxesPct.length} zone → ${JSON.stringify(boxesPct)}`);
        res.json({ boxes: boxesPct, count: boxesPct.length });
    } catch (e) {
        console.error('[DETECT] Eroare:', e.message);
        res.json({ boxes: [], count: 0 });
    } finally {
        if (framePath && fs.existsSync(framePath)) {
            try { fs.unlinkSync(framePath); } catch (_) {}
        }
    }
});

// ══════════════════════════════════════════════════════════════
// ██ MAIN ROUTE
// ══════════════════════════════════════════════════════════════
app.post('/api/remove-caption', authenticate, upload.single('video'), async (req, res) => {
    const inputPath = req.file?.path;

    try {
        if (!req.file) return res.status(400).json({ error: 'Video lipsă.' });

        const model = req.body.model || 'standard';
        if (!CREDIT_COSTS[model]) return res.status(400).json({ error: 'Model invalid.' });
        const cost = CREDIT_COSTS[model];

        console.log(`\n[CUT] 📥 userId=${req.userId} | model=${model} | file=${req.file.originalname} | size=${req.file.size}`);

        // Credit check
        let balance;
        try {
            balance = await hubAPI.checkCredits(req.userId);
        } catch (e) {
            console.error('[CUT] ❌ HUB checkCredits failed:', e.message);
            if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
            return res.status(503).json({ error: 'Serverul de credite nu răspunde. Încearcă din nou.' });
        }

        if (balance.credits < cost) {
            if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
            return res.status(403).json({ error: `Cost: ${cost} Credite. Fonduri insuficiente.` });
        }

        // Video metadata
        let meta;
        try {
            meta = await getVideoMeta(inputPath);
        } catch (e) {
            if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
            return res.status(500).json({ error: 'Eroare la analiza metadatelor video.' });
        }

        const { width, height, fps } = meta;

        // Box parsing + clamping
        const clampedBoxes = parseAndClampBoxes(req.body, width, height);
        if (clampedBoxes.length === 0) {
            if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
            return res.status(400).json({ error: 'Niciun box valid. Trage box-urile mai în interiorul cadrului.' });
        }

        const videoId = Date.now();
        const outputPath = path.join(DOWNLOAD_DIR, `clean_${videoId}.mp4`);

        console.log(`[CUT ${videoId}] ▶️  START | model=${model} | ${width}x${height} @ ${fps}fps | ${clampedBoxes.length} box(uri)`);
        clampedBoxes.forEach((b, i) => console.log(`[CUT ${videoId}] ✂️  Box #${i+1}: x=${b.x} y=${b.y} w=${b.w} h=${b.h}`));

        // Per-model timeout
        const timeouts = { fast: 60_000, standard: 15 * 60_000, ai: 32 * 60_000 };
        let timedOut = false;
        const timeoutHandle = setTimeout(() => {
            timedOut = true;
            console.error(`[CUT ${videoId}] ⏰ TIMEOUT`);
            if (!res.headersSent) res.status(504).json({ error: 'Procesarea a durat prea mult. Încearcă cu un video mai scurt.' });
        }, timeouts[model]);

        const t0 = Date.now();

        try {
            if (model === 'fast') {
                await processFast(inputPath, outputPath, clampedBoxes);
            } else if (model === 'standard') {
                await processStandard(inputPath, outputPath, clampedBoxes, width, height, fps, videoId);
            } else if (model === 'ai') {
                await processAI(inputPath, outputPath, clampedBoxes, width, height, fps, videoId);
            }
        } catch (e) {
            clearTimeout(timeoutHandle);
            if (timedOut) return;
            console.error(`[CUT ${videoId}] ❌ FAILED:`, e.message);
            if (fs.existsSync(outputPath)) try { fs.unlinkSync(outputPath); } catch (_) {}
            if (!res.headersSent) res.status(500).json({ error: 'Eroare la procesare. Încearcă o zonă puțin mai mică.' });
            return;
        }

        clearTimeout(timeoutHandle);
        if (timedOut) return;

        const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
        console.log(`[CUT ${videoId}] ✅ DONE în ${elapsed}s → clean_${videoId}.mp4`);

        // Deduct credits
        let creditsLeft = 0;
        try {
            const result = await hubAPI.useCredits(req.userId, cost);
            creditsLeft = result.credits;
        } catch (_) {}

        res.json({ status: 'ok', downloadUrl: `/download/clean_${videoId}.mp4`, creditsLeft });

        setTimeout(() => {
            try { if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath); } catch (_) {}
        }, FILE_TTL_MS);

    } catch (e) {
        console.error('[CUT] ❌ Excepție neașteptată:', e);
        if (!res.headersSent) res.status(500).json({ error: e.message || 'Eroare internă neașteptată.' });
    } finally {
        if (inputPath && fs.existsSync(inputPath)) {
            try { fs.unlinkSync(inputPath); } catch (_) {}
        }
    }
});

app.get('/download/:filename', (req, res) => {
    const file = path.join(DOWNLOAD_DIR, req.params.filename);
    if (fs.existsSync(file)) res.sendFile(file); else res.status(404).send('Expirat.');
});

// ── Global error handler ──────────────────────────────────────
app.use((err, req, res, next) => {
    console.error('💥 Express error:', err.message, err.code || '');
    if (err.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: 'Videoul este prea mare (max 200MB).' });
    if (err.code === 'LIMIT_UNEXPECTED_FILE' || err.code === 'LIMIT_PART_COUNT') return res.status(400).json({ error: 'Fișier invalid trimis.' });
    if (err.code === 'ECONNABORTED' || err.code === 'ECONNRESET' || err.message?.includes('aborted')) return res.status(499).json({ error: 'Conexiune întreruptă. Reîncearcă.' });
    if (res.headersSent) return next(err);
    res.status(err.status || 500).json({ error: err.message || 'Eroare internă neașteptată.' });
});

app.use((req, res) => res.status(404).json({ error: `Endpoint inexistent: ${req.method} ${req.path}` }));

app.listen(PORT, () => console.log(`🚀 Caption Eraser rulează pe portul ${PORT}!`));
