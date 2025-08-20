import os
import io
import uuid
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

import numpy as np
import soundfile as sf

# Coqui TTS
from TTS.api import TTS

# ---- Config ----
MODEL_ID = os.getenv("MODEL_ID", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE   = os.getenv("DEVICE", "cuda")  # "cuda" or "cpu"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

app = FastAPI(title="XTTS Web API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down if exposing publicly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup (first call will download weights if needed)
tts = TTS(MODEL_ID).to(DEVICE)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "default_language": DEFAULT_LANGUAGE
    }

def _normalize_to_mono_44k1_wav(in_bytes: bytes) -> str:
    """Normalize uploaded audio to mono/44.1k/PCM16 and return temp file path."""
    # Read any format soundfile supports
    data, sr = sf.read(io.BytesIO(in_bytes), always_2d=False)
    # Convert to mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Resample if needed (simple linear resample via soundfile not available; use numpy ratio)
    target_sr = 44100
    if sr != target_sr:
        # naive resample using numpy (ok for reference clips); for better quality, install librosa
        import math
        import numpy as np
        # Use simple polyphase-style with np.interp as a lightweight fallback
        duration = len(data) / sr
        t_old = np.linspace(0, duration, num=len(data), endpoint=False)
        t_new = np.linspace(0, duration, num=int(duration * target_sr), endpoint=False)
        data = np.interp(t_new, t_old, data)
        sr = target_sr
    # Peak normalize to -1 dBFS
    peak = np.max(np.abs(data)) if data.size else 1.0
    if peak > 0:
        data = (data / peak) * (10 ** (-1/20))  # -1 dBFS

    # Write to a temp WAV file
    tmp_wav = os.path.join(tempfile.gettempdir(), f"ref_{uuid.uuid4().hex}.wav")
    sf.write(tmp_wav, data, sr, subtype="PCM_16")
    return tmp_wav

@app.post("/synthesize")
async def synthesize(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    language: Optional[str] = Form(DEFAULT_LANGUAGE),
    normalize: Optional[bool] = Form(True),
    speaker: UploadFile = File(...)
):
    try:
        # Read uploaded reference audio
        ref_bytes = await speaker.read()

        # Optionally standardize to mono/44.1k WAV for best XTTS results
        if normalize:
            speaker_path = _normalize_to_mono_44k1_wav(ref_bytes)
        else:
            # Write raw upload to temp file as-is
            speaker_path = os.path.join(tempfile.gettempdir(), f"ref_{uuid.uuid4().hex}_{speaker.filename or 'speaker'}")
            with open(speaker_path, "wb") as f:
                f.write(ref_bytes)

        # Temp output path
        out_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")

        # Run XTTS zero-shot synthesis
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language or DEFAULT_LANGUAGE,
            file_path=out_path
        )

        # Schedule cleanup
        def _cleanup(paths):
            for p in paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        background_tasks.add_task(_cleanup, [speaker_path, out_path])

        # Return audio
        return FileResponse(out_path, media_type="audio/wav", filename="tts.wav")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
def root():
    # Serve the index.html from the same directory
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>XTTS Web API</h3><p>POST /synthesize with a reference file.</p>")
