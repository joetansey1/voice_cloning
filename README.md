# XTTS Web App & API (Zero-Shot Voice Clone)

Local, consent-first voice cloning built on **Coqui XTTS v2**.  
This repo gives you a tiny **web app** (`index.html`) and a **FastAPI server** (`server.py`) that can synthesize speech in a cloned voice from a short reference clip (15–60s).

> ⚠️ **Ethics & consent:** Clone only voices you own or have **explicit permission** to use. Do not deploy for deception or impersonation. Label outputs if you share them.
>
> How it works (high level)
server.py loads Coqui XTTS v2 via TTS.api.TTS(MODEL_ID).
On /synthesize, it: reads the speaker file (your 15–60s reference), runs XTTS in zero-shot mode with your text and language, streams back a WAV file. run on a web server with sample app in index.html here, basic functionality is to upload a .wav as the 1-shot source speaker audio, and a text box prompt to create the audio, and a 'generate' synthetic audio button.

index.html is a tiny upload form with an audio player for preview.

---

## Features

- **Zero-shot TTS** with XTTS v2 — no training step required
- **Local** inference (GPU recommended; CPU works for tests)
- **Web UI** to upload a reference clip + enter text
- **API** endpoint to integrate from scripts/apps
- **Multilingual** model; pass `language` to switch voices/languages

---

## Quickstart

### 1) Create env + install deps

```bash
python3 -m venv venv
source venv/bin/activate

# Install FastAPI, Coqui TTS, and friends
pip install "fastapi[all]" uvicorn TTS soundfile pydantic

# Install PyTorch for your platform (GPU strongly recommended)
# CUDA 11.8 example (Linux):
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# Or CPU-only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

2) Configure (optional)

Create a .env (or edit the defaults in server.py):
MODEL_ID=tts_models/multilingual/multi-dataset/xtts_v2
DEVICE=cuda          # or "cpu"
DEFAULT_LANGUAGE=en  # e.g. en, es, fr, de, it, pt, pl, tr, ru, zh, ja, ko

3) Run the server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload


Open the web UI at: http://localhost:7860

API
POST /synthesize → WAV audio

Consumes (multipart/form-data):

text (str) — required

speaker (file) — reference audio (.wav/.mp3); mono WAV 44.1kHz recommended

language (str, optional) — default en

normalize (bool, optional) — default true

Produces:

audio/wav binary (the synthesized speech)

cURL example:

curl -X POST "http://localhost:7860/synthesize" \
  -F "text=Welcome back to the late-night lab." \
  -F "language=en" \
  -F "speaker=@myref.wav" \
  --output out.wav


The web UI (index.html) submits the same form under the hood and plays the result in an <audio> element.

.
├── server.py            # FastAPI app (POST /synthesize)
├── index.html           # Minimal front-end (upload + text box + player)
├── requirements.txt     # Optional: pin package versions
├── .env.example         # Optional: sample environment settings
└── README.md

