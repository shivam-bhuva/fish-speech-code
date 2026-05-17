import os
import uuid
import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import List, Optional
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys
sys.path.insert(0, '/workspace/fish-speech')

import pyrootutils
pyrootutils.setup_root('/workspace/fish-speech', indicator=".project-root", pythonpath=True)

from tools.server.model_manager import ModelManager
from tools.server.inference import inference_wrapper as inference
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
VOICES_DIR = Path("/workspace/storage/voices")
OUTPUT_DIR = Path("/workspace/storage/output")
VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_CHECKPOINT   = "/workspace/checkpoints/s2-pro"
DECODER_CHECKPOINT = "/workspace/checkpoints/s2-pro/codec.pth"
DECODER_CONFIG     = "modded_dac_vq"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE        = 44100

# ─────────────────────────────────────────
# Smart Chunker
# ─────────────────────────────────────────
def smart_chunk_text(text: str, max_chars: int = 300) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        sentences = re.split(r'(?<=[.!?…])\s+', para)

        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sentence.strip())
                continue

            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return [c for c in chunks if c.strip()]


app = FastAPI(title="Fish Speech TTS API")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

model_manager: ModelManager = None

# ─────────────────────────────────────────
# Startup
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global model_manager
    print(f"Loading Fish Speech models on {DEVICE}...")
    model_manager = ModelManager(
        mode="tts",
        device=DEVICE,
        half=True,
        compile=False,
        llama_checkpoint_path=LLAMA_CHECKPOINT,
        decoder_checkpoint_path=DECODER_CHECKPOINT,
        decoder_config_name=DECODER_CONFIG,
    )
    print("✅ Models loaded and warmed up!")


# ─────────────────────────────────────────
# Helper: TTS for one chunk
# reference_text — voice file ke saath jo txt file hai uska content
# ─────────────────────────────────────────
def run_tts_chunk(voice_bytes: bytes, text: str, speed: float = 0.8, reference_text: str = "") -> np.ndarray:
    reference = ServeReferenceAudio(audio=voice_bytes, text=reference_text)
    req = ServeTTSRequest(
        text=text,
        references=[reference],
        format="wav",
        latency="normal",
        streaming=False,
        normalize=True,
        temperature=max(0.1, min(1.0, speed)),
    )

    results = list(inference(req, model_manager.tts_inference_engine))

    if not results:
        raise HTTPException(status_code=500, detail=f"No audio generated for: {text[:50]}")

    audio_arrays = []
    for result in results:
        if isinstance(result, np.ndarray):
            audio_arrays.append(result.flatten())
        elif isinstance(result, bytes):
            arr = np.frombuffer(result, dtype=np.int16).astype(np.float32) / 32768.0
            audio_arrays.append(arr)

    if not audio_arrays:
        raise HTTPException(status_code=500, detail=f"No audio data for: {text[:50]}")

    return np.concatenate(audio_arrays, axis=-1)


# ─────────────────────────────────────────
# API 1: Add Voice
# Form fields:
#   name  — voice ka naam (e.g. "narrator")
#   audio — audio file (.wav/.mp3/.flac/.ogg/.m4a)
#   text  — (optional) audio mein jo bola gaya hai woh text with tags
#            e.g. "[deep dramatic pause] The rain hammered the glass. [exhale]"
#            Yeh .txt file ke roop mein same naam se save hoga
# ─────────────────────────────────────────
@app.post("/add-voice")
async def add_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
    text: str = Form("")        # ← naya optional field
):
    allowed_ext = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    ext = Path(audio.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Invalid format '{ext}'. Allowed: {allowed_ext}")

    # Audio save karo
    save_path = VOICES_DIR / f"{name}{ext}"
    content = await audio.read()
    with open(save_path, 'wb') as f:
        f.write(content)

    # Agar text diya hai to .txt file bhi save karo same naam se
    txt_path = None
    if text.strip():
        txt_path = VOICES_DIR / f"{name}.txt"
        txt_path.write_text(text.strip(), encoding='utf-8')

    return JSONResponse({
        "status": "success",
        "message": f"Voice '{name}' added successfully",
        "voice_name": name,
        "file": str(save_path),
        "reference_text_saved": txt_path is not None,
        "reference_text_file": str(txt_path) if txt_path else None
    })


# ─────────────────────────────────────────
# API 2: List Voices
# ─────────────────────────────────────────
@app.get("/voices")
async def list_voices():
    allowed_ext = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    voices = []
    for f in VOICES_DIR.iterdir():
        if f.suffix.lower() in allowed_ext:
            txt_file = f.with_suffix('.txt')
            voices.append({
                "name": f.stem,
                "has_reference_text": txt_file.exists()
            })
    return {"voices": voices}


# ─────────────────────────────────────────
# API 3: Generate TTS
# POST /generate
# ─────────────────────────────────────────
@app.post("/generate")
async def generate_tts(request: Request):
    body = await request.json()

    voices_raw = body.get("voices", [])
    if isinstance(voices_raw, str):
        voices = [voices_raw]
    else:
        voices = voices_raw

    texts_raw = body.get("texts", [])
    if isinstance(texts_raw, str):
        texts = [texts_raw]
    else:
        texts = texts_raw

    speed = float(body.get("speed", 0.8))
    speed = max(0.1, min(1.0, speed))

    if not voices or not texts:
        raise HTTPException(status_code=400, detail="Both 'voices' and 'texts' required")

    if len(voices) != len(texts):
        raise HTTPException(status_code=400, detail="'voices' and 'texts' must have same length")

    all_audio = []
    total_chunks = 0

    for voice_name, text in zip(voices, texts):
        voice_file = None
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            p = VOICES_DIR / f"{voice_name}{ext}"
            if p.exists():
                voice_file = p
                break

        if not voice_file:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found. Upload via /add-voice first.")

        voice_bytes = voice_file.read_bytes()

        # Reference text load karo agar .txt file hai
        txt_file = voice_file.with_suffix('.txt')
        reference_text = txt_file.read_text(encoding='utf-8').strip() if txt_file.exists() else ""

        text_chunks = smart_chunk_text(text, max_chars=300)
        total_chunks += len(text_chunks)

        for chunk in text_chunks:
            if chunk.strip():
                audio_data = run_tts_chunk(voice_bytes, chunk, speed, reference_text)
                all_audio.append(audio_data)

    if not all_audio:
        raise HTTPException(status_code=500, detail="No audio generated")

    final_audio = np.concatenate(all_audio, axis=-1)
    final_tensor = torch.from_numpy(final_audio.astype(np.float32)).unsqueeze(0)

    output_filename = f"{uuid.uuid4()}.wav"
    output_path = OUTPUT_DIR / output_filename
    torchaudio.save(str(output_path), final_tensor, SAMPLE_RATE)

    return JSONResponse({
        "status": "success",
        "output_url": f"/output/{output_filename}",
        "total_chunks": total_chunks,
        "voice_count": len(voices),
        "speed_used": speed
    })


# ─────────────────────────────────────────
# Job Routes Register
# ─────────────────────────────────────────
from job_routes import router as job_router
app.include_router(job_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=16006, log_level="info")