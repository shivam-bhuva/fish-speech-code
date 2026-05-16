import uuid
import torch
import numpy as np
import torchaudio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────
# Yeh sab api.py se import hoga
# ─────────────────────────────────────────
from api import (
    run_tts_chunk,
    VOICES_DIR,
    OUTPUT_DIR,
    CHUNK_SIZE,
    SAMPLE_RATE,
)

router = APIRouter()

# GPU ek hi hai isliye max_workers=1
executor = ThreadPoolExecutor(max_workers=1)


# ─────────────────────────────────────────
# Job Store (RAM mein)
# ─────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@dataclass
class JobInfo:
    job_id:       str
    name:         str
    status:       JobStatus = JobStatus.PENDING
    total_chunks: int = 0
    done_chunks:  int = 0
    error:        str = ""
    output_url:   str = ""
    created_at:   str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at:  str = ""


jobs: dict[str, JobInfo] = {}


# ─────────────────────────────────────────
# Background Worker
# ─────────────────────────────────────────
def process_job(job_id: str, name: str, data: list):
    job = jobs[job_id]
    job.status = JobStatus.RUNNING

    try:
        # Step 1: Pehle saare chunks collect karo (count ke liye)
        all_chunks = []
        for item in data:
            voices_raw = item.get("voices", "")
            texts_raw  = item.get("texts", "")
            speed      = float(item.get("speed", 0.8))
            speed      = max(0.1, min(1.0, speed))

            voices = [voices_raw] if isinstance(voices_raw, str) else voices_raw
            texts  = [texts_raw]  if isinstance(texts_raw,  str) else texts_raw

            for voice_name, text in zip(voices, texts):
                text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
                for chunk in text_chunks:
                    if chunk.strip():
                        all_chunks.append((voice_name, chunk, speed))

        job.total_chunks = len(all_chunks)

        if job.total_chunks == 0:
            raise Exception("No valid text chunks found in data")

        # Step 2: Ek ek chunk process karo
        all_audio = []
        for voice_name, chunk, speed in all_chunks:
            voice_file = None
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                p = VOICES_DIR / f"{voice_name}{ext}"
                if p.exists():
                    voice_file = p
                    break

            if not voice_file:
                raise Exception(f"Voice '{voice_name}' not found. Upload via /add-voice first.")

            voice_bytes = voice_file.read_bytes()
            audio_data  = run_tts_chunk(voice_bytes, chunk, speed)
            all_audio.append(audio_data)

            job.done_chunks += 1  # Live progress update

        # Step 3: Sab audio jodo aur save karo
        final_audio     = np.concatenate(all_audio, axis=-1)
        final_tensor    = torch.from_numpy(final_audio.astype(np.float32)).unsqueeze(0)

        safe_name       = "".join(c for c in name if c.isalnum() or c in "-_")
        output_filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.wav"
        output_path     = OUTPUT_DIR / output_filename
        torchaudio.save(str(output_path), final_tensor, SAMPLE_RATE)

        job.status      = JobStatus.DONE
        job.output_url  = f"/output/{output_filename}"
        job.finished_at = datetime.now().isoformat()

    except Exception as e:
        job.status      = JobStatus.FAILED
        job.error       = str(e)
        job.finished_at = datetime.now().isoformat()


# ─────────────────────────────────────────
# Route 1: Job Submit
# POST /generate-job
# {
#   "name": "abc",
#   "data": [
#     {"voices": "dhara", "texts": "Hello world", "speed": 0.8},
#     {"voices": "dhara", "texts": "Kya haal hai", "speed": 0.7}
#   ]
# }
# ─────────────────────────────────────────
@router.post("/generate-job")
async def generate_job(request: Request):
    body = await request.json()

    name = body.get("name", "output")
    data = body.get("data", [])

    if not data:
        raise HTTPException(status_code=400, detail="'data' array required")

    job_id       = str(uuid.uuid4())
    jobs[job_id] = JobInfo(job_id=job_id, name=name)

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, process_job, job_id, name, data)

    return JSONResponse({
        "status":  "accepted",
        "job_id":  job_id,
        "message": "Job queued. Poll /job-status/{job_id} for progress."
    })


# ─────────────────────────────────────────
# Route 2: Job Status
# GET /job-status/{job_id}
# ─────────────────────────────────────────
@router.get("/job-status/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress_pct = 0
    if job.total_chunks > 0:
        progress_pct = round((job.done_chunks / job.total_chunks) * 100, 1)

    response = {
        "job_id":       job.job_id,
        "name":         job.name,
        "status":       job.status,
        "progress":     f"{progress_pct}%",
        "done_chunks":  job.done_chunks,
        "total_chunks": job.total_chunks,
        "created_at":   job.created_at,
    }

    if job.status == JobStatus.DONE:
        response["output_url"]  = job.output_url
        response["finished_at"] = job.finished_at

    if job.status == JobStatus.FAILED:
        response["error"]       = job.error
        response["finished_at"] = job.finished_at

    return JSONResponse(response)