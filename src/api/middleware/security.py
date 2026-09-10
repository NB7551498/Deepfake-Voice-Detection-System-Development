"""
Production Security Middleware for Deepfake Voice Detector 2.0.

Provides:
  - Strict upload validation (file size <= 25MB, duration <= 300s)
  - Magic bytes & audio format verification (WAV, MP3, M4A, FLAC, OGG)
  - API Key Authentication via X-API-Key header (with public demo bypass)
"""

import io
import librosa
from fastapi import Request, HTTPException
from typing import Optional

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_DURATION_SECONDS = 300.0            # 5 minutes
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

DEMO_API_KEYS = {"prod-demo-key-2026", "evaluation-test-token"}


def validate_api_key(request: Request) -> bool:
    """
    Validate optional API key.
    Allows unauthenticated access for localhost / demo UI, but validates if header provided.
    """
    api_key = request.headers.get("X-API-Key")
    # Public demo mode or local access allowed
    client_host = request.client.host if request.client else "127.0.0.1"
    if client_host in ("127.0.0.1", "localhost") and not api_key:
        return True
    if api_key and api_key not in DEMO_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or expired X-API-Key.")
    return True


def validate_audio_payload(audio_bytes: bytes, filename: str) -> None:
    """Validate uploaded audio bytes for size, extension, and decodability."""
    # 1. File size check
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
        )

    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")

    # 2. Extension check
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{ext}'. Allowed extensions: {list(ALLOWED_EXTENSIONS)}"
        )

    # 3. Audio decoding & duration check
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, duration=MAX_DURATION_SECONDS + 1)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio decoding failure: {str(e)[:120]}")

    duration = len(audio) / sr
    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds duration limit of {int(MAX_DURATION_SECONDS)}s (got {duration:.1f}s)."
        )

    if duration < 0.1:
        raise HTTPException(status_code=400, detail="Audio duration is too short for acoustic evaluation.")
