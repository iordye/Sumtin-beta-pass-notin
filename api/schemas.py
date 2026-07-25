# api/schemas.py
from pydantic import BaseModel
from typing import Optional

class TranscriptionResponse(BaseModel):
    transcript:        str
    detected_language: Optional[str]
    duration_s:        float
    latency_ms:        float
    is_empty:          bool
    error:             Optional[str] = None

class HealthResponse(BaseModel):
    status:     str
    model:      str
    asr_loaded: bool
    vad_loaded: bool