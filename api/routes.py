# api/routes.py — production version

import re
import os
import wave
import time
import tempfile
import subprocess
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI, UploadFile, File,
    HTTPException, Depends, Request,
    status, WebSocket, Query
)
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response
from llm.engine import LLMEngine as LLMEngineClass
from llm.config import LLMConfig
from config.settings import Settings
from asr.buffer import ASRBuffer
from asr.engine import ASREngine
from vad.engine import VADEngine
from api.websocket import websocket_transcribe
from api.schemas import TranscriptionResponse, HealthResponse
from tts.engine import TTSEngine
from tts.config import TTSConfig
from utils.logging_config import setup_logging, get_logger
import logging

# ── Logging setup ─────────────────────────────────────────────────
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
setup_logging(console_level=log_level, log_to_file=True)
logger = get_logger(__name__)

# ── Settings ──────────────────────────────────────────────────────
settings = Settings()
settings.asr.max_utterance_duration_s = float(
    os.getenv("MAX_AUDIO_DURATION_S", "30")
)

# ── Prometheus metrics ────────────────────────────────────────────
# These are the numbers Prometheus scrapes and Grafana displays.
# They answer the questions you care about in production:
# How many requests? How fast? How often does it fail?

REQUEST_COUNT = Counter(
    "asr_requests_total",
    "Total transcription requests",
    ["status"],          # label: success, error, rejected
)

REQUEST_LATENCY = Histogram(
    "asr_request_duration_seconds",
    "End-to-end request duration in seconds",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0],
)

AUDIO_DURATION = Histogram(
    "asr_audio_duration_seconds",
    "Duration of audio submitted for transcription",
    buckets=[1, 3, 5, 10, 15, 20, 30],
)

TRANSCRIPTION_LATENCY = Histogram(
    "asr_transcription_duration_seconds",
    "Time spent in ASR model inference",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)

ACTIVE_REQUESTS = Gauge(
    "asr_active_requests",
    "Number of requests currently being processed",
)

MODELS_LOADED = Gauge(
    "asr_models_loaded",
    "Whether AI models are loaded (1=yes, 0=no)",
)

# ── Module-level model instances ──────────────────────────────────
# Loaded once at startup, shared across all requests in this worker.
# Each Uvicorn worker process has its own copy of these.
vad_engine = VADEngine(settings)
asr_engine = ASREngine(settings)
asr_buffer = ASRBuffer(settings)
llm_config  = LLMConfig()
llm_engine  = LLMEngineClass(llm_config)  
tts_config = TTSConfig()
tts_engine = TTSEngine(tts_config)

# ── Lifespan: startup and shutdown logic ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Validate LLM config at startup
    llm_config.validate()
    logger.info("Loading ASR models...")
    try:
        vad_engine.load()
        asr_engine.load()
        logger.info("Loading LLM engine...")
        llm_engine.load()
        MODELS_LOADED.set(1)
        logger.info("Models loaded. Ready to serve requests.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        MODELS_LOADED.set(0)
        # Don't raise — let the server start but return unhealthy
        # so load balancers don't route traffic here

    yield  # Application runs here

    # Shutdown
    logger.info("Unloading models on shutdown...")
    vad_engine.unload()
    asr_engine.unload()
    llm_engine.unload()
    logger.info("Shutdown complete.")


# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="SBPN Nigerian Languages ASR",
    description=(
        "Speech-to-text for Yorùbá, Hausa, Igbo, "
        "Nigerian Pidgin, and English"
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production if you want to hide the API schema
    # docs_url=None, redoc_url=None
)


# ── Authentication ────────────────────────────────────────────────
# APIKeyHeader reads a header named "X-API-Key" from the request
# auto_error=False means we handle missing keys ourselves
# (gives us cleaner error messages than FastAPI's default)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load valid API key from environment
# In production this comes from a secret manager
# For demo it comes from the .env file
VALID_API_KEY = os.getenv("API_KEY")


async def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    """
    Dependency that validates the API key on every protected endpoint.

    FastAPI's Depends() system injects this automatically.
    If verification fails, the endpoint function never runs.

    Why not check VALID_API_KEY in the endpoint directly?
    Using Depends() means authentication is declared at the
    route level, not buried in the function body. It is
    reusable across many endpoints without code duplication.
    """
    if VALID_API_KEY is None:
        # API_KEY not set in environment — allow all requests
        # This makes local development easier (no key needed)
        # In production, always set API_KEY
        logger.warning(
            "API_KEY not set — running without authentication. "
            "Set API_KEY environment variable for production."
        )
        return

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )

    if api_key != VALID_API_KEY:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


# ── Request ID middleware ─────────────────────────────────────────
# Adds a unique ID to every request for end-to-end tracing.
# When something fails, you search logs by request_id to see
# exactly what happened across every component.

import uuid

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Helpers ───────────────────────────────────────────────────────

def resample_to_16k(input_path: str) -> str:
    output_path = input_path + "_16k.wav"
    result = subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_path, "-loglevel", "quiet"
    ], capture_output=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=422,
            detail=f"Audio preprocessing failed: {result.stderr.decode()}"
        )
    return output_path


def load_wav_as_numpy(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    return (audio_int16 / 32768.0).astype(np.float32)


_LANG_TAG = re.compile(r'<([^>]+)>')

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health endpoint. No authentication required.

    Returns two states:
        200 OK with status=healthy: models loaded, ready for traffic
        200 OK with status=degraded: server up but models not loaded

    Why not return 503 when degraded?
    Some load balancers stop routing to 503 responses. For a demo,
    we want the server accessible even if models failed to load,
    so the user can see a meaningful error rather than a 503 page.
    """
    is_healthy = asr_engine.is_loaded and vad_engine.is_loaded
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model=settings.asr.model_name,
        asr_loaded=asr_engine.is_loaded,
        vad_loaded=vad_engine.is_loaded,
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Prometheus scrapes this every 15 seconds.
    No authentication — metrics contain no sensitive data.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.websocket("/ws/transcribe")
async def ws_transcribe(
    websocket: WebSocket,
    api_key: Optional[str] = Query(default=None),
):
    """
    WebSocket endpoint for realtime transcription.
    Connect with: ws://yourserver/ws/transcribe?api_key=YOUR_KEY
    """
    await websocket_transcribe(
        websocket  = websocket,
        asr_engine = asr_engine,
        llm_engine = llm_engine,
        settings   = settings,
        llm_config = llm_config, 
        api_key    = api_key,
        tts_engine=tts_engine,
        tts_config=tts_config
    )

@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    dependencies=[Depends(verify_api_key)],  # Auth applied here
)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Transcribe an uploaded audio file.

    Accepts: WAV, MP3, M4A, FLAC (ffmpeg handles conversion)
    Returns: transcript text, detected language, timing metadata

    Authentication: X-API-Key header required
    Rate limiting: enforced by Nginx (10 req/s per IP)
    Max file size: 50MB (enforced by Nginx)
    Max audio duration: 30s (enforced here)
    """
    request_id = getattr(request.state, "request_id", "unknown")
    request_start = time.perf_counter()
    ACTIVE_REQUESTS.inc()

    tmp_path = None
    resampled_path = None

    try:
        # ── File size check ───────────────────────────────────────
        # Read the file once, check size before doing anything else
        audio_bytes = await file.read()
        file_size = len(audio_bytes)

        if file_size > MAX_FILE_SIZE_BYTES:
            REQUEST_COUNT.labels(status="rejected").inc()
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: {file_size / 1024 / 1024:.1f}MB. "
                    f"Maximum: {MAX_FILE_SIZE_MB}MB."
                ),
            )

        logger.info(
            f"Transcription request | "
            f"id={request_id} | "
            f"file={file.filename} | "
            f"size={file_size / 1024:.0f}KB"
        )

        # ── Write to temp file ────────────────────────────────────
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        tmp.write(audio_bytes)
        tmp.close()

        # ── Resample ──────────────────────────────────────────────
        resampled_path = resample_to_16k(tmp_path)

        # ── Load audio and check duration ─────────────────────────
        audio = load_wav_as_numpy(resampled_path)
        duration_s = len(audio) / settings.audio.sample_rate

        AUDIO_DURATION.observe(duration_s)

        if duration_s > settings.asr.max_utterance_duration_s:
            REQUEST_COUNT.labels(status="rejected").inc()
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Audio too long: {duration_s:.1f}s. "
                    f"Maximum: {settings.asr.max_utterance_duration_s}s."
                ),
            )

        if duration_s < 0.5:
            REQUEST_COUNT.labels(status="rejected").inc()
            raise HTTPException(
                status_code=422,
                detail="Audio too short. Minimum duration: 0.5 seconds.",
            )

        # ── Prepare utterance ─────────────────────────────────────
        from asr.buffer import PreparedUtterance
        utterance = PreparedUtterance(
            audio             = audio,
            raw_duration_s    = duration_s,
            padded_duration_s = duration_s,
            rms_energy        = float(np.sqrt(np.mean(audio ** 2))),
            peak_amplitude    = float(np.max(np.abs(audio))),
            chunk_count       = 0,
            was_padded        = False,
            was_normalized    = False,
        )

        # ── Transcribe ────────────────────────────────────────────
        transcription_start = time.perf_counter()
        result = asr_engine.transcribe(utterance)
        transcription_s = time.perf_counter() - transcription_start

        TRANSCRIPTION_LATENCY.observe(transcription_s)

        if result.error:
            REQUEST_COUNT.labels(status="error").inc()
            logger.error(
                f"Transcription error | id={request_id} | "
                f"error={result.error}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {result.error}",
            )

        REQUEST_COUNT.labels(status="success").inc()

        total_s = time.perf_counter() - request_start

        logger.info(
            f"Transcription complete | "
            f"id={request_id} | "
            f"text='{result.clean_text[:60]}' | "
            f"lang={result.detected_language} | "
            f"audio={duration_s:.2f}s | "
            f"transcription={transcription_s * 1000:.0f}ms | "
            f"total={total_s * 1000:.0f}ms"
        )

        REQUEST_LATENCY.observe(total_s)

        return TranscriptionResponse(
            transcript        = result.clean_text,
            detected_language = result.detected_language,
            duration_s        = duration_s,
            latency_ms        = result.transcription_ms,
            is_empty          = result.is_empty,
            error             = None,
        )

    finally:
        ACTIVE_REQUESTS.dec()
        for path in [tmp_path, resampled_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass