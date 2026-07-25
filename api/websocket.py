# api/websocket.py

"""
WebSocket endpoint for realtime speech transcription.

Protocol:
    Client connects to ws://server/ws/transcribe?api_key=YOUR_KEY

    Client sends:  raw audio bytes (float32, 16kHz, mono, 512 samples)
    Server sends:  JSON transcript messages

    Message format from server:
    {
        "type": "transcript",
        "text": "how you dey",
        "language": "pidgin",
        "latency_ms": 423,
        "is_empty": false
    }

    Control messages from server:
    {"type": "ready"}        connection established, send audio
    {"type": "error", "message": "..."}   something went wrong
    {"type": "goodbye"}      server is closing the connection

    Client can send:
    b"STOP"                  signal end of session (flushes final utterance)
"""

# api/websocket.py 

import json
import uuid
import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional

from config.settings import Settings
from tts.config import TTSConfig
from vad.engine import SILERO_CHUNK_SIZE
from asr.engine import ASREngine, TranscriptionResult
from api.session import WebSocketSession
from llm.orchestrator import Orchestrator
from llm.config import LLMConfig
from llm.schemas import OrchestratorInput, SupportedLanguage
from tts.engine import TTSEngine
from tts.audio_output import AudioOutputManager, synthesize_and_send
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def websocket_transcribe(
    websocket:   WebSocket,
    asr_engine:  ASREngine,
    llm_engine,           
    tts_engine:  TTSEngine,      
    settings:    Settings,
    llm_config:  LLMConfig,
    tts_config:  TTSConfig,
    api_key:     Optional[str] = None,
):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]

    logger.info(f"WebSocket connected | id={session_id}")

    # ── Create per-session LLM orchestrator ───────────────────────
    # One orchestrator per WebSocket connection.
    # It owns this session's conversation history.
    # The llm_engine is shared across all sessions (stateless).
    orchestrator = Orchestrator(
        config     = llm_config,
        engine     = llm_engine,
        session_id = session_id,
    )

    audio_manager = AudioOutputManager(
        config     = tts_config,     # TTSConfig instance
        session_id = session_id,
    )

    # ── Transcript callback — THE integration bridge ──────────────
    # This function is called by the ASR session every time
    # SBPN finishes transcribing a complete utterance.
    # It is where ASR output becomes LLM input.
    async def on_transcript(asr_result: TranscriptionResult) -> None:
        """
        Called when ASR produces a transcript.
        Feeds it into the LLM Brain and sends response to client.

        This is the exact handoff point between the two services.
        """

        # ── Step 1: Filter empty ASR results ─────────────────────
        # SBPN sometimes returns empty for near-silence.
        # Do not send empty transcripts to the LLM.
        if asr_result.is_empty:
            logger.debug(
                f"Empty transcript skipped | id={session_id}"
            )
            return

        logger.info(
            f"ASR → LLM handoff | "
            f"id={session_id} | "
            f"transcript='{asr_result.clean_text}' | "
            f"language={asr_result.detected_language}"
        )

        # ── Step 2: Send transcript to client immediately ─────────
        # Show what the user said BEFORE the LLM responds.
        # This gives immediate feedback — user sees their
        # words on screen while waiting for LLM response.
        await websocket.send_text(json.dumps({
            "type":       "transcript",
            "text":       asr_result.clean_text,
            "language":   asr_result.detected_language,
            "latency_ms": round(asr_result.transcription_ms),
        }))

        # ── Step 3: Map SBPN language → SupportedLanguage ────────
        # SBPN returns strings like "english", "pidgin", "yoruba".
        # Our LLM schemas use the SupportedLanguage enum.
        # We map between them here — the only place this
        # translation happens.
        try:
            language = SupportedLanguage(
                asr_result.detected_language or "english"
            )
        except ValueError:
            # SBPN returned a language our LLM does not support
            # Fall back to unknown so orchestrator uses safe default
            logger.warning(
                f"Unknown language from SBPN: "
                f"'{asr_result.detected_language}' | "
                f"falling back to UNKNOWN"
            )
            language = SupportedLanguage.UNKNOWN

        # ── Step 4: Build OrchestratorInput ──────────────────────
        # This is the formal handoff object.
        # ASR data is now in LLM Brain format.
        orch_input = OrchestratorInput(
            transcript        = asr_result.clean_text,
            detected_language = language,
            session_id        = session_id,
            audio_duration_s  = asr_result.audio_duration_s,
        )

        # ── Step 5: Send "thinking" indicator to client ───────────
        # LLM takes 300-700ms. Let the user know we received
        # their message and are processing it.
        # Send transcript to browser (show what user said)
        await websocket.send_text(json.dumps({
            "type":     "transcript",
            "text":     asr_result.clean_text,
            "language": asr_result.detected_language,
        }))

        # Send thinking indicator
        await websocket.send_text(json.dumps({"type": "thinking"}))

        # ── Step 6: Process through LLM Brain ────────────────────
        # This is the actual LLM call.
        # Orchestrator handles: history, prompt, engine, cleaning.
        orch_output = await orchestrator.process(orch_input)

        # ── Step 7: Send LLM response to client ──────────────────
        if orch_output.is_usable:
            await websocket.send_text(json.dumps({
                "type":          "response",
                "text":          orch_output.response_text,
                "language":      orch_output.response_language.value,
                "llm_latency_ms": round(orch_output.llm_latency_ms),
                "total_latency_ms": round(orch_output.total_latency_ms),
                # Phase 4: TTS engine will read this text and speak it
                # For now client receives text and can display it
            }))
        else:
            # Response had quality issues or error
            # Still send something — never leave user with silence
            await websocket.send_text(json.dumps({
                "type":    "response",
                "text":    orch_output.response_text,  # fallback message
                "language": orch_output.response_language.value,
                "error":   orch_output.error,
            }))

        if not orch_output.is_usable:
            await websocket.send_text(json.dumps({
                "type":  "error",
                "message": orch_output.error or "Response unavailable",
            }))
            return

        # ← This is the new TTS call
        # Synthesize and stream audio to browser
        await synthesize_and_send(
            text          = orch_output.response_text,
            language      = orch_output.response_language,
            session_id    = session_id,
            tts_engine    = tts_engine,
            audio_manager = audio_manager,
            websocket     = websocket,
        )

    # ── Create ASR session ────────────────────────────────────────
    # WebSocketSession owns the VAD pipeline for this connection.
    # It calls on_transcript whenever an utterance completes.
    asr_session = WebSocketSession(
        settings      = settings,
        asr_engine    = asr_engine,
        on_transcript = on_transcript,
        session_id    = session_id,
    )

    loop = asyncio.get_event_loop()
    asr_session.start(loop)

    await websocket.send_text(json.dumps({"type": "ready"}))

    # ── Audio receive loop ────────────────────────────────────────
    try:
        while True:
            data = await websocket.receive_bytes()

            if data == b"STOP":
                break

            try:
                audio_chunk = np.frombuffer(data, dtype=np.float32)
            except ValueError:
                continue

            if len(audio_chunk) != SILERO_CHUNK_SIZE:
                continue

            asr_session.push_audio(audio_chunk)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected | id={session_id}")

    except Exception as e:
        logger.error(f"WebSocket error | id={session_id} | {e}", exc_info=True)

    finally:
        # ── Clean shutdown ────────────────────────────────────────
        # Stop ASR session (flushes final utterance)
        asr_session.stop()
        await asr_session.wait_for_completion(timeout=15.0)

        # Close orchestrator (saves session stats, clears history)
        orchestrator.close()
        tts_stats = audio_manager.get_stats_dict()
        logger.info(
            f"Session TTS stats | "
            f"session={session_id} | "
            f"{tts_stats}"
        )

        try:
            await websocket.send_text(json.dumps({"type": "goodbye"}))
            await websocket.close()
        except Exception:
            pass

        logger.info(f"Session cleaned up | id={session_id}")