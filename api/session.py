# api/session.py

"""
WebSocket session — one instance per connected user.

Each WebSocket connection gets its own:
    - VAD engine instance (independent hidden state)
    - VAD state machine (independent utterance tracking)
    - ASR buffer (independent quality filtering)
    - Audio queue (independent chunk processing)

Why per-session model instances?
    VAD and state machine are stateful — they remember
    previous chunks. If two users shared one instance,
    User A's speech state would contaminate User B's.
    This was one of the production gaps we identified.
    WebSocket sessions make the isolation explicit.
"""

import asyncio
import threading
import numpy as np
from typing import Optional, Callable

from config.settings import Settings
from vad.engine import VADEngine
from vad.state_machine import VADStateMachine, VADEvent
from asr.buffer import ASRBuffer
from asr.engine import ASREngine, TranscriptionResult
from utils.logging_config import get_logger

logger = get_logger(__name__)


class WebSocketSession:
    """
    Manages the complete ASR pipeline for one WebSocket connection.

    Lifecycle:
        session = WebSocketSession(settings, asr_engine, on_transcript)
        session.start()                     # starts VAD worker thread
        session.push_audio(chunk)           # call for each audio chunk
        session.stop()                      # clean shutdown
        await session.wait_for_completion() # wait for final transcript

    Why pass asr_engine from outside?
        The ASR model is large (~500MB). We load it once at server
        startup and share it across sessions. Transcription is
        serialized through the shared model.

        VAD is tiny (~1MB) and stateful per-session, so each
        session gets its own VAD instance.

    on_transcript callback:
        Called from a background thread when a transcript is ready.
        Must be thread-safe. We use asyncio.run_coroutine_threadsafe
        to safely send results back to the async WebSocket handler.
    """

    def __init__(
        self,
        settings: Settings,
        asr_engine: ASREngine,
        on_transcript: Callable[[TranscriptionResult], None],
        session_id: str = "unknown",
    ):
        self.settings    = settings
        self.asr_engine  = asr_engine
        self.session_id  = session_id
        self._on_transcript = on_transcript

        # Per-session VAD — stateful, must not be shared
        self._vad_engine        = VADEngine(settings)
        self._vad_state_machine = VADStateMachine(settings)
        self._asr_buffer        = ASRBuffer(settings)

        # Audio queue: browser chunks land here
        # VAD worker thread reads from here
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        # Control
        self._stop_event    = threading.Event()
        self._vad_thread:   Optional[threading.Thread] = None
        self._is_running    = False

        # The event loop of the WebSocket handler
        # We need this to safely send results back from the VAD thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(f"Session created | id={session_id}")

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the VAD processing thread.

        Args:
            loop: The event loop running the WebSocket handler.
                  We store this to send transcripts back safely
                  from the background thread.
        """
        self._loop = loop
        self._vad_engine.load()
        self._is_running = True

        self._vad_thread = threading.Thread(
            target=self._vad_worker,
            name=f"VAD-{self.session_id}",
            daemon=True,
        )
        self._vad_thread.start()

        logger.info(f"Session started | id={self.session_id}")

    def push_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Push an audio chunk from the WebSocket into the pipeline.

        Called from the async WebSocket handler for each incoming chunk.
        Non-blocking: if the queue is full, the chunk is dropped.

        Returns:
            True if chunk was queued, False if dropped (queue full).

        Why non-blocking?
            The WebSocket handler is async. We cannot block it
            waiting for the queue — that would freeze the connection.
            If the VAD thread falls behind, we drop chunks rather
            than backing up the WebSocket receive loop.
        """
        try:
            # put_nowait raises QueueFull instead of blocking
            self._audio_queue.put_nowait(audio_chunk)
            return True
        except asyncio.QueueFull:
            logger.warning(
                f"Audio queue full — chunk dropped | "
                f"id={self.session_id}"
            )
            return False

    def stop(self) -> None:
        """Signal the session to stop after processing current audio."""
        self._stop_event.set()
        self._is_running = False
        logger.info(f"Session stop requested | id={self.session_id}")

    async def wait_for_completion(self, timeout: float = 10.0) -> None:
        """
        Wait for the VAD thread to finish processing.

        Called after stop() to ensure the final utterance is
        transcribed before the WebSocket closes.
        """
        if self._vad_thread and self._vad_thread.is_alive():
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._vad_thread.join(timeout=timeout)
            )

    # ── VAD worker thread ─────────────────────────────────────────

    def _vad_worker(self) -> None:
        """
        Runs in background thread.
        Pulls chunks from queue, runs VAD, sends utterances to ASR.
        """
        logger.debug(f"VAD worker started | id={self.session_id}")

        try:
            while not self._stop_event.is_set():
                chunk = self._get_chunk_from_queue()
                if chunk is None:
                    continue

                # Frame-level VAD
                prob = self._vad_engine.process_chunk(chunk)

                # Utterance-level state machine
                transition = self._vad_state_machine.process_chunk(
                    chunk, prob
                )

                if transition.event == VADEvent.UTTERANCE_COMPLETE:
                    self._handle_utterance(transition.utterance)

        except Exception as e:
            logger.error(
                f"VAD worker crashed | id={self.session_id} | "
                f"error={e}",
                exc_info=True,
            )
        finally:
            # Flush any in-progress utterance on session end
            self._flush_final()
            logger.debug(f"VAD worker exited | id={self.session_id}")

    def _get_chunk_from_queue(self) -> Optional[np.ndarray]:
        """
        Get next chunk from the async queue from a sync thread.

        This is the bridge between async (WebSocket) and sync (thread).
        We use run_coroutine_threadsafe to call async queue.get()
        from a regular thread.
        """
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._audio_queue.get(),
                self._loop,
            )
            # Wait up to 0.5s for a chunk
            # Timeout allows checking stop_event regularly
            return future.result(timeout=0.5)
        except Exception:
            return None

    def _handle_utterance(self, raw_audio: np.ndarray) -> None:
        """Prepare and transcribe a complete utterance."""
        prepared = self._asr_buffer.prepare(raw_audio, chunk_count=0)
        if prepared is None:
            return

        result = self.asr_engine.transcribe(prepared)

        # Send result back to the WebSocket handler
        # Must use run_coroutine_threadsafe because we are in a
        # background thread and the WebSocket send() is async
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._send_transcript(result),
                self._loop,
            )

    def _flush_final(self) -> None:
        """Flush final in-progress utterance on session end."""
        transition = self._vad_state_machine.flush()
        if transition is not None and transition.utterance is not None:
            logger.info(
                f"Flushing final utterance | id={self.session_id}"
            )
            self._handle_utterance(transition.utterance)

    async def _send_transcript(self, result: TranscriptionResult) -> None:
        """Coroutine that delivers transcript to the callback."""
        try:
            await self._on_transcript(result)
        except Exception as e:
            logger.error(
                f"Failed to send transcript | "
                f"id={self.session_id} | error={e}"
            )