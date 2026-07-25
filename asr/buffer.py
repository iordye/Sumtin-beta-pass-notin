# asr/buffer.py

"""
ASR input buffer — prepares VAD utterances for the ASR engine.

Responsibility:
    Receive complete utterances from the VAD state machine,
    validate them, prepare them for transcription, and hand
    them to the ASR engine queue.

This module is the quality gate between raw speech detection
and expensive model inference. Its job is to ensure that
what reaches Whisper is worth transcribing.

Why a separate module and not just code in the coordinator?
    - Testable in isolation: we can test preparation logic
      without loading Whisper or running the full pipeline
    - Single responsibility: buffer logic is complex enough
      to warrant its own module
    - Replaceable: we could swap in a different preparation
      strategy (e.g., add noise reduction) without touching
      the coordinator or ASR engine
"""

import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings
from utils.logging_config import get_logger, TimingLogger
from utils.audio_utils import (
    compute_rms,
    compute_peak,
    pad_audio,
    normalize_audio,
    samples_to_seconds,
    ms_to_samples,
    validate_audio_chunk,
    SILENCE_RMS_FLOOR,
)
import logging

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# PREPARED UTTERANCE — the output of the buffer
# ─────────────────────────────────────────────────────────────────

@dataclass
class PreparedUtterance:
    """
    A validated, prepared utterance ready for ASR transcription.

    This is what flows from the buffer into the ASR engine.
    It carries the audio plus enough metadata to understand
    the context and debug any issues.

    Fields:
        audio:          The processed audio, ready for Whisper.
                        float32, mono, 16kHz, padded, normalized.
        raw_duration_s: Duration of the original speech (before padding).
                        Used for timing analysis and logging.
        padded_duration_s: Duration of audio including padding.
                           This is what Whisper actually processes.
        rms_energy:     RMS of the original speech audio.
                        Low values suggest quiet or possibly corrupt audio.
        peak_amplitude: Peak amplitude of original speech.
                        > 0.95 suggests clipping upstream.
        chunk_count:    How many 32ms VAD chunks made this utterance.
                        Useful for debugging unexpected short utterances.
        utterance_id:   Unique identifier for this utterance.
                        Trace it through the pipeline in logs.
        created_at:     Unix timestamp when this utterance was prepared.
                        Used to compute end-to-end latency.
        was_normalized: Whether normalization was applied.
        was_padded:     Whether padding was applied.
    """
    audio:              np.ndarray
    raw_duration_s:     float
    padded_duration_s:  float
    rms_energy:         float
    peak_amplitude:     float
    chunk_count:        int
    utterance_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at:         float = field(default_factory=time.time)
    was_normalized:     bool = False
    was_padded:         bool = False

    def __repr__(self) -> str:
        return (
            f"PreparedUtterance("
            f"id={self.utterance_id} "
            f"duration={self.raw_duration_s:.2f}s "
            f"rms={self.rms_energy:.4f} "
            f"chunks={self.chunk_count})"
        )


# ─────────────────────────────────────────────────────────────────
# BUFFER REJECTION REASON — why an utterance was rejected
# ─────────────────────────────────────────────────────────────────

class RejectionReason:
    """
    Constants for why an utterance was rejected.

    Using constants instead of magic strings prevents typos
    and makes rejection reasons searchable across your codebase.
    """
    TOO_SHORT       = "too_short"
    TOO_LONG        = "too_long"
    TOO_QUIET       = "too_quiet"
    INVALID_AUDIO   = "invalid_audio"
    EMPTY           = "empty"


# ─────────────────────────────────────────────────────────────────
# ASR BUFFER
# ─────────────────────────────────────────────────────────────────

class ASRBuffer:
    """
    Validates and prepares utterances for ASR transcription.

    This class is stateless with respect to the audio stream —
    it processes one utterance at a time and has no memory of
    previous utterances. This makes it easy to test and reason about.

    The only state it holds is configuration (settings) and
    diagnostic counters (for monitoring).

    Usage:
        buffer = ASRBuffer(settings)

        # In the pipeline coordinator, when VAD fires UTTERANCE_COMPLETE:
        result = buffer.prepare(utterance_audio, chunk_count=87)

        if result is not None:
            asr_queue.put(result)
        # If result is None, utterance was rejected (too short, too quiet, etc.)

    Why not raise exceptions on rejection?
        Rejection is normal and expected, not exceptional.
        During a meeting, lots of short sounds (breathing, paper, 
        chair squeaks) will reach the buffer and be correctly rejected.
        Using exceptions for normal control flow is an anti-pattern.
        Returning None signals "nothing to do" cleanly.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Convert time thresholds to sample counts once at init
        # (not on every call — avoid recomputing in hot path)
        sr = settings.audio.sample_rate

        self._min_samples = ms_to_samples(
            settings.vad.min_speech_duration_ms, sr
        )
        self._max_samples = int(
            settings.asr.max_utterance_duration_s * sr
        )

        # Minimum RMS to consider audio "real speech"
        # Below this, the audio is effectively silence despite VAD
        # detecting it. This catches mic disconnections and buffer errors.
        self._min_rms_threshold = 0.001

        # Diagnostic counters — track how the buffer is performing
        self._total_received  = 0
        self._total_accepted  = 0
        self._total_rejected  = 0

        logger.info(
            f"ASRBuffer initialized | "
            f"min_duration={settings.vad.min_speech_duration_ms}ms "
            f"({self._min_samples} samples) | "
            f"max_duration={settings.asr.max_utterance_duration_s}s "
            f"({self._max_samples} samples)"
        )

    def prepare(
        self,
        utterance_audio: np.ndarray,
        chunk_count: int = 0,
    ) -> Optional[PreparedUtterance]:
        """
        Validate and prepare a raw utterance for ASR.

        This is the main entry point. Call this every time the VAD
        state machine fires UTTERANCE_COMPLETE.

        Args:
            utterance_audio: Raw concatenated speech audio from
                             the VAD state machine. float32, mono.
            chunk_count:     How many VAD chunks made this utterance.
                             Passed through to PreparedUtterance for
                             diagnostic purposes.

        Returns:
            PreparedUtterance ready for ASR, or None if rejected.
            Check the logs to understand why a None was returned.
        """
        self._total_received += 1

        # ── Step 1: Basic validation ─────────────────────────────
        rejection = self._validate(utterance_audio)
        if rejection is not None:
            self._reject(utterance_audio, rejection, chunk_count)
            return None

        # ── Step 2: Compute pre-processing metrics ───────────────
        # Compute on the original audio before any modifications
        raw_duration_s  = samples_to_seconds(
            len(utterance_audio), self.settings.audio.sample_rate
        )
        rms_energy      = compute_rms(utterance_audio)
        peak_amplitude  = compute_peak(utterance_audio)

        logger.debug(
            f"Utterance received | "
            f"duration={raw_duration_s:.3f}s | "
            f"rms={rms_energy:.4f} | "
            f"peak={peak_amplitude:.4f} | "
            f"chunks={chunk_count}"
        )

        # ── Step 3: Quality pre-screening ────────────────────────
        if rms_energy < self._min_rms_threshold:
            logger.warning(
                f"Utterance rejected: RMS too low | "
                f"rms={rms_energy:.6f} threshold={self._min_rms_threshold} | "
                f"duration={raw_duration_s:.3f}s | "
                f"Possible causes: mic disconnected, buffer overflow upstream"
            )
            self._reject(utterance_audio, RejectionReason.TOO_QUIET, chunk_count)
            return None

        # ── Step 4: Normalize ────────────────────────────────────
        was_normalized = False
        audio = utterance_audio

        if self.settings.asr.normalize_utterances:
            audio = normalize_audio(audio, target_peak=0.9)
            was_normalized = True
            logger.debug(
                f"Utterance normalized | "
                f"original_peak={peak_amplitude:.4f} → 0.9"
            )

        # ── Step 5: Pad with silence ─────────────────────────────
        was_padded = False

        if self.settings.vad.speech_pad_ms > 0:
            audio = pad_audio(
                audio,
                pad_ms=self.settings.vad.speech_pad_ms,
                sample_rate=self.settings.audio.sample_rate,
            )
            was_padded = True

        padded_duration_s = samples_to_seconds(
            len(audio), self.settings.audio.sample_rate
        )

        # ── Step 6: Final clamp ──────────────────────────────────
        # After normalization and padding, ensure padded duration
        # does not exceed Whisper's maximum (30 seconds hard limit)
        max_padded_samples = int(30.0 * self.settings.audio.sample_rate)
        if len(audio) > max_padded_samples:
            logger.warning(
                f"Padded utterance exceeds Whisper 30s limit. "
                f"Trimming from {padded_duration_s:.2f}s to 30.0s"
            )
            audio = audio[:max_padded_samples]
            padded_duration_s = 30.0

        # ── Step 7: Build PreparedUtterance ─────────────────────
        utterance = PreparedUtterance(
            audio             = audio,
            raw_duration_s    = raw_duration_s,
            padded_duration_s = padded_duration_s,
            rms_energy        = rms_energy,
            peak_amplitude    = peak_amplitude,
            chunk_count       = chunk_count,
            was_normalized    = was_normalized,
            was_padded        = was_padded,
        )

        self._total_accepted += 1

        logger.info(
            f"Utterance prepared | "
            f"id={utterance.utterance_id} | "
            f"raw={raw_duration_s:.2f}s → "
            f"padded={padded_duration_s:.2f}s | "
            f"rms={rms_energy:.4f} | "
            f"normalized={was_normalized} | "
            f"accept_rate={self._acceptance_rate:.1%}"
        )

        return utterance

    # ─────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────

    def _validate(self, audio: np.ndarray) -> Optional[str]:
        """
        Run structural validations on the utterance audio.

        Returns the rejection reason string if invalid, else None.

        We check structural validity here (is this a valid array?)
        and leave semantic validity (is there real speech?) to the
        quality pre-screening step above.
        """
        # Empty array
        if audio is None or len(audio) == 0:
            return RejectionReason.EMPTY

        # Wrong dtype — shouldn't happen if VAD state machine is correct,
        # but we check defensively at every boundary
        if audio.dtype != np.float32:
            logger.error(
                f"Utterance has wrong dtype: {audio.dtype}. "
                f"This indicates a bug upstream in the VAD pipeline."
            )
            return RejectionReason.INVALID_AUDIO

        # Too short — not enough audio to transcribe meaningfully
        if len(audio) < self._min_samples:
            duration_ms = (len(audio) / self.settings.audio.sample_rate) * 1000
            logger.debug(
                f"Utterance too short | "
                f"duration={duration_ms:.0f}ms "
                f"minimum={self.settings.vad.min_speech_duration_ms}ms"
            )
            return RejectionReason.TOO_SHORT

        # Too long — safety valve (state machine should prevent this,
        # but we double-check)
        if len(audio) > self._max_samples:
            duration_s = len(audio) / self.settings.audio.sample_rate
            logger.warning(
                f"Utterance too long | "
                f"duration={duration_s:.2f}s "
                f"maximum={self.settings.asr.max_utterance_duration_s}s"
            )
            return RejectionReason.TOO_LONG

        return None  # All checks passed

    def _reject(
        self,
        audio: np.ndarray,
        reason: str,
        chunk_count: int,
    ) -> None:
        """
        Log a rejection and update diagnostics.

        Separated from the main flow so rejection handling is consistent
        and all rejection cases go through one place.
        """
        self._total_rejected += 1
        duration_ms = (
            (len(audio) / self.settings.audio.sample_rate) * 1000
            if audio is not None and len(audio) > 0
            else 0
        )
        logger.debug(
            f"Utterance rejected | "
            f"reason={reason} | "
            f"duration={duration_ms:.0f}ms | "
            f"chunks={chunk_count} | "
            f"total_rejected={self._total_rejected}"
        )

    # ─────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────

    @property
    def _acceptance_rate(self) -> float:
        """Fraction of utterances that passed through to ASR."""
        if self._total_received == 0:
            return 0.0
        return self._total_accepted / self._total_received

    def get_stats(self) -> dict:
        """
        Return buffer performance statistics.

        Call this periodically to monitor pipeline health.
        A low acceptance rate means lots of false alarms from VAD —
        consider tightening VAD thresholds.

        Returns a plain dict so it's easy to log, serialize, or
        send to a metrics system.
        """
        return {
            "total_received":  self._total_received,
            "total_accepted":  self._total_accepted,
            "total_rejected":  self._total_rejected,
            "acceptance_rate": self._acceptance_rate,
        }

    def reset_stats(self) -> None:
        """Reset diagnostic counters. Call between sessions."""
        self._total_received = 0
        self._total_accepted = 0
        self._total_rejected = 0
        logger.debug("ASRBuffer stats reset")