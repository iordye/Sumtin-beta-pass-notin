# utils/audio_utils.py

"""
Audio math primitives for the realtime ASR system.

These utilities are stateless functions — they take audio in, return
a result, and have no side effects. This makes them easy to test,
easy to reason about, and easy to reuse across modules.

Why stateless?
    VAD engine is stateful (it remembers previous chunks).
    ASR buffer is stateful (it accumulates chunks).
    These utilities should NOT be. Pure functions are predictable.
    Given the same input, they always return the same output.
    This property is called "referential transparency" in computer science.

Dependencies: only numpy. No models, no hardware, no queues.
This module can be imported anywhere with zero side effects.
"""

import numpy as np
import logging
from utils.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

# Float32 is the standard dtype for audio in ML pipelines.
# Why float32 and not float64?
#   - Neural networks (Whisper, Silero VAD) use float32 internally
#   - float64 would be silently converted anyway, wasting memory
#   - float32 gives 7 decimal digits of precision — more than enough
#   - PyTorch tensors default to float32
AUDIO_DTYPE = np.float32

# Valid range for normalized audio.
# Audio outside [-1.0, 1.0] is "clipping" — the signal is distorted.
# Microphone ADCs (analog-to-digital converters) hard-clip at ±1.0.
AUDIO_MIN = -1.0
AUDIO_MAX = 1.0

# Silence floor — anything below this RMS is treated as digital silence.
# Real microphones always have some noise floor (thermal noise, electrical hum).
# 1e-6 is below any real microphone noise floor, so this only catches
# synthetically generated silence (e.g., numpy zeros arrays in tests).
SILENCE_RMS_FLOOR = 1e-6


# ─────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────

def validate_audio_chunk(
    audio: np.ndarray,
    expected_samples: int | None = None,
    caller: str = "unknown",
) -> None:
    """
    Validate that an audio chunk is safe to process.

    Call this at the ENTRY POINT of any function that receives audio
    from an external source (microphone, file, queue). Once audio is
    validated and inside the pipeline, you don't need to re-validate
    at every step.

    This is called "defensive programming" — you validate at the boundary
    where trust ends and your code begins.

    Args:
        audio: The audio array to validate.
        expected_samples: If provided, check that audio has exactly this
            many samples. Use this where chunk size must be exact (e.g., VAD).
        caller: Name of the calling function, for error messages.

    Raises:
        TypeError: If audio is not a numpy array.
        ValueError: If audio has wrong shape, dtype, or sample count.

    Why raise exceptions instead of returning False?
        Because invalid audio reaching VAD or ASR silently produces
        wrong results. We want to CRASH LOUDLY so the bug is immediately
        visible. Silent failures are the worst kind of bug.
    """
    # ── Type check ──────────────────────────────────────────────
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"[{caller}] Audio must be numpy array, got {type(audio).__name__}"
        )

    # ── Dimension check ─────────────────────────────────────────
    # Audio must be 1D — a flat array of samples.
    # (batch processing uses 2D, but we handle that separately)
    if audio.ndim != 1:
        raise ValueError(
            f"[{caller}] Audio must be 1D, got shape {audio.shape}. "
            f"If stereo, convert to mono before passing to this pipeline."
        )

    # ── Empty check ─────────────────────────────────────────────
    if len(audio) == 0:
        raise ValueError(f"[{caller}] Audio array is empty (length 0).")

    # ── Dtype check ─────────────────────────────────────────────
    # We enforce float32 throughout the pipeline.
    # Silero VAD requires float32. Whisper requires float32.
    # Accepting other dtypes would cause silent precision loss or model errors.
    if audio.dtype != AUDIO_DTYPE:
        raise ValueError(
            f"[{caller}] Audio dtype must be float32, got {audio.dtype}. "
            f"Call convert_to_float32() before processing."
        )

    # ── Sample count check ──────────────────────────────────────
    if expected_samples is not None and len(audio) != expected_samples:
        raise ValueError(
            f"[{caller}] Expected {expected_samples} samples, got {len(audio)}. "
            f"Check chunk_size in AudioConfig."
        )

    logger.debug(
        f"Audio validation passed | "
        f"samples={len(audio)} dtype={audio.dtype} caller={caller}"
    )


# ─────────────────────────────────────────────────────────────────
# TYPE CONVERSION
# ─────────────────────────────────────────────────────────────────

def convert_to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio from any integer format to float32 in [-1.0, 1.0].

    Microphones and audio libraries often give you INTEGER formats:
        int16: samples range from -32768 to +32767
        int32: samples range from -2147483648 to +2147483647
        uint8: samples range from 0 to 255

    ML models expect FLOAT formats:
        float32: samples range from -1.0 to +1.0

    This function handles the conversion correctly for each format.

    Args:
        audio: Raw audio array, any numeric dtype.

    Returns:
        Audio as float32 in range [-1.0, 1.0].

    Why divide by max value instead of just casting?
        If you do audio.astype(np.float32) on an int16 array,
        you get floats like -32768.0 and 32767.0.
        The model expects -1.0 to 1.0.
        Dividing by 32768 (2^15) maps the range correctly.
    """
    if audio.dtype == np.float32:
        return audio  # Already correct — no copy needed

    if audio.dtype == np.float64:
        # Simple cast — values are already in float range
        return audio.astype(np.float32)

    if audio.dtype == np.int16:
        # int16 range: -32768 to 32767
        # Divide by 32768 (not 32767) to keep symmetry around 0
        return (audio / 32768.0).astype(np.float32)

    if audio.dtype == np.int32:
        # int32 range: -2147483648 to 2147483647
        return (audio / 2147483648.0).astype(np.float32)

    if audio.dtype == np.uint8:
        # uint8 range: 0 to 255
        # Center at 128, then scale to [-1.0, 1.0]
        return ((audio.astype(np.float32) - 128.0) / 128.0)

    # Unknown dtype — attempt cast with a warning
    logger.warning(
        f"Unknown audio dtype {audio.dtype}, attempting float32 cast. "
        f"Values may be out of expected range."
    )
    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# ENERGY ANALYSIS
# ─────────────────────────────────────────────────────────────────

def compute_rms(audio: np.ndarray) -> float:
    """
    Compute Root Mean Square energy of an audio chunk.

    RMS = sqrt( mean( x[i]² ) )

    This is the single most useful number you can compute about an
    audio chunk. It answers: "how loud is this?"

    Use cases:
        - Pre-VAD sanity check: is the mic even working?
        - Post-normalization verification: did normalization work?
        - Debugging: why is VAD not detecting speech?
        - Silence detection fallback when VAD model is slow

    Args:
        audio: float32 audio array, any length.

    Returns:
        RMS value as float. Range: [0.0, ~1.0].
        Values above 1.0 mean clipping (distorted signal).

    Note: We don't validate dtype here because RMS is used BEFORE
    validation in some debugging scenarios.
    """
    if len(audio) == 0:
        return 0.0

    # The formula: square each sample, take mean, take square root
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    return rms


def is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS_FLOOR) -> bool:
    """
    Quick check: is this chunk effectively silent?

    Useful for skipping VAD processing on obviously silent chunks,
    saving compute. In production, this can be a significant optimization
    because a lot of audio time is silence.

    Args:
        audio: Audio chunk to check.
        threshold: RMS below this → silent.

    Returns:
        True if audio is below the silence threshold.
    """
    return compute_rms(audio) < threshold


def compute_peak(audio: np.ndarray) -> float:
    """
    Compute the peak (maximum absolute) amplitude.

    Different from RMS: RMS measures average energy, peak measures
    the single loudest moment.

    Use case: detecting clipping. If peak >= 0.999, the signal is
    hitting the limits of the ADC and is distorted. You should warn
    the user to move away from the mic or lower input gain.

    Returns:
        Peak absolute amplitude. 1.0 or above = clipping.
    """
    if len(audio) == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


# ─────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────

def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 0.9,
) -> np.ndarray:
    """
    Normalize audio so the peak amplitude equals target_peak.

    Why 0.9 and not 1.0?
        Leaving 10% headroom prevents any rounding errors from causing
        clipping after normalization. It's a conservative safety margin.

    When to use normalization:
        - Before ASR: helps models that are sensitive to input amplitude
        - For consistent VAD thresholds: if audio loudness varies wildly,
          a fixed RMS threshold won't work. Normalize first.

    When NOT to use normalization:
        - When you need to preserve relative loudness (e.g., for
          speaker diarization that uses volume as a cue)
        - When the signal is pure noise (normalizing noise just makes
          louder noise — not useful)

    Args:
        audio: float32 audio to normalize.
        target_peak: Target maximum absolute amplitude.

    Returns:
        Normalized audio array (new array, original unchanged).
    """
    peak = compute_peak(audio)

    # Don't normalize silence — dividing by near-zero amplifies noise
    if peak < SILENCE_RMS_FLOOR:
        logger.debug(
            f"Skipping normalization on near-silent audio | peak={peak:.6f}"
        )
        return audio.copy()

    scale_factor = target_peak / peak
    normalized = (audio * scale_factor).astype(AUDIO_DTYPE)

    logger.debug(
        f"Audio normalized | "
        f"original_peak={peak:.4f} scale={scale_factor:.4f} "
        f"target_peak={target_peak:.4f}"
    )

    return normalized


# ─────────────────────────────────────────────────────────────────
# CHANNEL CONVERSION
# ─────────────────────────────────────────────────────────────────

def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging the two channels.

    Most microphones are mono. But some hardware (USB audio interfaces,
    stereo headsets, some built-in laptop mics) gives you stereo.
    All our models (Whisper, Silero VAD) expect mono.

    Stereo audio arrives as shape (N, 2) — N samples, 2 channels.
    Mono audio is shape (N,) — N samples, 1 channel.

    Why average instead of taking just one channel?
        Averaging reduces noise. If one channel has a hum and the other
        doesn't, averaging partially cancels it. It also ensures we
        don't lose speech that's only in one channel.

    Args:
        audio: Audio array. Shape (N,) for mono or (N, 2) for stereo.

    Returns:
        Mono audio array with shape (N,).
    """
    if audio.ndim == 1:
        return audio  # Already mono

    if audio.ndim == 2:
        if audio.shape[1] == 1:
            return audio[:, 0]  # Single channel in 2D format → flatten
        if audio.shape[1] == 2:
            # Average left and right channels
            mono = np.mean(audio, axis=1).astype(AUDIO_DTYPE)
            logger.debug(
                f"Converted stereo to mono | "
                f"original_shape={audio.shape} mono_shape={mono.shape}"
            )
            return mono

    raise ValueError(
        f"Cannot convert audio with shape {audio.shape} to mono. "
        f"Expected (N,) or (N, 2)."
    )


# ─────────────────────────────────────────────────────────────────
# PADDING
# ─────────────────────────────────────────────────────────────────

def pad_audio(
    audio: np.ndarray,
    pad_ms: int,
    sample_rate: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Add silence padding to the beginning and end of an audio segment.

    Why this matters for ASR:
        Whisper and most ASR models were trained on audio that has
        a brief silence before and after speech. They expect it.
        When you clip speech exactly at the VAD boundary, you're
        feeding the model something slightly different from its
        training distribution, and accuracy drops.

        Adding 200ms of silence on each side is free in terms of
        compute (silence processes instantly) and measurably helps
        transcription accuracy at word boundaries.

    Args:
        audio: The speech audio segment to pad.
        pad_ms: Milliseconds of silence to add on each side.
        sample_rate: Sample rate of the audio (to compute pad length).
        pad_value: Value to fill padding with. 0.0 = digital silence.

    Returns:
        New array: [pad_samples | audio | pad_samples]

    Example:
        audio is 2.0 seconds at 16kHz → 32,000 samples
        pad_ms = 200 → 16000 * 0.2 = 3,200 pad samples each side
        output → 32,000 + 3,200 + 3,200 = 38,400 samples = 2.4 seconds
    """
    # How many samples is pad_ms milliseconds?
    pad_samples = int(sample_rate * pad_ms / 1000)

    silence = np.full(pad_samples, pad_value, dtype=AUDIO_DTYPE)
    padded = np.concatenate([silence, audio, silence])

    logger.debug(
        f"Audio padded | "
        f"original_samples={len(audio)} "
        f"pad_samples={pad_samples} each side | "
        f"total_samples={len(padded)}"
    )

    return padded


# ─────────────────────────────────────────────────────────────────
# DURATION UTILITIES
# ─────────────────────────────────────────────────────────────────

def samples_to_ms(num_samples: int, sample_rate: int) -> float:
    """
    Convert sample count to duration in milliseconds.

    Used everywhere for human-readable logging and threshold comparisons.

    Example:
        samples_to_ms(480, 16000) → 30.0  (one 30ms chunk)
        samples_to_ms(32000, 16000) → 2000.0  (2 seconds of audio)
    """
    return (num_samples / sample_rate) * 1000.0


def ms_to_samples(duration_ms: float, sample_rate: int) -> int:
    """
    Convert duration in milliseconds to sample count.

    Used when converting time-based config values to array sizes.

    Example:
        ms_to_samples(30, 16000) → 480   (samples in a 30ms chunk)
        ms_to_samples(200, 16000) → 3200  (samples in 200ms padding)
    """
    return int(sample_rate * duration_ms / 1000)


def samples_to_seconds(num_samples: int, sample_rate: int) -> float:
    """
    Convert sample count to duration in seconds.

    Used for human-readable logging of utterance durations.

    Example:
        samples_to_seconds(48000, 16000) → 3.0  (3 seconds)
    """
    return num_samples / sample_rate