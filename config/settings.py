# config/settings.py

"""
Central configuration for the realtime ASR system.

Engineering principle: All tunable values live here.
No module should contain hardcoded constants that affect behavior.

Why a dataclass?
- Type hints catch mistakes (e.g., passing a string where int expected)
- IDE autocomplete works
- Easy to print the whole config for debugging
- Easy to override in tests
"""

from dataclasses import dataclass, field
import os


@dataclass
class AudioConfig:
    """
    Configuration for audio capture and processing.

    sample_rate: How many audio samples per second.
        16000 Hz is the standard for most speech models (Whisper, Silero VAD).
        The microphone might output 44100 or 48000 — we'll resample.

    chunk_duration_ms: How long each audio chunk is in milliseconds.
        30ms is the standard for VAD (Silero VAD specifically requires this).
        Too short → more CPU overhead, more queue messages.
        Too long → more latency before VAD can react.

    chunk_size: Computed from sample_rate and chunk_duration_ms.
        This is how many samples are in each chunk.
        At 16kHz and 30ms: 16000 * 0.030 = 480 samples.

    channels: Mono (1) or stereo (2).
        ASR models always want mono. We convert early.

    input_device_index: Which microphone to use. None = system default.
    """
    sample_rate: int = 16000
    chunk_duration_ms: int = 32
    channels: int = 1
    input_device_index: int | None = None

    @property
    def chunk_size(self) -> int:
        """Number of samples per chunk. Computed, never set manually."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


@dataclass
class VADConfig:
    """
    Configuration for Voice Activity Detection.

    speech_threshold: VAD probability above this → speech detected.
        Range: 0.0 to 1.0.
        Lower = more sensitive (catches quiet speech, also catches more noise).
        Higher = less sensitive (misses quiet speech, more robust to noise).
        Silero VAD default: 0.5. Good starting point.

    silence_threshold: VAD probability below this → silence detected.
        Usually lower than speech_threshold to create hysteresis.
        If speech_threshold = 0.5 and silence_threshold = 0.35:
            → We require MORE confidence to exit speech than to enter.
            → This prevents choppy on/off switching.

    speech_pad_ms: Milliseconds of audio to add BEFORE and AFTER speech.
        Why? VAD is slightly late. By the time it detects speech,
        you've already missed the first few milliseconds of the phoneme.
        Padding ensures the first consonant isn't clipped.
        Also adds trailing audio so words don't get cut off.

    min_speech_duration_ms: Ignore speech segments shorter than this.
        Prevents "eh" sounds, coughs, clicks from triggering ASR.

    min_silence_duration_ms: How long silence must persist before we
        consider an utterance "done." 
        Too short → cuts off slow speakers mid-sentence.
        Too long → adds latency before ASR fires.
        300-800ms is typical for conversational speech.
    """
    speech_threshold: float = 0.5
    silence_threshold: float = 0.35
    speech_pad_ms: int = 200
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500


@dataclass
class ASRConfig:
    """
    Configuration for the ASR (Speech-to-Text) engine.

    model_name: Whisper model size.
        Options: tiny, base, small, medium, large-v3
        Bigger → more accurate, slower, more memory.
        For learning: 'base' is fast enough on CPU and good quality.

    language: Force a language or let Whisper auto-detect.
        None = auto-detect (slower, uses extra compute).
        'en' = English only (faster, more accurate for English).

    device: 'cpu' or 'cuda'.
        Start with 'cpu'. Add GPU later.

    max_utterance_duration_s: Safety valve — if a speech segment runs
        longer than this, we force-flush the buffer to ASR.
        Prevents runaway memory if VAD fails to detect end of speech.
    """
    model_name: str   = "ogunlao/SBPN_multilingual_base"
    language: str | None = "pcm"
    device: str = "cpu"
    max_utterance_duration_s: float = 30.0
    normalize_utterances:     bool  = True
    chunk_length_s:           float = 30.0

@dataclass
class PipelineConfig:
    """
    Configuration for the pipeline orchestration layer.

    queue_maxsize: Maximum items in each queue before producer blocks.
        0 = unlimited (dangerous in production, can run out of memory).
        Small number = backpressure (producer slows down when consumer is slow).
        For learning: 50 is reasonable.

    worker_threads: How many ASR worker threads.
        ASR is the bottleneck. More threads = more parallelism.
        But each thread needs its own model loaded (memory cost).
        Start with 1. Add more later.
    """
    queue_maxsize: int = 50
    worker_threads: int = 1


@dataclass
class Settings:
    """
    Root settings object. Import this everywhere.

    Usage:
        from config.settings import Settings
        settings = Settings()
        print(settings.audio.sample_rate)  # 16000
        print(settings.vad.speech_threshold)  # 0.5

    To override for testing:
        settings = Settings()
        settings.vad.speech_threshold = 0.8  # stricter VAD for this test
    """
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self):
        """
        After initialization, apply any environment variable overrides.

        Real production systems do this extensively.
        We implement just a few examples here.
        """
        # Allow overriding model via environment variable
        # e.g., ASR_MODEL=large-v3 python main.py
        if env_model := os.getenv("ASR_MODEL"):
            self.asr.model_name = env_model

        if env_device := os.getenv("ASR_DEVICE"):
            self.asr.device = env_device

        if env_language := os.getenv("ASR_LANGUAGE"):
            self.asr.language = env_language if env_language != "auto" else None