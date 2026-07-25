# vad/engine.py

"""
Frame-level Voice Activity Detection engine.

Responsibility: Given a single audio chunk, return the probability
that it contains human speech.

This module does ONE thing: run Silero VAD on a chunk and return
a float. It does NOT track state over time. It does NOT decide
when utterances start or end. That is the state machine's job.

Why Silero VAD?
    - Extremely small (~1MB) — loads in milliseconds
    - Runs fast on CPU — processes 32ms chunks in <1ms on modern hardware
    - High accuracy across noise conditions and languages
    - Free and open source (AGPL license)
    - Returns a probability, not just binary yes/no
    - Widely used in production speech systems

Silero VAD requirements:
    - Sample rate: 16000 Hz (or 8000 Hz, but we use 16000)
    - Chunk size:  EXACTLY 512 samples at 16kHz
    - Dtype:       float32
    - Shape:       (512,) — 1D mono
    - Range:       [-1.0, 1.0]

Any deviation from these requirements produces wrong results or crashes.
"""

import numpy as np
import torch
import threading
from typing import Optional

from config.settings import Settings
from utils.logging_config import get_logger, TimingLogger
from utils.audio_utils import validate_audio_chunk, compute_rms, is_silent
import logging

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# SILERO VAD CHUNK SIZE — this is not configurable
# ─────────────────────────────────────────────────────────────────

# Silero VAD REQUIRES exactly 512 samples at 16kHz.
# This constant lives here so any module that needs to know
# the VAD chunk size imports it from here, not from settings.
# The model requirement takes precedence over our preferences.
SILERO_CHUNK_SIZE_16K = 512   # samples at 16kHz = 32ms
SILERO_CHUNK_SIZE_8K  = 256   # samples at 8kHz  = 32ms

# We will use 16kHz throughout this system
SILERO_SAMPLE_RATE = 16000
SILERO_CHUNK_SIZE  = SILERO_CHUNK_SIZE_16K


class VADEngine:
    """
    Wraps Silero VAD for frame-level speech probability estimation.

    Lifecycle:
        1. __init__: store settings, do NOT load model yet
        2. load(): load model from torch.hub — call this explicitly
        3. process_chunk(): call repeatedly for each audio chunk
        4. reset(): reset model hidden state between conversations
        5. unload(): release GPU/CPU memory when done

    Why separate load() from __init__?
        Model loading takes time (downloading, initialization).
        If we load in __init__, the constructor blocks for seconds.
        Explicit load() lets the caller decide WHEN to pay that cost.
        It also makes testing easier — you can construct a VADEngine
        in tests without loading the full model.

        This pattern is called "lazy initialization" or
        "two-phase construction."

    Thread safety:
        process_chunk() is NOT thread-safe.
        The Silero model has internal state (hidden state of the RNN).
        If two threads call process_chunk() simultaneously on the same
        VADEngine instance, they will corrupt each other's hidden state.

        Solution: one VADEngine per thread, or use a lock.
        We use a lock here for safety, documented clearly.

    Usage:
        engine = VADEngine(settings)
        engine.load()

        for chunk in audio_source.stream():
            probability = engine.process_chunk(chunk)
            print(f"Speech probability: {probability:.3f}")

        engine.unload()
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._model_utils = None     # Silero returns utility functions alongside the model
        self._is_loaded = False
        self._lock = threading.Lock()  # Protects model state in multithreaded use

        # Verify our settings are compatible with Silero's requirements
        self._validate_settings()

    def _validate_settings(self) -> None:
        """
        Check settings compatibility with Silero VAD at construction time.

        Failing here at startup is far better than failing silently
        during a live transcription session.

        This is called "fail-fast" design: detect problems as early
        as possible, before they cause mysterious downstream failures.
        """
        cfg = self.settings.audio

        if cfg.sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(
                f"VADEngine requires sample_rate={SILERO_SAMPLE_RATE}Hz. "
                f"Got {cfg.sample_rate}Hz in settings. "
                f"Update AudioConfig.sample_rate to {SILERO_SAMPLE_RATE}."
            )

        if cfg.chunk_size != SILERO_CHUNK_SIZE:
            raise ValueError(
                f"VADEngine requires chunk_size={SILERO_CHUNK_SIZE} samples "
                f"({SILERO_CHUNK_SIZE/SILERO_SAMPLE_RATE*1000:.0f}ms at {SILERO_SAMPLE_RATE}Hz). "
                f"Got chunk_size={cfg.chunk_size} in settings.\n"
                f"Fix: set chunk_duration_ms=32 in AudioConfig "
                f"(32ms * 16000Hz / 1000 = 512 samples)."
            )

        logger.debug(
            f"VADEngine settings validated | "
            f"sample_rate={cfg.sample_rate} "
            f"chunk_size={cfg.chunk_size}"
        )

    def load(self) -> None:
        """
        Load the Silero VAD model.

        First call: downloads model from torch.hub (~1MB), caches locally.
        Subsequent calls: loads from cache instantly.

        Cache location: ~/.cache/torch/hub/

        This method is idempotent — calling it twice is safe.
        The second call returns immediately if already loaded.

        Raises:
            RuntimeError: If model fails to load (no internet, corrupted cache)
        """
        if self._is_loaded:
            logger.debug("VADEngine already loaded — skipping")
            return

        logger.info("Loading Silero VAD model...")

        with TimingLogger(logger, "Silero VAD model load", level=logging.INFO):
            try:
                # torch.hub.load downloads the model from GitHub if not cached.
                # 'snakers4/silero-vad' is the official Silero VAD repository.
                # 'silero_vad' is the model name within that repository.
                #
                # trust_repo=True: required in newer PyTorch versions.
                # It means "I trust this repository's code."
                # Only use with repositories you actually trust.
                model, utils = torch.hub.load(
                    repo_or_dir=r'models/snakers4-silero-vad-980b17e',
                    model='silero_vad',
                    force_reload=True,    # Use cached version if available
                    trust_repo=True,
                    verbose=False,
                    source='local'         # Suppress torch.hub's own logging
                )

                self._model = model
                self._model_utils = utils

                # Put model in evaluation mode.
                # PyTorch models have two modes:
                #   training mode: dropout active, batch norm uses batch statistics
                #   eval mode:     dropout disabled, batch norm uses running statistics
                # We ALWAYS use eval mode for inference. Forgetting this is a
                # common PyTorch bug that causes inconsistent predictions.
                self._model.eval()

                self._is_loaded = True

                logger.info(
                    f"Silero VAD loaded successfully | "
                    f"device=cpu | "
                    f"required_chunk_size={SILERO_CHUNK_SIZE} samples"
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Silero VAD model: {e}\n"
                    f"Check: internet connection (first run), "
                    f"torch installation, disk space."
                ) from e

    def process_chunk(self, audio_chunk: np.ndarray) -> float:
        """
        Run VAD on a single audio chunk.

        This is the core method. It is called for EVERY audio chunk —
        roughly 31 times per second (one call per 32ms chunk).

        It must be FAST. On modern CPU hardware, Silero processes
        one chunk in under 1ms. If this takes longer, you fall behind
        the real-time audio stream.

        Args:
            audio_chunk: float32 numpy array of exactly SILERO_CHUNK_SIZE
                         samples. Range [-1.0, 1.0].

        Returns:
            Speech probability: float in [0.0, 1.0].
            Values closer to 1.0 mean "very likely speech."
            Values closer to 0.0 mean "very likely silence/noise."

        Note on the lock:
            We acquire a lock to protect the model's hidden state.
            If you're only calling this from one thread (which is normal),
            the lock has zero contention and near-zero overhead.
            If you're calling from multiple threads, the lock prevents
            state corruption at the cost of serializing calls.
        """
        if not self._is_loaded:
            raise RuntimeError(
                "VADEngine not loaded. Call engine.load() before process_chunk()."
            )

        # Fast path: skip model on obvious silence
        # RMS check is ~100x faster than running the neural network.
        # If the chunk is clearly silent (all zeros or near-zeros),
        # we know the answer without running the model.
        if is_silent(audio_chunk):
            logger.debug("Chunk is silent — skipping model inference")
            return 0.0

        # Validate the chunk before feeding to model
        # Wrong shapes cause cryptic PyTorch errors deep in the model
        validate_audio_chunk(
            audio_chunk,
            expected_samples=SILERO_CHUNK_SIZE,
            caller="VADEngine.process_chunk"
        )

        with self._lock:
            probability = self._run_inference(audio_chunk)

        logger.debug(
            f"VAD result | "
            f"prob={probability:.4f} "
            f"rms={compute_rms(audio_chunk):.4f} "
            f"speech={'YES' if probability >= self.settings.vad.speech_threshold else 'no'}"
        )

        return probability

    def _run_inference(self, audio_chunk: np.ndarray) -> float:
        """
        Run the actual neural network inference.

        Separated from process_chunk() for clarity:
        process_chunk() handles validation and fast paths.
        _run_inference() handles the PyTorch mechanics.

        The steps:
        1. Convert numpy array → PyTorch tensor
        2. Feed tensor to model
        3. Convert output back to Python float
        """
        # Step 1: numpy → torch tensor
        #
        # Why not just pass the numpy array directly?
        # Silero VAD (like all PyTorch models) operates on Tensors,
        # not numpy arrays. They are different objects in memory.
        #
        # torch.from_numpy() is a zero-copy operation when possible —
        # it creates a tensor that shares memory with the numpy array.
        # No data is copied, just a new view is created.
        #
        # We must NOT call .to(device) here because we're on CPU.
        # If you add GPU support later, add: tensor = tensor.to(self.device)
        tensor = torch.from_numpy(audio_chunk)

        # Step 2: Run inference
        #
        # torch.no_grad() tells PyTorch: "do not compute gradients."
        # Gradients are needed for training (backpropagation).
        # For inference, they are pure waste — memory and compute.
        # ALWAYS use torch.no_grad() during inference.
        # Forgetting this is another very common PyTorch bug.
        #
        # self._model(tensor, SILERO_SAMPLE_RATE):
        #   First arg:  the audio tensor
        #   Second arg: sample rate — the model needs this to know
        #               how many samples = how much time
        #
        # The model internally updates its hidden state (RNN state)
        # each time you call it. This is why reset() matters.
        with torch.no_grad():
            speech_probability = self._model(tensor, SILERO_SAMPLE_RATE)

        # Step 3: tensor → Python float
        #
        # .item() extracts a scalar value from a 0-dimensional tensor.
        # Without .item(), you'd have a Tensor object, not a float.
        # Returning a plain float keeps this module's output clean —
        # callers don't need to know about PyTorch.
        return float(speech_probability.item())

    def reset(self) -> None:
        """
        Reset the model's internal hidden state.

        Call this:
            - Between different conversations/sessions
            - After a long silence (optional but recommended)
            - Before processing a new audio file

        Why does this matter?
            Silero VAD is a recurrent neural network. It maintains
            internal state between calls. This state encodes
            "what happened recently" — is speech building up?
            tapering off? just starting?

            If you process Audio A, then immediately start processing
            Audio B WITHOUT resetting, the model begins Audio B with
            the memory of Audio A. This can cause:
            - Speech at the start of B being missed (model thinks it's
              mid-silence from A's ending)
            - Silence in B being classified as speech (model thinks
              it's continuing from A's speech)

            In practice this matters most for long-running systems
            that process many utterances without stopping.
        """
        if not self._is_loaded:
            return

        with self._lock:
            self._model.reset_states()

        logger.debug("VAD model hidden state reset")

    def unload(self) -> None:
        """
        Release model memory.

        On CPU this frees RAM. On GPU this frees VRAM.
        Call when you are done with the VAD engine entirely
        (e.g., at system shutdown).

        Idempotent: safe to call multiple times.
        """
        if not self._is_loaded:
            return

        self._model = None
        self._model_utils = None
        self._is_loaded = False

        logger.info("VAD model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_chunk_size(self) -> int:
        """
        Return the chunk size this engine requires.

        Other modules call this instead of hardcoding 512.
        If we ever switch VAD models with different requirements,
        this is the only place to update.
        """
        return SILERO_CHUNK_SIZE

    def get_sample_rate(self) -> int:
        """Return the sample rate this engine requires."""
        return SILERO_SAMPLE_RATE