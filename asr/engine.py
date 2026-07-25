# asr/engine.py — corrected for SBPN via NVIDIA NeMo toolkit

"""
ASR engine using SBPN models via NVIDIA NeMo.

Key differences from the Whisper implementation:

    1. Library: nemo_toolkit, not openai-whisper or transformers
    2. Architecture: FastConformer-Transducer (hybrid RNNT/CTC)
    3. Input: NeMo's transcribe() takes FILE PATHS, not numpy arrays.
              We write audio to a temp WAV file, transcribe, then delete.
    4. Output: Raw output includes a language tag: "<english> hello world"
               We strip it with a regex before returning clean text.
    5. Language detection: Built-in — the model detects language
              automatically. We extract it from the output tag.

The TranscriptionResult and PreparedUtterance contracts are unchanged.
The coordinator and all other modules need zero changes.
"""

import re
import time
import wave
import tempfile
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings
from utils.logging_config import get_logger, TimingLogger
from utils.audio_utils import compute_rms, samples_to_seconds
from asr.buffer import PreparedUtterance
import logging

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# TRANSCRIPTION RESULT — IDENTICAL to before
# The coordinator, buffer, queue manager — none of them change.
# ─────────────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    text:                str
    utterance_id:        str
    is_empty:            bool
    detected_language:   Optional[str]  = None
    no_speech_prob:      float          = 0.0
    transcription_ms:    float          = 0.0
    audio_duration_s:    float          = 0.0
    created_at:          float          = field(default_factory=time.time)
    error:               Optional[str]  = None

    @property
    def realtime_factor(self) -> float:
        if self.transcription_ms <= 0:
            return 0.0
        return (self.audio_duration_s * 1000) / self.transcription_ms

    @property
    def clean_text(self) -> str:
        return self.text.strip()

    def __repr__(self) -> str:
        status = "ERROR" if self.error else ("EMPTY" if self.is_empty else "OK")
        return (
            f"TranscriptionResult("
            f"id={self.utterance_id} "
            f"status={status} "
            f"text='{self.clean_text[:50]}' "
            f"rtf={self.realtime_factor:.1f}x)"
        )


# ─────────────────────────────────────────────────────────────────
# LANGUAGE TAG REGEX
# ─────────────────────────────────────────────────────────────────

# SBPN prepends a language tag to every output:
#   "<english> hello world"
#   "<yoruba> ẹ káàárọ̀"
#   "<pidgin> how you dey"
#   "<|en|> you know"
#   "<|yo|> báwo ni"
#
# This regex matches either <english> or the newer short code form <|en|>.
_LANGUAGE_TAG_PATTERN = re.compile(r'<\|?([^>|]+?)\|?>')
_LANGUAGE_CANONICAL = {
    "en": "english",
    "english": "english",
    "pd": "pidgin",
    "yo": "yoruba",
    "yoruba": "yoruba",
    "ha": "hausa",
    "hausa": "hausa",
    "ig": "igbo",
    "igbo": "igbo",
}


def _extract_language_tag(raw_text: str) -> tuple[str, str]:
    """
    Split raw NeMo output into (language_id, clean_text).

    Args:
        raw_text: e.g. "<english> hello world" or "<|en|> hello world"

    Returns:
        ("english", "hello world")
        If no tag found, returns (None, raw_text)

    Examples:
        "<yoruba> ẹ káàárọ̀"    → ("yoruba", "ẹ káàárọ̀")
        "<pidgin> how you dey"  → ("pidgin", "how you dey")
        "<|en|> you know"       → ("english", "you know")
        "hello world"           → (None, "hello world")
    """
    stripped = raw_text.strip()
    match = _LANGUAGE_TAG_PATTERN.match(stripped)
    if match:
        language_tag = match.group(1).strip().lower()
        language = _LANGUAGE_CANONICAL.get(language_tag, language_tag)
        clean = _LANGUAGE_TAG_PATTERN.sub('', stripped).strip()
        return language, clean
    return None, stripped


# ─────────────────────────────────────────────────────────────────
# TEMP FILE HELPER
# ─────────────────────────────────────────────────────────────────

def _write_temp_wav(
    audio: np.ndarray,
    sample_rate: int,
) -> str:
    """
    Write a numpy audio array to a temporary WAV file.

    NeMo's transcribe() method accepts file paths, not numpy arrays.
    This function bridges that gap: write audio to a temp file,
    return the path, let NeMo read it, then caller deletes it.

    Why a temp file and not an in-memory buffer?
        NeMo's ASR pipeline internally uses soundfile or librosa
        to read audio from disk. It does not expose a direct
        numpy array interface at the transcribe() level.
        Writing to disk is the documented and reliable approach.

    Args:
        audio: float32 numpy array, range [-1.0, 1.0]
        sample_rate: sample rate of the audio (must be 16000 for SBPN)

    Returns:
        Path to the temporary WAV file.
        CALLER IS RESPONSIBLE for deleting this file after use.

    Why delete=False in NamedTemporaryFile?
        On Windows, a NamedTemporaryFile cannot be opened by another
        process while it is still open in the creating process.
        NeMo needs to open the file. delete=False lets us close our
        handle before NeMo opens it, then we manually delete afterward.
    """
    # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
    # WAV files conventionally store int16 PCM audio
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create a named temp file that persists until we explicitly delete it
    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )
    tmp_path = tmp.name
    tmp.close()  # Close our handle so NeMo can open it on Windows

    with wave.open(tmp_path, "wb") as wav_file:
        wav_file.setnchannels(1)              # mono
        wav_file.setsampwidth(2)              # 2 bytes = int16
        wav_file.setframerate(sample_rate)    # 16000 Hz
        wav_file.writeframes(audio_int16.tobytes())

    return tmp_path


# ─────────────────────────────────────────────────────────────────
# ASR ENGINE
# ─────────────────────────────────────────────────────────────────

class ASREngine:
    """
    SBPN ASR engine using NVIDIA NeMo toolkit.

    Model architecture: FastConformer-Transducer (hybrid RNNT/CTC BPE)
    Supported languages: Yorùbá, Hausa, Igbo, Nigerian Pidgin, Nigerian English
    Model sizes:
        ogunlao/SBPN_multilingual_base   — 120M parameters
        ogunlao/SBPN_multilingual_large  — 600M parameters

    License: CC BY-NC-SA 4.0 — research/non-commercial use only.

    Lifecycle:
        engine = ASREngine(settings)
        engine.load()                          # downloads ~500MB or ~2.3GB
        result = engine.transcribe(utterance)  # returns TranscriptionResult
        engine.unload()
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model    = None
        self._is_loaded = False

        # Diagnostics
        self._total_transcribed = 0
        self._total_empty       = 0
        self._total_errors      = 0
        self._total_latency_ms  = 0.0

        logger.info(
            f"ASREngine (SBPN/NeMo) created | "
            f"model={settings.asr.model_name} | "
            f"device={settings.asr.device}"
        )

    def load(self) -> None:
        """
        Load the SBPN model via NeMo's from_pretrained().

        from_pretrained() downloads from HuggingFace Hub and caches at:
            ~/.cache/huggingface/hub/

        Download sizes:
            SBPN_multilingual_base:  ~500MB
            SBPN_multilingual_large: ~2.3GB

        NeMo model class used:
            EncDecHybridRNNTCTCBPEModel
            This is the correct class for FastConformer hybrid RNNT/CTC models.
            Using the wrong class (e.g., ASRModel) still works via duck typing
            but EncDecHybridRNNTCTCBPEModel is explicit and correct.

        This method is idempotent — safe to call multiple times.
        """
        if self._is_loaded:
            logger.debug("ASREngine already loaded — skipping")
            return

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NVIDIA NeMo toolkit not installed.\n"
                "Run: pip install nemo_toolkit['all']\n"
                "Note: this is a large install (~2GB) and takes several minutes.\n"
                "Requires PyTorch to be installed first."
            )

        cfg = self.settings.asr

        logger.info(
            f"Loading SBPN model '{cfg.model_name}' via NeMo..."
        )

        with TimingLogger(logger, f"SBPN '{cfg.model_name}' load", logging.INFO):
            try:
                # Use the specific model class from the model card
                # EncDecHybridRNNTCTCBPEModel = FastConformer with
                # hybrid RNNT + CTC decoding and BPE tokenizer

                # Step 1: Download config only

                sbpn_config = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                    model_name="ogunlao/SBPN_multilingual_base",
                    return_config=True
                    )
                
                # Step 2: Swap out the graph_rnnt loss for one that's already installed
                from omegaconf import OmegaConf
                OmegaConf.set_struct(sbpn_config, False)
                sbpn_config.loss.loss_name = "warprnnt_numba"

                self._model = (
                    nemo_asr.models.EncDecHybridRNNTCTCBPEModel
                    .from_pretrained(
                        model_name=cfg.model_name,
                        override_config_path=sbpn_config
                    )
                )

                # Move to target device
                if cfg.device == "cuda":
                    self._model = self._model.cuda()
                else:
                    self._model = self._model.cpu()

                # Eval mode — always for inference
                self._model.eval()

                self._is_loaded = True

                logger.info(
                    f"SBPN model loaded | "
                    f"model={cfg.model_name} | "
                    f"device={cfg.device}"
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load SBPN model '{cfg.model_name}': {e}\n"
                    f"Check: internet connection, disk space, "
                    f"nemo_toolkit installation."
                ) from e

    def transcribe(
        self,
        utterance: PreparedUtterance,
    ) -> TranscriptionResult:
        """
        Transcribe a PreparedUtterance using SBPN/NeMo.

        Same signature as the Whisper version — coordinator is unchanged.

        Internally:
            1. Write utterance.audio to a temporary WAV file
            2. Pass the file path to asr_model.transcribe()
            3. Parse language tag from raw output
            4. Clean text and return TranscriptionResult
            5. Delete the temporary WAV file

        Step 5 runs in a finally block — the temp file is always
        deleted even if transcription fails.
        """
        if not self._is_loaded:
            return TranscriptionResult(
                text="",
                utterance_id=utterance.utterance_id,
                is_empty=True,
                error="ASREngine not loaded. Call engine.load() first.",
            )

        start_time = time.perf_counter()

        try:
            result = self._run_transcription(utterance)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._total_errors += 1
            logger.error(
                f"Transcription failed | "
                f"id={utterance.utterance_id} | "
                f"error={type(e).__name__}: {e}",
                exc_info=True,
            )
            return TranscriptionResult(
                text="",
                utterance_id=utterance.utterance_id,
                is_empty=True,
                transcription_ms=elapsed_ms,
                error=str(e),
            )

        return result

    def _run_transcription(
        self,
        utterance: PreparedUtterance,
    ) -> TranscriptionResult:
        """
        Core NeMo transcription logic.

        NeMo API:
            output = asr_model.transcribe(["path/to/file.wav"])
            raw_text = output[0].text
            # raw_text = "<english> hello world"

        We must:
            1. Write audio to temp WAV file (NeMo needs a file path)
            2. Call transcribe() with the file path in a list
            3. Extract language tag from raw output
            4. Optionally strip the language tag
            5. Delete temp file
        """
        cfg = self.settings.asr
        start_time = time.perf_counter()
        tmp_path = None

        try:
            # ── Step 1: Write audio to temp WAV file ─────────────
            # NeMo's transcribe() takes a list of file paths.
            # It does not accept numpy arrays directly.
            tmp_path = _write_temp_wav(
                utterance.audio,
                sample_rate=self.settings.audio.sample_rate,
            )

            logger.debug(
                f"Temp WAV written | "
                f"path={tmp_path} | "
                f"duration={utterance.raw_duration_s:.2f}s"
            )

            # ── Step 2: Run NeMo transcription ───────────────────
            # transcribe() takes a LIST of file paths.
            # Returns a list of Hypothesis objects (one per file).
            # We pass one file so we get one result at index [0].
            output = self._model.transcribe([tmp_path])

            # ── Step 3: Extract raw text from Hypothesis object ──
            # NeMo returns Hypothesis objects with a .text attribute.
            # Raw output includes language tag: "<english> hello world"
            raw_text = output[0].text

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # ── Step 4: Parse language tag ───────────────────────
            detected_language, text_without_tag = _extract_language_tag(raw_text)

            logger.debug(
                f"Raw NeMo output: '{raw_text}' | "
                f"detected_language={detected_language} | "
                f"clean_text='{text_without_tag}'"
            )

            # ── Step 5: Decide what text to return ───────────────
            # The ASR language tag is metadata only. Remove it before
            # sending text to the client and to the LLM.
            final_text = text_without_tag
            clean = final_text.strip()
            is_empty = (
                len(clean) == 0
                or clean in {".", ",", "!", "?", "..."}
            )

            self._total_transcribed += 1
            self._total_latency_ms += elapsed_ms
            if is_empty:
                self._total_empty += 1

            log_level = logging.INFO if not is_empty else logging.DEBUG
            logger.log(
                log_level,
                f"Transcription {'complete' if not is_empty else 'empty'} | "
                f"id={utterance.utterance_id} | "
                f"text='{clean[:80]}' | "
                f"lang={detected_language} | "
                f"latency={elapsed_ms:.0f}ms | "
                f"rtf={utterance.raw_duration_s * 1000 / elapsed_ms:.1f}x"
            )

            return TranscriptionResult(
                text=final_text,
                utterance_id=utterance.utterance_id,
                is_empty=is_empty,
                detected_language=detected_language,
                no_speech_prob=0.0,   # NeMo does not expose this
                transcription_ms=elapsed_ms,
                audio_duration_s=utterance.padded_duration_s,
            )

        finally:
            # ── Always delete the temp file ───────────────────────
            # This runs even if transcription raised an exception.
            # Temp files accumulate fast at 31 utterances/minute —
            # always clean up immediately.
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    logger.debug(f"Temp WAV deleted | path={tmp_path}")
                except OSError as e:
                    logger.warning(
                        f"Failed to delete temp WAV '{tmp_path}': {e}"
                    )

    def unload(self) -> None:
        """Release model memory. Identical pattern to before."""
        if not self._is_loaded:
            return

        self._model = None
        self._is_loaded = False

        import gc
        gc.collect()

        logger.info("SBPN model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_stats(self) -> dict:
        avg_latency = (
            self._total_latency_ms / self._total_transcribed
            if self._total_transcribed > 0 else 0.0
        )
        return {
            "model_name":        self.settings.asr.model_name,
            "total_transcribed": self._total_transcribed,
            "total_empty":       self._total_empty,
            "total_errors":      self._total_errors,
            "avg_latency_ms":    avg_latency,
            "empty_rate": (
                self._total_empty / self._total_transcribed
                if self._total_transcribed > 0 else 0.0
            ),
            "is_loaded": self._is_loaded,
        }