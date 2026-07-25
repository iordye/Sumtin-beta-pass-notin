# tests/test_asr_engine.py

"""
Testing the ASR engine.

The ASR engine wraps Whisper, so our testing strategy mirrors
what we did for the VAD engine wrapper:

    Unit tests (mocked Whisper):
        Test OUR logic — error handling, metadata extraction,
        empty detection, latency measurement, diagnostic counters.
        Fast. No model download required.

    Integration tests (real Whisper):
        Test actual transcription. Marked @pytest.mark.slow.
        Require model download on first run.

What failures mean:
    - Error handling broken → one bad utterance crashes the worker thread
      → pipeline silently stops transcribing → no one notices until
      someone checks logs
    - Empty detection broken → punctuation-only results displayed to user
    - Latency not measured → can't detect performance regressions
    - Counter broken → ops team sees wrong metrics, misses real problems
    - fp16 on CPU → silent numerical errors or crash on some hardware
"""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from config.settings import Settings
from asr.engine import ASREngine, TranscriptionResult
from asr.buffer import PreparedUtterance


# ─────────────────────────────────────────────────────────────────
# FIXTURES AND HELPERS
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    s = Settings()
    s.asr.model_name = "base"
    s.asr.device = "cpu"
    s.asr.language = "en"
    return s

@pytest.fixture
def engine(settings):
    """Unloaded engine — for testing construction and validation."""
    return ASREngine(settings)

def make_utterance(
    text_hint: str = "test",
    duration_s: float = 2.0,
    sample_rate: int = 16000,
) -> PreparedUtterance:
    """
    Create a PreparedUtterance with synthetic audio.
    text_hint is just for readability in test names.
    """
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    return PreparedUtterance(
        audio             = audio,
        raw_duration_s    = duration_s,
        padded_duration_s = duration_s + 0.4,
        rms_energy        = 0.21,
        peak_amplitude    = 0.3,
        chunk_count       = int(duration_s * 1000 / 32),
    )

def make_mock_engine(
    settings: Settings,
    transcribe_return: dict = None,
) -> ASREngine:
    """
    ASREngine with mocked Whisper model.

    transcribe_return: what model.transcribe() should return.
    Defaults to a simple English transcription.
    """
    if transcribe_return is None:
        transcribe_return = {
            "text": " Hello world",
            "language": "en",
            "segments": [
                {
                    "text": " Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "no_speech_prob": 0.02,
                    "avg_logprob": -0.21,
                    "compression_ratio": 1.1,
                }
            ],
        }

    eng = ASREngine(settings)
    eng._is_loaded = True
    eng._model = MagicMock()
    eng._model.transcribe.return_value = transcribe_return

    return eng


# ─────────────────────────────────────────────────────────────────
# CONSTRUCTION TESTS
# ─────────────────────────────────────────────────────────────────

class TestASREngineConstruction:

    def test_starts_unloaded(self, engine):
        assert engine.is_loaded is False

    def test_stats_start_at_zero(self, engine):
        stats = engine.get_stats()
        assert stats["total_transcribed"] == 0
        assert stats["total_empty"] == 0
        assert stats["total_errors"] == 0

    def test_transcribe_without_load_returns_error_result(self, engine):
        """
        Calling transcribe() before load() should return an error result,
        NOT raise an exception. The pipeline must not crash.
        """
        utterance = make_utterance()
        result = engine.transcribe(utterance)

        assert isinstance(result, TranscriptionResult)
        assert result.error is not None
        assert result.is_empty is True


# ─────────────────────────────────────────────────────────────────
# TRANSCRIPTION RESULT TESTS — using mocked model
# ─────────────────────────────────────────────────────────────────

class TestTranscriptionResult:

    def test_returns_transcription_result_type(self, settings):
        engine = make_mock_engine(settings)
        result = engine.transcribe(make_utterance())
        assert isinstance(result, TranscriptionResult)

    def test_text_extracted_from_whisper_output(self, settings):
        """Text from Whisper dict should appear in result."""
        engine = make_mock_engine(settings, transcribe_return={
            "text": " Hello world",
            "language": "en",
            "segments": [{"no_speech_prob": 0.02}],
        })
        result = engine.transcribe(make_utterance())
        assert "Hello world" in result.clean_text

    def test_clean_text_strips_whitespace(self, settings):
        """Whisper often adds leading space. clean_text must remove it."""
        engine = make_mock_engine(settings, transcribe_return={
            "text": " Hello world ",
            "language": "en",
            "segments": [],
        })
        result = engine.transcribe(make_utterance())
        assert result.clean_text == "Hello world"
        assert not result.clean_text.startswith(" ")

    def test_language_extracted(self, settings):
        engine = make_mock_engine(settings, transcribe_return={
            "text": " Hola",
            "language": "es",
            "segments": [],
        })
        result = engine.transcribe(make_utterance())
        assert result.detected_language == "es"

    def test_language_tag_stripped_from_transcript(self, settings):
        engine = make_mock_engine(settings, transcribe_return={
            "text": "<|en|> you know",
            "language": "en",
            "segments": [],
        })
        result = engine.transcribe(make_utterance())
        assert result.clean_text == "you know"
        assert result.detected_language == "english"

    def test_pidgin_language_tag_is_normalized(self, settings):
        engine = make_mock_engine(settings, transcribe_return={
            "text": "<pidgin> how you dey",
            "language": "pidgin",
            "segments": [],
        })
        result = engine.transcribe(make_utterance())
        assert result.clean_text == "how you dey"
        assert result.detected_language == "pidgin"

    def test_no_speech_prob_extracted_from_segment(self, settings):
        engine = make_mock_engine(settings, transcribe_return={
            "text": " Hello",
            "language": "en",
            "segments": [{"no_speech_prob": 0.73}],
        })
        result = engine.transcribe(make_utterance())
        assert result.no_speech_prob == pytest.approx(0.73)

    def test_utterance_id_preserved(self, settings):
        """
        The utterance_id from PreparedUtterance must be carried through
        to TranscriptionResult for end-to-end tracing.
        """
        engine = make_mock_engine(settings)
        utterance = make_utterance()
        result = engine.transcribe(utterance)
        assert result.utterance_id == utterance.utterance_id

    def test_transcription_time_is_measured(self, settings):
        """transcription_ms must be a positive number."""
        engine = make_mock_engine(settings)
        result = engine.transcribe(make_utterance())
        assert result.transcription_ms > 0

    def test_audio_duration_in_result(self, settings):
        """audio_duration_s should reflect the padded utterance duration."""
        engine = make_mock_engine(settings)
        utterance = make_utterance(duration_s=2.0)
        result = engine.transcribe(utterance)
        assert result.audio_duration_s == pytest.approx(
            utterance.padded_duration_s, abs=0.01
        )


# ─────────────────────────────────────────────────────────────────
# EMPTY DETECTION TESTS
# ─────────────────────────────────────────────────────────────────

class TestEmptyDetection:

    def _make_empty_result_engine(self, settings, text: str) -> ASREngine:
        return make_mock_engine(settings, transcribe_return={
            "text": text,
            "language": "en",
            "segments": [],
        })

    def test_empty_string_is_empty(self, settings):
        engine = self._make_empty_result_engine(settings, "")
        result = engine.transcribe(make_utterance())
        assert result.is_empty is True

    def test_whitespace_only_is_empty(self, settings):
        engine = self._make_empty_result_engine(settings, "   ")
        result = engine.transcribe(make_utterance())
        assert result.is_empty is True

    def test_punctuation_only_is_empty(self, settings):
        """
        Whisper sometimes returns just a period for near-silence.
        This is a hallucination and should be treated as empty.
        """
        for punct in [".", ",", "!", "?", "...", " ."]:
            eng = self._make_empty_result_engine(settings, punct)
            result = eng.transcribe(make_utterance())
            assert result.is_empty is True, (
                f"'{punct}' should be treated as empty transcription"
            )

    def test_real_text_is_not_empty(self, settings):
        engine = self._make_empty_result_engine(settings, " Hello world")
        result = engine.transcribe(make_utterance())
        assert result.is_empty is False


# ─────────────────────────────────────────────────────────────────
# ERROR HANDLING TESTS
# ─────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_whisper_exception_returns_error_result(self, settings):
        """
        If Whisper throws any exception, transcribe() must return
        an error result — NOT propagate the exception.

        This is critical: an unhandled exception in a worker thread
        kills the thread silently.
        """
        engine = make_mock_engine(settings)
        engine._model.transcribe.side_effect = RuntimeError("CUDA out of memory")

        utterance = make_utterance()
        result = engine.transcribe(utterance)

        # Must return a result, not raise
        assert isinstance(result, TranscriptionResult)
        assert result.error is not None
        assert "CUDA out of memory" in result.error
        assert result.is_empty is True

    def test_error_increments_counter(self, settings):
        engine = make_mock_engine(settings)
        engine._model.transcribe.side_effect = ValueError("bad audio")

        engine.transcribe(make_utterance())

        stats = engine.get_stats()
        assert stats["total_errors"] == 1

    def test_error_does_not_prevent_subsequent_transcriptions(self, settings):
        """
        After one failure, the engine must continue working normally.
        The pipeline should degrade gracefully, not stop entirely.
        """
        engine = make_mock_engine(settings)

        # First call fails
        engine._model.transcribe.side_effect = RuntimeError("transient error")
        result1 = engine.transcribe(make_utterance())
        assert result1.error is not None

        # Second call succeeds
        engine._model.transcribe.side_effect = None
        engine._model.transcribe.return_value = {
            "text": " Hello",
            "language": "en",
            "segments": [],
        }
        result2 = engine.transcribe(make_utterance())
        assert result2.error is None
        assert not result2.is_empty


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC COUNTER TESTS
# ─────────────────────────────────────────────────────────────────

class TestDiagnostics:

    def test_transcribed_counter_increments(self, settings):
        engine = make_mock_engine(settings)
        engine.transcribe(make_utterance())
        engine.transcribe(make_utterance())
        assert engine.get_stats()["total_transcribed"] == 2

    def test_empty_counter_increments(self, settings):
        engine = make_mock_engine(settings, transcribe_return={
            "text": "",
            "language": "en",
            "segments": [],
        })
        engine.transcribe(make_utterance())
        assert engine.get_stats()["total_empty"] == 1

    def test_realtime_factor_property(self, settings):
        """RTF = audio_duration_s * 1000 / transcription_ms."""
        result = TranscriptionResult(
            text="Hello",
            utterance_id="test123",
            is_empty=False,
            audio_duration_s=3.0,
            transcription_ms=500.0,
        )
        expected_rtf = (3.0 * 1000) / 500.0  # = 6.0
        assert result.realtime_factor == pytest.approx(expected_rtf)

    def test_unload_sets_loaded_false(self, settings):
        engine = make_mock_engine(settings)
        assert engine.is_loaded is True
        engine.unload()
        assert engine.is_loaded is False


# ─────────────────────────────────────────────────────────────────
# INTEGRATION TESTS — real Whisper model
# ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestASREngineIntegration:
    """
    Real Whisper transcription tests.
    Require model download on first run (~145MB for 'base').
    Run with: pytest -m slow
    """

    @pytest.fixture
    def loaded_engine(self, settings):
        settings.asr.model_name = "base"
        settings.asr.language = "en"
        engine = ASREngine(settings)
        engine.load()
        yield engine
        engine.unload()

    def test_load_succeeds(self, settings):
        engine = ASREngine(settings)
        engine.load()
        assert engine.is_loaded is True
        engine.unload()

    def test_silence_produces_empty_result(self, loaded_engine):
        """
        Digital silence (zeros) should produce empty or near-empty output.
        This validates our three-layer hallucination defense.
        """
        silence = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds
        utterance = PreparedUtterance(
            audio             = silence,
            raw_duration_s    = 2.0,
            padded_duration_s = 2.4,
            rms_energy        = 0.0,
            peak_amplitude    = 0.0,
            chunk_count       = 62,
        )
        result = loaded_engine.transcribe(utterance)
        # Whisper should either return empty or have high no_speech_prob
        assert result.is_empty or result.no_speech_prob > 0.5, (
            f"Silence produced non-empty transcription: '{result.clean_text}' "
            f"no_speech_prob={result.no_speech_prob:.3f}"
        )

    def test_transcription_returns_result_type(self, loaded_engine):
        """Real transcription should return TranscriptionResult."""
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        utterance = PreparedUtterance(
            audio=audio,
            raw_duration_s=2.0,
            padded_duration_s=2.4,
            rms_energy=0.21,
            peak_amplitude=0.3,
            chunk_count=62,
        )
        result = loaded_engine.transcribe(utterance)
        assert isinstance(result, TranscriptionResult)
        assert result.transcription_ms > 0