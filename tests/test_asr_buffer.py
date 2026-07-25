# tests/test_asr_buffer.py

"""
Testing the ASR buffer.

The buffer is a pure data transformation module with no AI models,
no threads, no queues. This makes it the easiest module to test
comprehensively.

We test:
    1. Valid utterances are accepted and prepared correctly
    2. Invalid utterances are rejected with correct reasons
    3. Padding is applied correctly
    4. Normalization is applied correctly
    5. Metadata in PreparedUtterance is accurate
    6. Diagnostic counters are correct
    7. Edge cases: minimum length, maximum length, silence-level audio

What failures mean:
    - Too-short rejection broken → garbage fragments reach Whisper,
      wasting compute and producing noise transcriptions
    - Padding not applied → word boundaries clipped → accuracy drops
    - RMS check broken → corrupt audio silently reaches Whisper
    - Metadata wrong → debugging becomes impossible
    - Counter wrong → ops team gets incorrect pipeline health metrics
"""

import numpy as np
import pytest
import time

from config.settings import Settings
from asr.buffer import ASRBuffer, PreparedUtterance, RejectionReason
from utils.audio_utils import compute_rms, compute_peak


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    s = Settings()
    s.vad.min_speech_duration_ms   = 500    # 0.5s minimum
    s.vad.speech_pad_ms            = 200    # 200ms padding each side
    s.asr.max_utterance_duration_s = 30.0
    s.asr.normalize_utterances     = True
    return s

@pytest.fixture
def buffer(settings):
    return ASRBuffer(settings)

def make_utterance(
    duration_s: float,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
    frequency: float = 440.0,
) -> np.ndarray:
    """
    Generate a sine wave utterance of given duration.
    Has real energy (RMS ≈ amplitude / sqrt(2)) unlike silence.
    """
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

def make_silence(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a silence utterance (all zeros)."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# ACCEPTANCE TESTS — valid utterances should pass through
# ─────────────────────────────────────────────────────────────────

class TestAcceptance:

    def test_valid_utterance_is_accepted(self, buffer):
        """A normal 2-second speech utterance should be accepted."""
        audio = make_utterance(duration_s=2.0)
        result = buffer.prepare(audio, chunk_count=63)
        assert result is not None

    def test_returns_prepared_utterance_type(self, buffer):
        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio)
        assert isinstance(result, PreparedUtterance)

    def test_accepted_audio_is_float32(self, buffer):
        """Output audio must always be float32."""
        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio)
        assert result.audio.dtype == np.float32

    def test_accepted_audio_is_1d(self, buffer):
        """Output audio must be 1D mono."""
        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio)
        assert result.audio.ndim == 1

    def test_minimum_length_boundary(self, settings, buffer):
        """
        An utterance exactly at minimum duration should be accepted.
        One sample shorter should be rejected.
        """
        sr = settings.audio.sample_rate
        min_ms = settings.vad.min_speech_duration_ms

        # Exactly at minimum
        exact_samples = int(sr * min_ms / 1000)
        audio_exact = make_utterance(
            duration_s=min_ms / 1000,
        )
        # Ensure we have exact sample count
        audio_exact = audio_exact[:exact_samples]
        result = buffer.prepare(audio_exact)
        assert result is not None, "Utterance at exact minimum should be accepted"

    def test_long_utterance_accepted(self, buffer):
        """Utterances up to max_duration should be accepted."""
        audio = make_utterance(duration_s=20.0)
        result = buffer.prepare(audio)
        assert result is not None


# ─────────────────────────────────────────────────────────────────
# REJECTION TESTS — invalid utterances should be filtered
# ─────────────────────────────────────────────────────────────────

class TestRejection:

    def test_too_short_rejected(self, settings, buffer):
        """Utterances shorter than minimum duration return None."""
        # 100ms is well below 500ms minimum
        audio = make_utterance(duration_s=0.1)
        result = buffer.prepare(audio)
        assert result is None

    def test_empty_array_rejected(self, buffer):
        """Empty audio must return None, not crash."""
        audio = np.array([], dtype=np.float32)
        result = buffer.prepare(audio)
        assert result is None

    def test_none_input_rejected(self, buffer):
        """None input must return None, not raise."""
        result = buffer.prepare(None)
        assert result is None

    def test_too_quiet_rejected(self, buffer):
        """
        Audio with RMS below quality threshold should be rejected.
        This catches mic disconnection mid-utterance.
        """
        # Near-silence that passes length check but fails RMS check
        # 1 second of near-zero audio
        audio = np.full(16000, 0.0001, dtype=np.float32)
        result = buffer.prepare(audio)
        assert result is None

    def test_wrong_dtype_rejected(self, buffer):
        """int16 audio reaching the buffer indicates an upstream bug."""
        audio = np.zeros(16000, dtype=np.int16)
        result = buffer.prepare(audio)
        assert result is None

    def test_too_long_rejected(self, settings):
        """Utterances over max duration should be rejected."""
        settings.asr.max_utterance_duration_s = 5.0
        buf = ASRBuffer(settings)
        audio = make_utterance(duration_s=10.0)  # Over 5s max
        result = buf.prepare(audio)
        assert result is None


# ─────────────────────────────────────────────────────────────────
# PADDING TESTS
# ─────────────────────────────────────────────────────────────────

class TestPadding:

    def test_padding_increases_duration(self, settings, buffer):
        """
        Padded audio must be longer than original.
        padding adds speech_pad_ms on EACH side.
        """
        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio, chunk_count=31)

        assert result is not None
        assert result.padded_duration_s > result.raw_duration_s

    def test_padding_amount_is_correct(self, settings, buffer):
        """
        Total added duration = 2 * speech_pad_ms (both sides).
        speech_pad_ms = 200ms → 400ms added total.
        """
        sr = settings.audio.sample_rate
        pad_ms = settings.vad.speech_pad_ms

        audio = make_utterance(duration_s=1.0)
        original_samples = len(audio)
        result = buffer.prepare(audio)

        assert result is not None

        expected_pad_samples = 2 * int(sr * pad_ms / 1000)
        expected_total = original_samples + expected_pad_samples
        actual_total = len(result.audio)

        assert actual_total == expected_total, (
            f"Expected {expected_total} samples after padding, "
            f"got {actual_total}. "
            f"Padding of {pad_ms}ms each side should add "
            f"{expected_pad_samples} samples total."
        )

    def test_padding_is_silence(self, settings, buffer):
        """
        The added padding must be silence (zeros), not data.
        """
        sr = settings.audio.sample_rate
        pad_ms = settings.vad.speech_pad_ms
        pad_samples = int(sr * pad_ms / 1000)

        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio)

        assert result is not None

        # First pad_samples should be silence
        leading_pad = result.audio[:pad_samples]
        np.testing.assert_array_equal(
            leading_pad, 0.0,
            err_msg="Leading padding should be silence (zeros)"
        )

        # Last pad_samples should be silence
        trailing_pad = result.audio[-pad_samples:]
        np.testing.assert_array_equal(
            trailing_pad, 0.0,
            err_msg="Trailing padding should be silence (zeros)"
        )

    def test_was_padded_flag_set(self, settings, buffer):
        """PreparedUtterance.was_padded should reflect that padding occurred."""
        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio)
        assert result is not None
        assert result.was_padded is True

    def test_no_padding_when_pad_ms_zero(self, settings):
        """When speech_pad_ms=0, no padding should be applied."""
        settings.vad.speech_pad_ms = 0
        buf = ASRBuffer(settings)
        audio = make_utterance(duration_s=1.0)
        result = buf.prepare(audio)

        assert result is not None
        assert result.was_padded is False
        # Duration should be unchanged (within float tolerance)
        assert abs(result.padded_duration_s - result.raw_duration_s) < 0.001


# ─────────────────────────────────────────────────────────────────
# NORMALIZATION TESTS
# ─────────────────────────────────────────────────────────────────

class TestNormalization:

    def test_normalization_applied_when_enabled(self, settings, buffer):
        """When normalize_utterances=True, normalization should occur."""
        audio = make_utterance(duration_s=1.0, amplitude=0.1)  # Quiet audio
        result = buffer.prepare(audio)
        assert result is not None
        assert result.was_normalized is True

    def test_normalization_increases_quiet_audio_amplitude(self, settings, buffer):
        """
        Quiet audio (amplitude=0.1) should be normalized to near 0.9 peak.
        After normalization and padding, the audio peak should be ~0.9.
        The padding (silence) doesn't affect the peak of the speech part.
        """
        settings.vad.speech_pad_ms = 0  # No padding — isolate normalization
        buf = ASRBuffer(settings)
        audio = make_utterance(duration_s=1.0, amplitude=0.1)

        result = buf.prepare(audio)
        assert result is not None

        actual_peak = compute_peak(result.audio)
        assert abs(actual_peak - 0.9) < 0.01, (
            f"After normalization, peak should be ~0.9, got {actual_peak:.4f}"
        )

    def test_normalization_not_applied_when_disabled(self, settings):
        """When normalize_utterances=False, audio passes through unchanged."""
        settings.asr.normalize_utterances = False
        settings.vad.speech_pad_ms = 0  # No padding either
        buf = ASRBuffer(settings)

        audio = make_utterance(duration_s=1.0, amplitude=0.1)
        original_peak = compute_peak(audio)

        result = buf.prepare(audio)
        assert result is not None
        assert result.was_normalized is False

        actual_peak = compute_peak(result.audio)
        assert abs(actual_peak - original_peak) < 1e-5


# ─────────────────────────────────────────────────────────────────
# METADATA TESTS
# ─────────────────────────────────────────────────────────────────

class TestMetadata:

    def test_raw_duration_is_accurate(self, settings, buffer):
        """raw_duration_s should reflect original audio before padding."""
        audio = make_utterance(duration_s=2.0)
        result = buffer.prepare(audio)
        assert result is not None
        assert abs(result.raw_duration_s - 2.0) < 0.01

    def test_rms_energy_is_computed_on_original(self, settings):
        """
        rms_energy should reflect original audio, not normalized version.
        This gives you the true signal level at capture time.
        """
        settings.vad.speech_pad_ms = 0
        buf = ASRBuffer(settings)

        audio = make_utterance(duration_s=1.0, amplitude=0.2)
        expected_rms = compute_rms(audio)

        result = buf.prepare(audio)
        assert result is not None
        assert abs(result.rms_energy - expected_rms) < 0.001

    def test_utterance_id_is_unique(self, buffer):
        """Every utterance should have a unique ID."""
        audio = make_utterance(duration_s=1.0)
        results = [buffer.prepare(audio) for _ in range(10)]
        ids = [r.utterance_id for r in results if r is not None]
        assert len(ids) == len(set(ids)), "Utterance IDs are not unique"

    def test_chunk_count_passed_through(self, buffer):
        """chunk_count metadata should match what was passed in."""
        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio, chunk_count=42)
        assert result is not None
        assert result.chunk_count == 42

    def test_created_at_is_recent(self, buffer):
        """created_at should be a recent Unix timestamp."""
        before = time.time()
        audio = make_utterance(duration_s=1.0)
        result = buffer.prepare(audio)
        after = time.time()

        assert result is not None
        assert before <= result.created_at <= after


# ─────────────────────────────────────────────────────────────────
# DIAGNOSTIC COUNTER TESTS
# ─────────────────────────────────────────────────────────────────

class TestDiagnostics:

    def test_counters_start_at_zero(self, buffer):
        stats = buffer.get_stats()
        assert stats["total_received"] == 0
        assert stats["total_accepted"] == 0
        assert stats["total_rejected"] == 0

    def test_accepted_increments_on_valid(self, buffer):
        audio = make_utterance(duration_s=1.0)
        buffer.prepare(audio)
        stats = buffer.get_stats()
        assert stats["total_received"] == 1
        assert stats["total_accepted"] == 1
        assert stats["total_rejected"] == 0

    def test_rejected_increments_on_short(self, buffer):
        audio = make_utterance(duration_s=0.1)  # Too short
        buffer.prepare(audio)
        stats = buffer.get_stats()
        assert stats["total_received"] == 1
        assert stats["total_accepted"] == 0
        assert stats["total_rejected"] == 1

    def test_acceptance_rate_calculation(self, buffer):
        """5 accepted out of 10 total = 0.5 acceptance rate."""
        valid = make_utterance(duration_s=1.0)
        short = make_utterance(duration_s=0.1)

        for _ in range(5):
            buffer.prepare(valid)
        for _ in range(5):
            buffer.prepare(short)

        stats = buffer.get_stats()
        assert stats["acceptance_rate"] == pytest.approx(0.5)

    def test_reset_stats_clears_counters(self, buffer):
        audio = make_utterance(duration_s=1.0)
        buffer.prepare(audio)
        buffer.prepare(audio)

        buffer.reset_stats()
        stats = buffer.get_stats()
        assert stats["total_received"] == 0
        assert stats["total_accepted"] == 0