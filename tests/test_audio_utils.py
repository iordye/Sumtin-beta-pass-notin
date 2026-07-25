# tests/test_audio_utils.py

"""
Testing audio utility functions.

Philosophy: test the math, not the implementation.
We verify that the RESULTS are correct, not HOW the function
computes them. This lets us optimize the implementation later
without breaking tests.

What failures mean:
    - compute_rms wrong → VAD thresholds are wrong → speech detection fails
    - convert_to_float32 wrong → audio values are 32768x too large →
      model gets garbage input, silently produces garbage output
    - stereo_to_mono wrong → shape mismatch crashes VAD engine
    - pad_audio wrong → ASR clips word boundaries → accuracy drops

These are all silent correctness bugs. The pipeline runs,
it just produces wrong answers. The tests here catch them.
"""

import numpy as np
import pytest
from utils.audio_utils import (
    validate_audio_chunk,
    convert_to_float32,
    compute_rms,
    compute_peak,
    is_silent,
    normalize_audio,
    stereo_to_mono,
    pad_audio,
    samples_to_ms,
    ms_to_samples,
    samples_to_seconds,
    AUDIO_DTYPE,
    SILENCE_RMS_FLOOR,
)


# ─────────────────────────────────────────────────────────────────
# FIXTURES — reusable audio arrays for tests
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def silence_chunk():
    """480 samples of digital silence (all zeros). Represents a 30ms silent chunk."""
    return np.zeros(480, dtype=np.float32)

@pytest.fixture
def sine_wave_chunk():
    """
    480 samples of a 440Hz sine wave at 16kHz.
    A pure tone — mathematically perfect audio for testing.
    RMS of a sine wave with amplitude A = A / sqrt(2) ≈ 0.707 * A.
    So amplitude=0.5 → RMS ≈ 0.354.
    """
    t = np.linspace(0, 480 / 16000, 480, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

@pytest.fixture
def int16_chunk():
    """A sine wave in int16 format — as a real microphone might deliver."""
    t = np.linspace(0, 480 / 16000, 480, endpoint=False)
    # int16 range is ±32767, so full-scale sine uses amplitude 16384
    return (16384 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)

@pytest.fixture
def stereo_chunk():
    """480 samples of stereo audio, shape (480, 2)."""
    t = np.linspace(0, 480 / 16000, 480, endpoint=False)
    left = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    return np.stack([left, right], axis=1)


# ─────────────────────────────────────────────────────────────────
# VALIDATION TESTS
# ─────────────────────────────────────────────────────────────────

class TestValidateAudioChunk:

    def test_valid_chunk_passes(self, sine_wave_chunk):
        """Valid float32 1D array should not raise."""
        validate_audio_chunk(sine_wave_chunk, caller="test")

    def test_rejects_non_array(self):
        """Lists and tuples should be rejected — we need numpy arrays."""
        with pytest.raises(TypeError, match="numpy array"):
            validate_audio_chunk([0.0, 0.1, 0.2], caller="test")

    def test_rejects_2d_array(self):
        """Stereo audio (2D) must be converted before validation."""
        audio_2d = np.zeros((480, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            validate_audio_chunk(audio_2d, caller="test")

    def test_rejects_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            validate_audio_chunk(np.array([], dtype=np.float32), caller="test")

    def test_rejects_wrong_dtype(self, int16_chunk):
        """int16 audio must be converted before entering the pipeline."""
        with pytest.raises(ValueError, match="float32"):
            validate_audio_chunk(int16_chunk, caller="test")

    def test_rejects_wrong_sample_count(self, sine_wave_chunk):
        """VAD requires exact sample counts. Wrong size must raise."""
        with pytest.raises(ValueError, match="Expected 512"):
            validate_audio_chunk(sine_wave_chunk, expected_samples=512, caller="test")

    def test_accepts_correct_sample_count(self, sine_wave_chunk):
        """Correct sample count should pass without raising."""
        validate_audio_chunk(sine_wave_chunk, expected_samples=480, caller="test")


# ─────────────────────────────────────────────────────────────────
# CONVERSION TESTS
# ─────────────────────────────────────────────────────────────────

class TestConvertToFloat32:

    def test_int16_to_float32_range(self, int16_chunk):
        """
        int16 max value (32767) should map to approximately 1.0.
        int16 min value (-32768) should map to approximately -1.0.
        This is the most important conversion test.
        """
        result = convert_to_float32(int16_chunk)
        assert result.dtype == np.float32
        assert np.max(np.abs(result)) <= 1.0 + 1e-5  # small tolerance for float math

    def test_int16_preserves_waveform_shape(self, int16_chunk):
        """
        Conversion should scale values, not change their relative relationships.
        The waveform shape must be preserved.
        """
        result = convert_to_float32(int16_chunk)
        # The sign of each sample should be preserved
        assert np.all(np.sign(result) == np.sign(int16_chunk))

    def test_float32_passthrough(self, sine_wave_chunk):
        """float32 input should return the same object — no unnecessary copy."""
        result = convert_to_float32(sine_wave_chunk)
        assert result is sine_wave_chunk  # Same object, not a copy

    def test_float64_conversion(self):
        """float64 should convert to float32."""
        audio_f64 = np.array([0.1, 0.5, -0.3], dtype=np.float64)
        result = convert_to_float32(audio_f64)
        assert result.dtype == np.float32

    def test_uint8_center_maps_to_zero(self):
        """
        For uint8 audio, value 128 is "silence" (the center of 0-255).
        It should map to 0.0.
        """
        audio_u8 = np.array([128], dtype=np.uint8)
        result = convert_to_float32(audio_u8)
        assert abs(result[0]) < 1e-5


# ─────────────────────────────────────────────────────────────────
# ENERGY ANALYSIS TESTS
# ─────────────────────────────────────────────────────────────────

class TestComputeRMS:

    def test_silence_rms_is_zero(self, silence_chunk):
        """All-zero audio has RMS of exactly 0."""
        assert compute_rms(silence_chunk) == 0.0

    def test_sine_wave_rms(self, sine_wave_chunk):
        """
        For a sine wave with amplitude A, RMS = A / sqrt(2).
        Our sine wave has amplitude 0.5.
        Expected RMS = 0.5 / sqrt(2) ≈ 0.3536.
        We allow a small tolerance for floating-point imprecision.
        """
        expected_rms = 0.5 / np.sqrt(2)
        actual_rms = compute_rms(sine_wave_chunk)
        assert abs(actual_rms - expected_rms) < 0.01

    def test_rms_is_always_non_negative(self):
        """RMS is a magnitude — it must never be negative."""
        audio = np.array([-0.5, -0.3, 0.1, 0.8, -0.9], dtype=np.float32)
        assert compute_rms(audio) >= 0.0

    def test_louder_audio_has_higher_rms(self):
        """Basic sanity: louder audio should produce higher RMS."""
        quiet = np.full(480, 0.1, dtype=np.float32)
        loud = np.full(480, 0.5, dtype=np.float32)
        assert compute_rms(loud) > compute_rms(quiet)

    def test_empty_audio_returns_zero(self):
        assert compute_rms(np.array([], dtype=np.float32)) == 0.0


class TestIsSilent:

    def test_zeros_are_silent(self, silence_chunk):
        assert is_silent(silence_chunk) is True

    def test_sine_wave_is_not_silent(self, sine_wave_chunk):
        assert is_silent(sine_wave_chunk) is False

    def test_custom_threshold(self):
        """A stricter threshold should classify more audio as non-silent."""
        quiet_audio = np.full(480, 0.005, dtype=np.float32)
        # Default threshold (1e-6): definitely not silent
        assert is_silent(quiet_audio, threshold=SILENCE_RMS_FLOOR) is False
        # Aggressive threshold (0.01): now it is "silent"
        assert is_silent(quiet_audio, threshold=0.01) is True


# ─────────────────────────────────────────────────────────────────
# NORMALIZATION TESTS
# ─────────────────────────────────────────────────────────────────

class TestNormalizeAudio:

    def test_normalized_peak_matches_target(self, sine_wave_chunk):
        """After normalization, peak should equal target_peak."""
        target = 0.9
        result = normalize_audio(sine_wave_chunk, target_peak=target)
        actual_peak = compute_peak(result)
        assert abs(actual_peak - target) < 1e-5

    def test_normalization_preserves_waveform_shape(self, sine_wave_chunk):
        """
        Normalization is just scaling — the waveform shape must be identical.
        We verify this by checking that the normalized version is proportional
        to the original.
        """
        result = normalize_audio(sine_wave_chunk, target_peak=0.9)
        # The ratio between any two samples should be preserved
        # (where original is non-zero)
        nonzero_mask = np.abs(sine_wave_chunk) > 1e-6
        ratios = result[nonzero_mask] / sine_wave_chunk[nonzero_mask]
        # All ratios should be the same scale factor
        assert np.allclose(ratios, ratios[0], rtol=1e-4)

    def test_silence_not_normalized(self, silence_chunk):
        """
        Normalizing silence would divide by zero or amplify noise.
        Silent audio should be returned unchanged.
        """
        result = normalize_audio(silence_chunk)
        np.testing.assert_array_equal(result, silence_chunk)

    def test_returns_float32(self, sine_wave_chunk):
        result = normalize_audio(sine_wave_chunk)
        assert result.dtype == np.float32

    def test_does_not_modify_original(self, sine_wave_chunk):
        """
        Pure functions should not mutate their inputs.
        This is critical when the same chunk is processed by multiple modules.
        """
        original_copy = sine_wave_chunk.copy()
        normalize_audio(sine_wave_chunk)
        np.testing.assert_array_equal(sine_wave_chunk, original_copy)


# ─────────────────────────────────────────────────────────────────
# CHANNEL CONVERSION TESTS
# ─────────────────────────────────────────────────────────────────

class TestStereoToMono:

    def test_mono_passthrough(self, sine_wave_chunk):
        """Mono input should pass through unchanged."""
        result = stereo_to_mono(sine_wave_chunk)
        assert result is sine_wave_chunk

    def test_stereo_to_mono_shape(self, stereo_chunk):
        """Stereo (N, 2) should become mono (N,)."""
        result = stereo_to_mono(stereo_chunk)
        assert result.ndim == 1
        assert len(result) == len(stereo_chunk)

    def test_stereo_to_mono_averages_channels(self, stereo_chunk):
        """
        Mono should be the average of left and right.
        We verify by computing what the average should be for sample 0.
        """
        result = stereo_to_mono(stereo_chunk)
        expected_first_sample = np.mean(stereo_chunk[0]).astype(np.float32)
        assert abs(result[0] - expected_first_sample) < 1e-6

    def test_rejects_invalid_shape(self):
        """3D audio is not supported."""
        bad_audio = np.zeros((480, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            stereo_to_mono(bad_audio)


# ─────────────────────────────────────────────────────────────────
# PADDING TESTS
# ─────────────────────────────────────────────────────────────────

class TestPadAudio:

    def test_padded_length(self, sine_wave_chunk):
        """
        Output length = original + 2 * pad_samples.
        200ms at 16kHz = 3200 samples per side.
        Original = 480.
        Expected = 480 + 3200 + 3200 = 6880.
        """
        result = pad_audio(sine_wave_chunk, pad_ms=200, sample_rate=16000)
        pad_samples = ms_to_samples(200, 16000)  # 3200
        expected_length = len(sine_wave_chunk) + 2 * pad_samples
        assert len(result) == expected_length

    def test_padding_is_silence(self, sine_wave_chunk):
        """The padding samples should all be zero (digital silence)."""
        pad_ms = 100
        pad_samples = ms_to_samples(pad_ms, 16000)
        result = pad_audio(sine_wave_chunk, pad_ms=pad_ms, sample_rate=16000)
        # First pad_samples samples should all be 0.0
        np.testing.assert_array_equal(result[:pad_samples], 0.0)
        # Last pad_samples samples should all be 0.0
        np.testing.assert_array_equal(result[-pad_samples:], 0.0)

    def test_original_audio_preserved_in_middle(self, sine_wave_chunk):
        """The original audio should sit unchanged between the padding."""
        pad_ms = 50
        pad_samples = ms_to_samples(pad_ms, 16000)
        result = pad_audio(sine_wave_chunk, pad_ms=pad_ms, sample_rate=16000)
        middle = result[pad_samples: pad_samples + len(sine_wave_chunk)]
        np.testing.assert_array_equal(middle, sine_wave_chunk)


# ─────────────────────────────────────────────────────────────────
# DURATION UTILITY TESTS
# ─────────────────────────────────────────────────────────────────

class TestDurationUtils:

    def test_samples_to_ms_one_chunk(self):
        """480 samples at 16kHz = 30ms."""
        assert samples_to_ms(480, 16000) == pytest.approx(30.0)

    def test_ms_to_samples_roundtrip(self):
        """Converting 30ms to samples and back should give 30ms."""
        samples = ms_to_samples(30, 16000)
        back_to_ms = samples_to_ms(samples, 16000)
        assert back_to_ms == pytest.approx(30.0)

    def test_samples_to_seconds(self):
        """16000 samples at 16kHz = 1 second."""
        assert samples_to_seconds(16000, 16000) == pytest.approx(1.0)

    def test_ms_to_samples_integer_result(self):
        """Sample counts are always integers."""
        result = ms_to_samples(30, 16000)
        assert isinstance(result, int)