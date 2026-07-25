# tests/test_settings.py

"""
Testing the settings module.

What we're testing:
    1. Default values are correct
    2. Computed properties work (chunk_size)
    3. Environment variable overrides work
    4. Settings can be modified for test-specific scenarios

Why this matters:
    If chunk_size is computed wrong, every downstream module gets the wrong
    buffer size. Silent bugs like this are the worst kind.

What failures would mean:
    - Wrong chunk_size → VAD gets wrong-sized chunks → runtime errors or wrong behavior
    - Environment overrides broken → can't configure without code changes
"""

import os
import pytest
from config.settings import Settings, AudioConfig, VADConfig


class TestAudioConfig:

    def test_default_sample_rate(self):
        """Standard speech sample rate should be 16kHz."""
        config = AudioConfig()
        assert config.sample_rate == 16000

    def test_chunk_size_computation(self):
        """
        chunk_size = sample_rate * chunk_duration_ms / 1000
        At 16kHz, 30ms: 16000 * 30 / 1000 = 480 samples.

        This is critical. Silero VAD REQUIRES exactly 512 samples at 16kHz
        (or 256 at 8kHz). If this is wrong, VAD will throw an error.
        (Note: we use 30ms = 480 here; we'll align to Silero's exact
        requirements in the VAD module.)
        """
        config = AudioConfig(sample_rate=16000, chunk_duration_ms=30)
        assert config.chunk_size == 480

    def test_chunk_size_different_rate(self):
        """Test chunk_size math with different parameters."""
        config = AudioConfig(sample_rate=8000, chunk_duration_ms=30)
        assert config.chunk_size == 240

    def test_chunk_size_is_readonly_computed(self):
        """chunk_size should be derived, not a free variable."""
        config = AudioConfig(sample_rate=16000, chunk_duration_ms=60)
        # 16000 * 60 / 1000 = 960
        assert config.chunk_size == 960


class TestVADConfig:

    def test_silence_threshold_lower_than_speech(self):
        """
        This is an architectural invariant.
        If silence_threshold >= speech_threshold, there's no hysteresis
        and the state machine will thrash between states on borderline audio.
        """
        config = VADConfig()
        assert config.silence_threshold < config.speech_threshold

    def test_default_thresholds_in_valid_range(self):
        config = VADConfig()
        assert 0.0 < config.speech_threshold < 1.0
        assert 0.0 < config.silence_threshold < 1.0


class TestSettings:

    def test_settings_instantiates_with_defaults(self):
        """The whole settings tree should build without errors."""
        settings = Settings()
        assert settings.audio is not None
        assert settings.vad is not None
        assert settings.asr is not None
        assert settings.pipeline is not None

    def test_env_override_asr_model(self, monkeypatch):
        """
        Environment variable override should work.
        This is how we'd swap models without code changes in production.
        """
        monkeypatch.setenv("ASR_MODEL", "tiny")
        settings = Settings()
        assert settings.asr.model_name == "tiny"

    def test_env_override_language_auto(self, monkeypatch):
        """'auto' should map to None (Whisper's auto-detect signal)."""
        monkeypatch.setenv("ASR_LANGUAGE", "auto")
        settings = Settings()
        assert settings.asr.language is None

    def test_settings_mutation_for_testing(self):
        """
        We should be able to modify settings for a specific test.
        This is how test files customize behavior without touching config files.
        """
        settings = Settings()
        settings.vad.speech_threshold = 0.9
        assert settings.vad.speech_threshold == 0.9
        # A fresh Settings() still has the default
        fresh = Settings()
        assert fresh.vad.speech_threshold == 0.5