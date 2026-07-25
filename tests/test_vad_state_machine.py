# tests/test_vad_state_machine.py

"""
Testing the VAD state machine.

This is where state machine testing shines: we can feed exact probability
sequences and verify exact state transitions and events. No real audio
or models needed — just controlled sequences of floats.

Testing strategy:
    Build synthetic probability sequences that exercise every
    transition path in the state machine. Verify:
        - Correct state after each chunk
        - Correct event fired at each transition
        - Pre-roll audio is included in utterances
        - Hysteresis works (mid-speech dip doesn't split utterance)
        - Max duration safety valve triggers correctly
        - flush() captures in-progress utterances at end of stream

What failures mean:
    - Wrong state transitions → utterances split or merged incorrectly
    - Pre-roll not included → first phonemes clipped → bad ASR accuracy
    - Hysteresis broken → one sentence becomes many fragments
    - flush() broken → last utterance in a session is always lost
    - reset() broken → state leaks between sessions
"""

import numpy as np
import pytest
from config.settings import Settings
from vad.state_machine import (
    VADStateMachine,
    VADState,
    VADEvent,
    VADTransition,
)
from vad.engine import SILERO_CHUNK_SIZE


# ─────────────────────────────────────────────────────────────────
# FIXTURES AND HELPERS
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    s = Settings()
    # Use small confirmation counts for faster testing
    # (real values require many chunks to confirm; tests should be quick)
    s.vad.speech_threshold = 0.5
    s.vad.silence_threshold = 0.35
    s.vad.min_speech_duration_ms = 64    # 2 chunks at 32ms
    s.vad.min_silence_duration_ms = 96   # 3 chunks at 32ms
    s.vad.speech_pad_ms = 64             # 2 chunks preroll
    s.asr.max_utterance_duration_s = 10.0
    return s

@pytest.fixture
def machine(settings):
    return VADStateMachine(settings)

def make_chunk(value: float = 0.1) -> np.ndarray:
    """
    Create a test audio chunk.
    Value doesn't matter much for state machine tests —
    we're testing state transitions, not audio content.
    We use non-zero values so silence fast-path doesn't trigger.
    """
    return np.full(SILERO_CHUNK_SIZE, value, dtype=np.float32)

def feed_sequence(
    machine: VADStateMachine,
    probabilities: list[float],
) -> list[VADTransition]:
    """
    Feed a sequence of probabilities to the state machine.
    Returns all transitions produced.
    Helper for writing readable test sequences.
    """
    chunk = make_chunk(0.1)
    return [machine.process_chunk(chunk, prob) for prob in probabilities]

def get_events(transitions: list[VADTransition]) -> list[VADEvent]:
    """Extract just the events from a transition sequence."""
    return [t.event for t in transitions]

def get_states(transitions: list[VADTransition]) -> list[VADState]:
    """Extract just the states from a transition sequence."""
    return [t.state for t in transitions]


# ─────────────────────────────────────────────────────────────────
# INITIAL STATE TESTS
# ─────────────────────────────────────────────────────────────────

class TestInitialState:

    def test_starts_in_silence(self, machine):
        assert machine.state == VADState.SILENCE

    def test_reset_returns_to_silence(self, machine, settings):
        # Drive into speech state
        feed_sequence(machine, [0.9] * 10)
        assert machine.state != VADState.SILENCE

        machine.reset()
        assert machine.state == VADState.SILENCE


# ─────────────────────────────────────────────────────────────────
# SILENCE STATE TESTS
# ─────────────────────────────────────────────────────────────────

class TestSilenceState:

    def test_silence_below_threshold_stays_silence(self, machine):
        """Probabilities below threshold should keep us in SILENCE."""
        transitions = feed_sequence(machine, [0.1, 0.2, 0.3, 0.1])
        for t in transitions:
            assert t.state == VADState.SILENCE
            assert t.event == VADEvent.SILENCE

    def test_single_spike_moves_to_starting(self, machine):
        """One above-threshold chunk should move to SPEECH_STARTING."""
        transitions = feed_sequence(machine, [0.1, 0.6])
        assert transitions[-1].state == VADState.SPEECH_STARTING

    def test_silence_event_while_silent(self, machine):
        """Events during silence should all be SILENCE."""
        transitions = feed_sequence(machine, [0.0, 0.1, 0.2])
        assert all(t.event == VADEvent.SILENCE for t in transitions)


# ─────────────────────────────────────────────────────────────────
# SPEECH STARTING STATE TESTS
# ─────────────────────────────────────────────────────────────────

class TestSpeechStartingState:

    def test_false_alarm_returns_to_silence(self, machine):
        """
        One above-threshold chunk followed by below-threshold
        should be a false alarm → back to SILENCE.
        """
        transitions = feed_sequence(machine, [0.6, 0.2])
        # After spike then drop: back to SILENCE
        assert transitions[-1].state == VADState.SILENCE
        assert transitions[-1].event == VADEvent.SILENCE

    def test_confirmed_speech_moves_to_active(self, settings, machine):
        """
        speech_confirm_chunks consecutive above-threshold chunks
        should move to SPEECH_ACTIVE.
        Settings: min_speech_duration_ms=64, chunk=32ms → 2 chunks needed.
        """
        # 2 chunks above threshold should confirm speech
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        probs = [0.8] * confirm_count
        transitions = feed_sequence(machine, probs)

        final = transitions[-1]
        assert final.state == VADState.SPEECH_ACTIVE
        assert final.event == VADEvent.SPEECH_STARTED

    def test_speech_started_event_fired_exactly_once(self, settings, machine):
        """
        SPEECH_STARTED should fire exactly once per utterance,
        at the moment speech is confirmed.
        """
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        # Confirm + extra speech chunks
        probs = [0.8] * (confirm_count + 5)
        transitions = feed_sequence(machine, probs)

        started_events = [
            t for t in transitions if t.event == VADEvent.SPEECH_STARTED
        ]
        assert len(started_events) == 1, (
            f"Expected exactly 1 SPEECH_STARTED, got {len(started_events)}"
        )


# ─────────────────────────────────────────────────────────────────
# SPEECH ACTIVE STATE TESTS
# ─────────────────────────────────────────────────────────────────

class TestSpeechActiveState:

    def _enter_speech_active(self, machine, settings) -> None:
        """Helper: drive machine into SPEECH_ACTIVE."""
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        feed_sequence(machine, [0.8] * confirm_count)
        assert machine.state == VADState.SPEECH_ACTIVE

    def test_speech_chunk_events_during_active(self, machine, settings):
        """
        While in SPEECH_ACTIVE, every chunk should fire SPEECH_CHUNK.
        """
        self._enter_speech_active(machine, settings)
        transitions = feed_sequence(machine, [0.9, 0.85, 0.92])
        for t in transitions:
            assert t.event == VADEvent.SPEECH_CHUNK
            assert t.chunk is not None

    def test_drop_below_silence_threshold_moves_to_ending(
        self, machine, settings
    ):
        """
        Probability dropping below silence_threshold should
        move to SPEECH_ENDING (not immediately end utterance).
        """
        self._enter_speech_active(machine, settings)
        transitions = feed_sequence(machine, [0.2])
        assert transitions[-1].state == VADState.SPEECH_ENDING

    def test_hysteresis_band_stays_active(self, machine, settings):
        """
        Probability in hysteresis band (between silence and speech
        thresholds) should NOT change state.
        settings: silence=0.35, speech=0.5 → band is [0.35, 0.50]
        """
        self._enter_speech_active(machine, settings)
        # 0.40 is in the hysteresis band
        transitions = feed_sequence(machine, [0.40, 0.42, 0.38])
        for t in transitions:
            assert t.state == VADState.SPEECH_ACTIVE, (
                f"Hysteresis band should keep SPEECH_ACTIVE, "
                f"got {t.state} at prob within band"
            )


# ─────────────────────────────────────────────────────────────────
# SPEECH ENDING AND COMPLETION TESTS
# ─────────────────────────────────────────────────────────────────

class TestSpeechEnding:

    def _enter_speech_ending(self, machine, settings) -> None:
        """Helper: drive machine into SPEECH_ENDING."""
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        feed_sequence(machine, [0.8] * confirm_count)   # → SPEECH_ACTIVE
        feed_sequence(machine, [0.2])                    # → SPEECH_ENDING
        assert machine.state == VADState.SPEECH_ENDING

    def test_speech_resumes_returns_to_active(self, machine, settings):
        """
        If speech probability rises above speech_threshold while in
        SPEECH_ENDING, we return to SPEECH_ACTIVE (not split utterance).
        """
        self._enter_speech_ending(machine, settings)
        transitions = feed_sequence(machine, [0.8])
        assert transitions[-1].state == VADState.SPEECH_ACTIVE

    def test_confirmed_silence_fires_utterance_complete(
        self, machine, settings
    ):
        """
        After silence_confirm_chunks consecutive below-threshold chunks,
        UTTERANCE_COMPLETE should fire.
        settings: min_silence_duration_ms=96, chunk=32ms → 3 chunks needed.
        """
        self._enter_speech_ending(machine, settings)

        silence_count = round(
            settings.vad.min_silence_duration_ms / settings.audio.chunk_duration_ms
        )
        transitions = feed_sequence(machine, [0.1] * silence_count)

        complete_events = [
            t for t in transitions if t.event == VADEvent.UTTERANCE_COMPLETE
        ]
        assert len(complete_events) == 1

    def test_utterance_complete_includes_audio(self, machine, settings):
        """
        The UTTERANCE_COMPLETE transition must include the utterance audio.
        Without audio, ASR has nothing to transcribe.
        """
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        silence_count = round(
            settings.vad.min_silence_duration_ms / settings.audio.chunk_duration_ms
        )

        # Full utterance lifecycle
        feed_sequence(machine, [0.8] * confirm_count)  # → SPEECH_ACTIVE
        speech_transitions = feed_sequence(machine, [0.9] * 5)  # speech
        feed_sequence(machine, [0.2])                  # → SPEECH_ENDING
        end_transitions = feed_sequence(machine, [0.1] * silence_count)

        complete = next(
            t for t in end_transitions
            if t.event == VADEvent.UTTERANCE_COMPLETE
        )

        assert complete.utterance is not None
        assert isinstance(complete.utterance, np.ndarray)
        assert len(complete.utterance) > 0

    def test_utterance_complete_returns_to_silence(self, machine, settings):
        """After UTTERANCE_COMPLETE, state must be SILENCE."""
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        silence_count = round(
            settings.vad.min_silence_duration_ms / settings.audio.chunk_duration_ms
        )

        feed_sequence(machine, [0.8] * confirm_count)
        feed_sequence(machine, [0.9] * 3)
        feed_sequence(machine, [0.2])
        feed_sequence(machine, [0.1] * silence_count)

        assert machine.state == VADState.SILENCE


# ─────────────────────────────────────────────────────────────────
# PRE-ROLL TESTS
# ─────────────────────────────────────────────────────────────────

class TestPreRoll:

    def test_utterance_includes_preroll_audio(self, machine, settings):
        """
        The utterance audio should include chunks from BEFORE speech
        was confirmed (pre-roll). This captures word beginnings.

        We verify by using distinguishable chunk values:
        pre-roll chunks have value 0.1, speech chunks have value 0.5.
        The utterance should contain both.
        """
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        silence_count = round(
            settings.vad.min_silence_duration_ms / settings.audio.chunk_duration_ms
        )
        preroll_count = round(
            settings.vad.speech_pad_ms / settings.audio.chunk_duration_ms
        )

        # Feed silence chunks (these go into pre-roll buffer)
        silence_chunk = np.full(SILERO_CHUNK_SIZE, 0.001, dtype=np.float32)
        for _ in range(preroll_count + 2):
            machine.process_chunk(silence_chunk, 0.1)

        # Feed speech chunks (these trigger confirmation)
        speech_chunk = np.full(SILERO_CHUNK_SIZE, 0.5, dtype=np.float32)
        for _ in range(confirm_count):
            machine.process_chunk(speech_chunk, 0.8)

        # Feed more speech
        for _ in range(3):
            machine.process_chunk(speech_chunk, 0.9)

        # End utterance
        end_chunk = np.full(SILERO_CHUNK_SIZE, 0.001, dtype=np.float32)
        final_transition = None
        for _ in range(silence_count):
            t = machine.process_chunk(end_chunk, 0.1)
            if t.event == VADEvent.UTTERANCE_COMPLETE:
                final_transition = t

        assert final_transition is not None
        utterance = final_transition.utterance

        # Utterance should be longer than just the confirmed speech chunks
        # because pre-roll added chunks from before confirmation
        min_expected_without_preroll = (
            (confirm_count + 3 + silence_count) * SILERO_CHUNK_SIZE
        )
        assert len(utterance) > min_expected_without_preroll, (
            f"Utterance length {len(utterance)} should be longer than "
            f"{min_expected_without_preroll} (without pre-roll). "
            f"Pre-roll is not being included."
        )


# ─────────────────────────────────────────────────────────────────
# HYSTERESIS INTEGRATION TEST
# ─────────────────────────────────────────────────────────────────

class TestHysteresis:

    def test_mid_speech_dip_does_not_split_utterance(
        self, machine, settings
    ):
        """
        A brief dip below speech_threshold during an utterance
        should NOT split it into two utterances.

        This is the most important behavioral test.
        It verifies that "Hello [pause] world" is ONE utterance.
        """
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        silence_count = round(
            settings.vad.min_silence_duration_ms / settings.audio.chunk_duration_ms
        )

        # Enter speech
        probs = (
            [0.8] * confirm_count +    # enter SPEECH_ACTIVE
            [0.9, 0.92, 0.88] +        # active speech
            [0.3, 0.28] +              # brief dip (fewer than silence_count)
            [0.91, 0.89, 0.93] +       # speech resumes
            [0.1] * silence_count      # real silence ends utterance
        )

        transitions = feed_sequence(machine, probs)

        complete_events = [
            t for t in transitions if t.event == VADEvent.UTTERANCE_COMPLETE
        ]

        assert len(complete_events) == 1, (
            f"Expected 1 utterance but got {len(complete_events)}. "
            f"Mid-speech dip incorrectly split the utterance."
        )


# ─────────────────────────────────────────────────────────────────
# FLUSH AND SAFETY VALVE TESTS
# ─────────────────────────────────────────────────────────────────

class TestFlushAndSafetyValve:

    def test_flush_in_silence_returns_none(self, machine):
        """Flushing while silent should return None — nothing to flush."""
        result = machine.flush()
        assert result is None

    def test_flush_during_speech_returns_complete(self, machine, settings):
        """
        If we flush mid-utterance (e.g., end of audio file),
        we should get UTTERANCE_COMPLETE with the accumulated audio.
        """
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )
        feed_sequence(machine, [0.8] * confirm_count)  # enter SPEECH_ACTIVE
        feed_sequence(machine, [0.9] * 5)              # accumulate speech

        # Force flush (simulating end of audio stream)
        result = machine.flush()

        assert result is not None
        assert result.event == VADEvent.UTTERANCE_COMPLETE
        assert result.utterance is not None
        assert len(result.utterance) > 0

    def test_max_duration_forces_flush(self, settings):
        """
        Utterances exceeding max_utterance_duration_s should be
        force-completed even if speech is still detected.
        """
        # Use a very short max duration for testing
        settings.asr.max_utterance_duration_s = 0.5  # 500ms

        machine = VADStateMachine(settings)
        confirm_count = round(
            settings.vad.min_speech_duration_ms / settings.audio.chunk_duration_ms
        )

        # Enter speech
        feed_sequence(machine, [0.8] * confirm_count)

        # Keep feeding speech beyond the max duration
        # 500ms / 32ms = ~16 chunks max
        transitions = feed_sequence(machine, [0.9] * 50)

        force_complete = [
            t for t in transitions if t.event == VADEvent.UTTERANCE_COMPLETE
        ]

        assert len(force_complete) >= 1, (
            "Max duration safety valve should have force-completed utterance"
        )