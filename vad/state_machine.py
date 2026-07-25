# vad/state_machine.py

"""
Utterance-level Voice Activity Detection state machine.

Responsibility: Track speech state over time and fire events when
speech starts, continues, or completes.

This module sits ABOVE vad/engine.py in the abstraction hierarchy:
    engine.py    → "is there speech in this 32ms chunk?"  (frame level)
    state_machine.py → "is an utterance happening right now?" (utterance level)

The state machine consumes VAD probabilities from the engine and
produces VADEvent objects that tell the ASR buffer what to do.

Key concepts implemented here:
    - Hysteresis: different thresholds for entering and exiting speech
    - Pre-roll:   include audio from before speech was detected
    - Post-roll:  include audio after speech appears to end
    - Minimum durations: ignore speech/silence segments that are too short
    - Maximum duration:  safety valve for runaway speech segments
"""

import numpy as np
import collections
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings
from utils.logging_config import get_logger
from utils.audio_utils import samples_to_ms, ms_to_samples

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# STATE DEFINITIONS
# ─────────────────────────────────────────────────────────────────

class VADState(Enum):
    """
    The four states of the VAD state machine.

    Why an Enum and not string constants like "silence"?
        Enums are safer — a typo like "sielce" is caught at definition time,
        not at runtime. Enums also give you autocomplete in IDEs and make
        comparisons explicit and readable.
    """
    SILENCE         = auto()   # No speech detected
    SPEECH_STARTING = auto()   # Speech tentatively detected, confirming
    SPEECH_ACTIVE   = auto()   # Speech confirmed and ongoing
    SPEECH_ENDING   = auto()   # Silence detected, waiting to confirm end


class VADEvent(Enum):
    """
    Events fired by the state machine.

    These events are what the ASR buffer and pipeline coordinator
    respond to. They decouple the state machine from its consumers.
    """
    SPEECH_STARTED      = auto()  # New utterance beginning
    SPEECH_CHUNK        = auto()  # Audio chunk to add to current utterance
    UTTERANCE_COMPLETE  = auto()  # Utterance finished — ready for ASR
    SILENCE             = auto()  # No speech — nothing to do


# ─────────────────────────────────────────────────────────────────
# TRANSITION RESULT — what the state machine returns per chunk
# ─────────────────────────────────────────────────────────────────

@dataclass
class VADTransition:
    """
    The result of processing one audio chunk through the state machine.

    Every call to process_chunk() returns one of these.
    It tells the caller:
        - What event occurred
        - What state we are in now
        - The audio chunk associated with this event (if any)
        - Whether an utterance is now complete

    Why a dataclass instead of a tuple?
        Tuples like (event, state, chunk) are fragile — you must remember
        the order. Dataclasses are self-documenting. Adding a new field
        later doesn't break existing code.

    Fields:
        event:      The VADEvent that occurred this chunk.
        state:      The VADState we are in AFTER this chunk.
        chunk:      The audio chunk (present for SPEECH_CHUNK events).
        utterance:  The complete utterance audio (present for UTTERANCE_COMPLETE).
        speech_prob: The VAD probability that triggered this transition.
    """
    event: VADEvent
    state: VADState
    chunk: Optional[np.ndarray] = None
    utterance: Optional[np.ndarray] = None
    speech_prob: float = 0.0


# ─────────────────────────────────────────────────────────────────
# THE STATE MACHINE
# ─────────────────────────────────────────────────────────────────

class VADStateMachine:
    """
    Finite state machine for utterance-level speech detection.

    Processes one audio chunk at a time.
    Maintains state between calls.
    Fires VADEvents when meaningful transitions occur.

    Lifecycle:
        machine = VADStateMachine(settings)

        for chunk, prob in zip(audio_chunks, vad_probabilities):
            transition = machine.process_chunk(chunk, prob)

            if transition.event == VADEvent.SPEECH_STARTED:
                buffer.new_utterance()

            elif transition.event == VADEvent.SPEECH_CHUNK:
                buffer.add_chunk(transition.chunk)

            elif transition.event == VADEvent.UTTERANCE_COMPLETE:
                asr_queue.put(transition.utterance)

        machine.reset()  # between sessions

    NOT thread-safe: call from a single thread only.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        cfg = self.settings

        # ── Threshold shortcuts ──────────────────────────────────
        self._speech_threshold  = cfg.vad.speech_threshold
        self._silence_threshold = cfg.vad.silence_threshold

        # ── Convert time durations to chunk counts ───────────────
        # The state machine counts chunks, not milliseconds.
        # We convert once here so the main loop is just incrementing
        # counters and comparing integers — very fast.
        #
        # chunk_duration_ms = 32ms (Silero's requirement)
        # So 500ms silence = 500/32 ≈ 15 chunks of silence needed

        chunk_ms = cfg.audio.chunk_duration_ms

        # How many consecutive speech chunks before we confirm speech?
        # e.g., min_speech_duration_ms=256ms / 32ms = 8 chunks
        self._speech_confirm_chunks = max(
            1,
            ms_to_samples(cfg.vad.min_speech_duration_ms, 1000 // chunk_ms)
            # Note: we're converting ms to "chunk counts" here
            # ms_to_samples(256ms, 31.25 chunks/sec) = 8 chunks
        )
        # Simpler, clearer version:
        self._speech_confirm_chunks = max(
            1,
            round(cfg.vad.min_speech_duration_ms / chunk_ms)
        )

        # How many consecutive silence chunks before we end an utterance?
        self._silence_confirm_chunks = max(
            1,
            round(cfg.vad.min_silence_duration_ms / chunk_ms)
        )

        # Maximum utterance length in chunks (safety valve)
        max_ms = cfg.asr.max_utterance_duration_s * 1000
        self._max_utterance_chunks = round(max_ms / chunk_ms)

        # Pre-roll: how many chunks to keep BEFORE speech is detected
        # This captures the beginning of words that arrive before VAD fires
        self._preroll_chunks = max(
            1,
            round(cfg.vad.speech_pad_ms / chunk_ms)
        )

        logger.debug(
            f"VADStateMachine initialized | "
            f"speech_confirm={self._speech_confirm_chunks} chunks "
            f"silence_confirm={self._silence_confirm_chunks} chunks "
            f"max_utterance={self._max_utterance_chunks} chunks "
            f"preroll={self._preroll_chunks} chunks"
        )

        # ── State initialization ─────────────────────────────────
        self._reset_state()

    def _reset_state(self) -> None:
        """
        Initialize (or re-initialize) all state variables.

        Called at construction and by reset().
        Separated so we can reset without reconstructing the object.
        """
        # Current state
        self._state = VADState.SILENCE

        # Counters: how many consecutive chunks have we seen
        # in the current direction?
        self._consecutive_speech_chunks  = 0
        self._consecutive_silence_chunks = 0
        self._utterance_chunk_count      = 0

        # Pre-roll ring buffer: keeps the last N chunks
        # so we can include them when speech starts
        # deque with maxlen automatically drops old items
        self._preroll_buffer: collections.deque = collections.deque(
            maxlen=self._preroll_chunks
        )

        # Utterance buffer: accumulates chunks for the current utterance
        self._utterance_chunks: list[np.ndarray] = []

        logger.debug("VADStateMachine state reset")

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        speech_probability: float,
    ) -> VADTransition:
        """
        Process one audio chunk and return the resulting state transition.

        This is the heart of the state machine. It is called for every
        single audio chunk — roughly 31 times per second.

        Args:
            audio_chunk:       The raw audio (float32, SILERO_CHUNK_SIZE samples).
            speech_probability: The VAD engine's output for this chunk [0.0, 1.0].

        Returns:
            VADTransition describing what happened and what to do next.

        The logic follows the state diagram exactly:
            SILENCE       → check if speech starting
            SPEECH_STARTING → check if confirmed or false alarm
            SPEECH_ACTIVE   → check if silence starting
            SPEECH_ENDING   → check if utterance complete or speech resumed
        """

        # Always add to pre-roll buffer (even during speech)
        # We only USE the preroll at the start of an utterance,
        # but we always collect it so it's ready
        self._preroll_buffer.append(audio_chunk.copy())

        # Route to the handler for the current state
        if self._state == VADState.SILENCE:
            return self._handle_silence(audio_chunk, speech_probability)

        elif self._state == VADState.SPEECH_STARTING:
            return self._handle_speech_starting(audio_chunk, speech_probability)

        elif self._state == VADState.SPEECH_ACTIVE:
            return self._handle_speech_active(audio_chunk, speech_probability)

        elif self._state == VADState.SPEECH_ENDING:
            return self._handle_speech_ending(audio_chunk, speech_probability)

        else:
            # Should never happen — but if it does, fail loudly
            raise RuntimeError(f"Unknown VAD state: {self._state}")

    # ─────────────────────────────────────────────────────────────
    # STATE HANDLERS — one method per state
    # ─────────────────────────────────────────────────────────────

    def _handle_silence(
        self,
        chunk: np.ndarray,
        prob: float,
    ) -> VADTransition:
        """
        We are in SILENCE. Check if speech is starting.

        Exit condition: prob > speech_threshold
        Action on exit: transition to SPEECH_STARTING
        """
        if prob > self._speech_threshold:
            # Tentative speech detected — move to SPEECH_STARTING
            self._state = VADState.SPEECH_STARTING
            self._consecutive_speech_chunks = 1

            logger.debug(
                f"SILENCE → SPEECH_STARTING | prob={prob:.3f}"
            )

            return VADTransition(
                event=VADEvent.SILENCE,  # Not confirmed yet
                state=self._state,
                speech_prob=prob,
            )

        # Still silence
        return VADTransition(
            event=VADEvent.SILENCE,
            state=self._state,
            speech_prob=prob,
        )

    def _handle_speech_starting(
        self,
        chunk: np.ndarray,
        prob: float,
    ) -> VADTransition:
        """
        We MIGHT be in speech. We're waiting for confirmation.

        This state prevents false positives: a single loud chunk
        (door slam, cough, keyboard click) won't trigger transcription.
        We require N consecutive above-threshold chunks.

        Two exit conditions:
            1. prob > speech_threshold again → increment counter
               If counter reaches speech_confirm_chunks → SPEECH_ACTIVE
            2. prob drops below speech_threshold → false alarm → SILENCE
        """
        if prob > self._speech_threshold:
            self._consecutive_speech_chunks += 1

            if self._consecutive_speech_chunks >= self._speech_confirm_chunks:
                # Speech confirmed. Transition to SPEECH_ACTIVE.
                self._state = VADState.SPEECH_ACTIVE
                self._consecutive_silence_chunks = 0
                self._utterance_chunk_count = 0

                # Start utterance buffer with pre-roll chunks
                # This recovers audio from before we detected speech
                self._utterance_chunks = list(self._preroll_buffer)

                logger.info(
                    f"SPEECH_STARTING → SPEECH_ACTIVE | "
                    f"prob={prob:.3f} | "
                    f"preroll_chunks={len(self._utterance_chunks)}"
                )

                return VADTransition(
                    event=VADEvent.SPEECH_STARTED,
                    state=self._state,
                    chunk=chunk,
                    speech_prob=prob,
                )
            else:
                # Still accumulating confirmation chunks
                return VADTransition(
                    event=VADEvent.SILENCE,
                    state=self._state,
                    speech_prob=prob,
                )

        else:
            # Probability dropped. False alarm. Back to silence.
            logger.debug(
                f"SPEECH_STARTING → SILENCE (false alarm) | "
                f"prob={prob:.3f} after "
                f"{self._consecutive_speech_chunks} chunks"
            )

            self._state = VADState.SILENCE
            self._consecutive_speech_chunks = 0

            return VADTransition(
                event=VADEvent.SILENCE,
                state=self._state,
                speech_prob=prob,
            )

    def _handle_speech_active(
        self,
        chunk: np.ndarray,
        prob: float,
    ) -> VADTransition:
        """
        We ARE in speech. Accumulate chunks. Watch for silence.

        Every chunk here gets added to the utterance buffer.
        We also watch for:
            1. Probability dropping below silence_threshold → SPEECH_ENDING
            2. Utterance exceeding max duration → force flush to ASR
        """
        # Add chunk to utterance
        self._utterance_chunks.append(chunk.copy())
        self._utterance_chunk_count += 1

        # Safety valve: if utterance runs too long, force-flush
        if self._utterance_chunk_count >= self._max_utterance_chunks:
            logger.warning(
                f"Utterance exceeded max duration "
                f"({self._max_utterance_chunks} chunks). "
                f"Force-flushing to ASR."
            )
            return self._complete_utterance(prob, force=True)

        if prob < self._silence_threshold:
            # Tentative silence detected
            self._state = VADState.SPEECH_ENDING
            self._consecutive_silence_chunks = 1

            logger.debug(
                f"SPEECH_ACTIVE → SPEECH_ENDING | prob={prob:.3f}"
            )

            return VADTransition(
                event=VADEvent.SPEECH_CHUNK,
                state=self._state,
                chunk=chunk,
                speech_prob=prob,
            )

        # Still active speech
        self._consecutive_silence_chunks = 0

        return VADTransition(
            event=VADEvent.SPEECH_CHUNK,
            state=self._state,
            chunk=chunk,
            speech_prob=prob,
        )

    def _handle_speech_ending(
        self,
        chunk: np.ndarray,
        prob: float,
    ) -> VADTransition:
        """
        Speech MIGHT be ending. We're waiting to confirm silence.

        This state prevents choppy splitting: a brief pause in the
        middle of a sentence won't end the utterance.

        Two exit conditions:
            1. Silence persists for silence_confirm_chunks → UTTERANCE_COMPLETE
            2. Speech resumes (prob > speech_threshold) → back to SPEECH_ACTIVE
        """
        # Always accumulate during SPEECH_ENDING
        # We keep post-roll audio — if speech resumes, we have it
        # If silence is confirmed, this becomes the tail of the utterance
        self._utterance_chunks.append(chunk.copy())
        self._utterance_chunk_count += 1

        # Safety valve applies here too
        if self._utterance_chunk_count >= self._max_utterance_chunks:
            return self._complete_utterance(prob, force=True)

        if prob > self._speech_threshold:
            # Speech resumed — back to active
            self._state = VADState.SPEECH_ACTIVE
            self._consecutive_silence_chunks = 0

            logger.debug(
                f"SPEECH_ENDING → SPEECH_ACTIVE (speech resumed) | "
                f"prob={prob:.3f}"
            )

            return VADTransition(
                event=VADEvent.SPEECH_CHUNK,
                state=self._state,
                chunk=chunk,
                speech_prob=prob,
            )

        elif prob < self._silence_threshold:
            self._consecutive_silence_chunks += 1

            if self._consecutive_silence_chunks >= self._silence_confirm_chunks:
                # Silence confirmed — utterance is complete
                return self._complete_utterance(prob)

            # Silence still accumulating
            return VADTransition(
                event=VADEvent.SPEECH_CHUNK,
                state=self._state,
                chunk=chunk,
                speech_prob=prob,
            )

        else:
            # Probability in hysteresis band (between thresholds)
            # Don't change state — wait for clearer signal
            return VADTransition(
                event=VADEvent.SPEECH_CHUNK,
                state=self._state,
                chunk=chunk,
                speech_prob=prob,
            )

    # ─────────────────────────────────────────────────────────────
    # UTTERANCE COMPLETION
    # ─────────────────────────────────────────────────────────────

    def _complete_utterance(
        self,
        prob: float,
        force: bool = False,
    ) -> VADTransition:
        """
        Finalize an utterance and prepare it for ASR.

        This is called when we are confident the speaker has finished,
        or when the safety valve triggers.

        Steps:
            1. Concatenate all buffered chunks into one array
            2. Transition back to SILENCE
            3. Fire UTTERANCE_COMPLETE event with the full audio
            4. Clear the utterance buffer
        """
        utterance_audio = np.concatenate(self._utterance_chunks)

        duration_ms = (len(utterance_audio) / self.settings.audio.sample_rate) * 1000

        logger.info(
            f"{'FORCE-' if force else ''}UTTERANCE_COMPLETE | "
            f"duration_ms={duration_ms:.0f} | "
            f"chunks={self._utterance_chunk_count} | "
            f"state={self._state.name} → SILENCE"
        )

        # Transition back to silence
        self._state = VADState.SILENCE
        self._consecutive_speech_chunks  = 0
        self._consecutive_silence_chunks = 0
        self._utterance_chunk_count      = 0
        self._utterance_chunks           = []

        return VADTransition(
            event=VADEvent.UTTERANCE_COMPLETE,
            state=self._state,
            utterance=utterance_audio,
            speech_prob=prob,
        )

    # ─────────────────────────────────────────────────────────────
    # CONTROL METHODS
    # ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset all state. Call between sessions or conversations.

        If called mid-utterance (e.g., user interrupts session),
        the current utterance buffer is discarded.
        """
        self._reset_state()
        logger.debug("VADStateMachine reset")

    def flush(self) -> Optional[VADTransition]:
        """
        Force-complete any in-progress utterance.

        Call this when the audio stream ends (e.g., end of file,
        user pressed stop) to ensure the final utterance is not lost.

        Returns:
            A UTTERANCE_COMPLETE transition if there was speech in progress.
            None if we were in SILENCE.
        """
        if self._state == VADState.SILENCE:
            logger.debug("flush() called in SILENCE state — nothing to flush")
            return None

        if not self._utterance_chunks:
            self._reset_state()
            return None

        logger.info(
            f"Flushing in-progress utterance | "
            f"state={self._state.name} | "
            f"chunks={self._utterance_chunk_count}"
        )

        return self._complete_utterance(prob=0.0, force=True)

    @property
    def state(self) -> VADState:
        """Current state. Read-only property."""
        return self._state

    @property
    def utterance_duration_ms(self) -> float:
        """
        Duration of the currently accumulating utterance in milliseconds.
        Returns 0 if in SILENCE.

        Useful for logging and for the safety valve check.
        """
        samples = self._utterance_chunk_count * self.settings.audio.chunk_size
        return samples_to_ms(samples, self.settings.audio.sample_rate)