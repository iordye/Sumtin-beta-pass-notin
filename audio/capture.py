# audio/capture.py

"""
Audio capture module — the entry point for all audio into the pipeline.

This module sits at the trust boundary between hardware and pipeline.
It transforms raw hardware audio into clean, validated numpy arrays.

Two sources are supported:
    MicrophoneCapture  — captures from a physical microphone
    FileCapture        — streams from a WAV file at realtime rate

Both implement the same interface: a stream() generator that yields
validated float32 numpy arrays of exactly chunk_size samples.

Why a generator?
    Generators use Python's 'yield' keyword to produce values one at a time.
    The caller can iterate them with 'for chunk in source.stream()'.
    This is memory-efficient (we never hold more than one chunk at a time)
    and composable (the generator can be wrapped, filtered, or logged
    without changing this module).

Why a class instead of a function?
    The capture source has STATE: is it running? what device? what stream?
    Classes are the right tool when you have both data (state) and behavior
    (methods) that belong together.
"""

import time
import wave
import threading
import numpy as np
from pathlib import Path
from typing import Generator, Optional
from abc import ABC, abstractmethod

from config.settings import Settings
from utils.logging_config import get_logger
from utils.audio_utils import (
    convert_to_float32,
    stereo_to_mono,
    validate_audio_chunk,
    compute_rms,
    compute_peak,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# ABSTRACT BASE — the contract both sources must fulfill
# ─────────────────────────────────────────────────────────────────

class AudioSource(ABC):
    """
    Abstract base class for all audio sources.

    Why abstract?
        We want to guarantee that MicrophoneCapture and FileCapture
        both have the same interface. If we later add NetworkCapture
        or SynthesizedCapture (for testing), they automatically fit
        into the pipeline.

        This is called "programming to an interface" — the pipeline
        coordinator will accept any AudioSource without knowing
        which concrete implementation it is.

    The contract:
        - stream() yields float32 numpy arrays
        - Each array has exactly settings.audio.chunk_size samples
        - Audio is mono (1 channel)
        - Sample rate is settings.audio.sample_rate
        - stop() signals the stream to end cleanly
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stop_event = threading.Event()

    @abstractmethod
    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Yield audio chunks as float32 numpy arrays.

        This is a generator — it yields one chunk at a time.
        The caller iterates it: for chunk in source.stream()

        Yields:
            numpy array, dtype=float32, shape=(chunk_size,)
        """
        ...

    def stop(self) -> None:
        """
        Signal the stream to stop after the current chunk.

        Thread-safe: can be called from any thread.
        The stream() generator will see this and stop yielding.

        Why threading.Event?
            It's the standard Python way to signal between threads.
            One thread calls stop() → sets the event.
            The other thread calls _stop_event.is_set() → sees True → stops.
            No locks needed. No shared mutable state.
        """
        logger.info(f"{self.__class__.__name__} stop requested")
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()


# ─────────────────────────────────────────────────────────────────
# MICROPHONE CAPTURE
# ─────────────────────────────────────────────────────────────────

class MicrophoneCapture(AudioSource):
    """
    Captures audio from a physical microphone using PyAudio.

    Lifecycle:
        1. __init__: store settings, don't open hardware yet
        2. stream(): open microphone, start yielding chunks
        3. stop(): signal stream to stop
        4. stream() cleans up hardware on exit

    Why not open the microphone in __init__?
        Hardware resources should be acquired as late as possible
        and released as early as possible. If you open the microphone
        in __init__ but then an error happens before stream() is called,
        you've leaked a hardware resource. Opening in stream() means
        the resource is held only while actively capturing.

        This pattern is called RAII (Resource Acquisition Is
        Initialization) in C++, and "context manager" in Python.
        We implement the same idea manually here.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._pyaudio_instance = None
        self._stream = None

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Open the microphone and yield audio chunks.

        Each chunk is:
            - Exactly chunk_size samples
            - float32 dtype
            - Mono (1 channel)
            - At sample_rate Hz (resampling happens in audio/resampler.py
              if the mic doesn't support the target rate natively)
        """
        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "PyAudio is not installed. Run: pip install pyaudio\n"
                "On Linux you may also need: sudo apt-get install portaudio19-dev\n"
                "On Mac: brew install portaudio"
            )

        cfg = self.settings.audio

        logger.info(
            f"Opening microphone | "
            f"device_index={cfg.input_device_index} "
            f"sample_rate={cfg.sample_rate}Hz "
            f"chunk_size={cfg.chunk_size} samples "
            f"chunk_duration={cfg.chunk_duration_ms}ms"
        )

        # PyAudio expects int16 from most hardware — we'll convert to float32
        # after reading. Some devices support float32 natively, but int16
        # is universally supported and is what real microphones give you.
        pa_format = pyaudio.paInt16

        self._pyaudio_instance = pyaudio.PyAudio()

        # Log available input devices — critical for debugging
        # "Why isn't it using my USB mic?" is a very common support question
        self._log_available_devices(self._pyaudio_instance)

        try:
            self._stream = self._pyaudio_instance.open(
                format=pa_format,
                channels=cfg.channels,
                rate=cfg.sample_rate,
                input=True,
                input_device_index=cfg.input_device_index,  # None = default
                frames_per_buffer=cfg.chunk_size,
            )

            logger.info("Microphone stream opened successfully. Listening...")

            while not self._stop_event.is_set():
                chunk = self._read_chunk()

                if chunk is None:
                    # _read_chunk returns None on recoverable errors
                    # (e.g., brief buffer overflow). Log and continue.
                    continue

                yield chunk

        except OSError as e:
            # OSError covers: device not found, permission denied,
            # device already in use, sample rate not supported
            logger.error(
                f"Microphone error: {e}\n"
                f"Check: is a microphone connected? "
                f"Is another application using it? "
                f"Does the device support {cfg.sample_rate}Hz?"
            )
            raise

        finally:
            # ALWAYS clean up hardware, even if an exception occurred
            # 'finally' runs whether we exited normally or via exception
            self._cleanup()

    def _read_chunk(self) -> Optional[np.ndarray]:
        """
        Read exactly one chunk from the microphone.

        Returns:
            Validated float32 numpy array, or None on recoverable error.

        Why is this a separate method?
            Separation of concerns. stream() handles the loop and lifecycle.
            _read_chunk handles the single-chunk read logic.
            This makes each piece independently testable.
        """
        cfg = self.settings.audio

        try:
            # exception_on_overflow=False: if the system is too slow to
            # read and the hardware buffer overflows, log a warning instead
            # of raising an exception. Buffer overflows mean we DROPPED audio.
            # We log it so you know, but we don't crash.
            raw_bytes = self._stream.read(
                cfg.chunk_size,
                exception_on_overflow=False,
            )
        except OSError as e:
            logger.warning(f"Failed to read audio chunk: {e}")
            return None

        # Convert raw bytes → numpy int16 → float32
        # np.frombuffer interprets raw bytes as an array of the given dtype
        # This is a zero-copy operation — no memory allocation
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = convert_to_float32(audio_int16)

        # Handle stereo hardware even if we asked for mono
        # Some drivers ignore the channel count request
        if audio_float32.ndim == 2 or len(audio_float32) == cfg.chunk_size * 2:
            if len(audio_float32) == cfg.chunk_size * 2:
                # Interleaved stereo: [L0, R0, L1, R1, ...]
                audio_float32 = audio_float32.reshape(-1, 2)
            audio_float32 = stereo_to_mono(audio_float32)

        # Validate before yielding — this is the trust boundary
        try:
            validate_audio_chunk(
                audio_float32,
                expected_samples=cfg.chunk_size,
                caller="MicrophoneCapture"
            )
        except ValueError as e:
            logger.warning(f"Invalid chunk from microphone: {e}")
            return None

        # Log signal level at DEBUG — too verbose for INFO but
        # essential when debugging "why isn't VAD detecting speech"
        rms = compute_rms(audio_float32)
        peak = compute_peak(audio_float32)

        logger.debug(f"Mic chunk | rms={rms:.4f} peak={peak:.4f}")

        # Warn if signal is clipping — hardware input gain too high
        if peak > 0.95:
            logger.warning(
                f"Audio clipping detected! peak={peak:.4f}. "
                f"Lower your microphone input gain."
            )

        return audio_float32

    def _log_available_devices(self, pa: "pyaudio.PyAudio") -> None:
        """
        Log all available audio input devices.

        This runs once at startup. When a user says "it's not using
        my USB mic," you look at these logs and immediately see
        what device index to set in settings.
        """
        logger.info("Available audio input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                logger.info(
                    f"  [{i}] {info['name']} | "
                    f"channels={info['maxInputChannels']} | "
                    f"default_rate={info['defaultSampleRate']:.0f}Hz"
                )

    def _cleanup(self) -> None:
        """Release hardware resources. Safe to call multiple times."""
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
                logger.debug("Microphone stream closed")
            except Exception as e:
                logger.warning(f"Error closing mic stream: {e}")
            self._stream = None

        if self._pyaudio_instance is not None:
            try:
                self._pyaudio_instance.terminate()
                logger.debug("PyAudio instance terminated")
            except Exception as e:
                logger.warning(f"Error terminating PyAudio: {e}")
            self._pyaudio_instance = None


# ─────────────────────────────────────────────────────────────────
# FILE CAPTURE — for testing and offline processing
# ─────────────────────────────────────────────────────────────────

class FileCapture(AudioSource):
    """
    Streams audio from a WAV file, chunked to match pipeline expectations.

    This is invaluable for:
        1. Testing the pipeline without a microphone
        2. Reproducible debugging (same file = same results every time)
        3. Benchmarking (measure latency against known audio)
        4. CI/CD pipelines (automated tests can't use a real mic)

    Two streaming modes:
        realtime=True:  Sleep between chunks to simulate real microphone timing.
                        Use this to test the full realtime pipeline behavior.
        realtime=False: Stream as fast as possible.
                        Use this for batch processing or fast testing.

    Important: FileCapture resamples if the WAV file's sample rate
    doesn't match settings.audio.sample_rate. This handles real-world
    audio files which might be at 44100Hz, 48000Hz, etc.
    """

    def __init__(self, settings: Settings, file_path: str, realtime: bool = True):
        super().__init__(settings)
        self.file_path = Path(file_path)
        self.realtime = realtime

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {self.file_path}"
            )

        if self.file_path.suffix.lower() != ".wav":
            raise ValueError(
                f"Only WAV files are supported. Got: {self.file_path.suffix}\n"
                f"Convert with: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav"
            )

    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Read the WAV file in chunks and yield them.

        Handles:
            - Any WAV sample rate (resamples to target if needed)
            - Mono or stereo WAV files
            - int16, int32, or float32 WAV encoding
            - Files shorter than one chunk (yields partial + silence pad)
            - Realtime pacing (sleep between chunks to simulate live audio)
        """
        cfg = self.settings.audio

        logger.info(
            f"Opening audio file | path={self.file_path} "
            f"realtime={self.realtime}"
        )

        with wave.open(str(self.file_path), "rb") as wav_file:
            file_info = self._log_file_info(wav_file)
            file_sample_rate = wav_file.getframerate()
            file_channels = wav_file.getnchannels()
            file_sampwidth = wav_file.getsampwidth()  # bytes per sample
            total_frames = wav_file.getnframes()

            # How many bytes is one chunk from this file?
            # chunk_size is in samples at TARGET sample rate.
            # If file is at a different rate, we need to read proportionally more.
            if file_sample_rate != cfg.sample_rate:
                # Read enough frames from the file to cover one target chunk
                # after resampling.
                # e.g., target=16kHz, file=44100Hz, chunk=480 samples
                # We need: 480 * (44100/16000) = 1323 frames from file
                frames_per_chunk = int(
                    cfg.chunk_size * file_sample_rate / cfg.sample_rate
                )
                needs_resample = True
            else:
                frames_per_chunk = cfg.chunk_size
                needs_resample = False

            bytes_per_frame = file_sampwidth * file_channels
            bytes_per_chunk = frames_per_chunk * bytes_per_frame

            # How long should we sleep between chunks to simulate realtime?
            # At 16kHz with 480 samples: 480/16000 = 0.030 seconds = 30ms
            chunk_duration_s = cfg.chunk_size / cfg.sample_rate

            frames_read = 0
            chunk_count = 0

            while not self._stop_event.is_set():
                raw_bytes = wav_file.readframes(frames_per_chunk)

                # End of file
                if len(raw_bytes) == 0:
                    logger.info(
                        f"End of audio file | "
                        f"chunks_yielded={chunk_count} "
                        f"total_frames={frames_read}"
                    )
                    break

                # Convert raw bytes to numpy array based on sample width
                audio = self._bytes_to_array(raw_bytes, file_sampwidth)

                # Handle stereo → mono
                if file_channels == 2:
                    audio = audio.reshape(-1, 2)
                    audio = stereo_to_mono(audio)

                # Resample if necessary
                if needs_resample:
                    audio = self._simple_resample(
                        audio, file_sample_rate, cfg.sample_rate
                    )

                # Convert to float32
                audio = convert_to_float32(audio)

                # Handle final chunk that might be shorter than chunk_size
                if len(audio) < cfg.chunk_size:
                    audio = self._pad_to_chunk_size(audio, cfg.chunk_size)
                elif len(audio) > cfg.chunk_size:
                    # Resampling can occasionally produce one extra sample
                    audio = audio[:cfg.chunk_size]

                # Final validation
                try:
                    validate_audio_chunk(
                        audio,
                        expected_samples=cfg.chunk_size,
                        caller="FileCapture"
                    )
                except ValueError as e:
                    logger.warning(f"Skipping invalid chunk from file: {e}")
                    continue

                frames_read += frames_per_chunk
                chunk_count += 1

                logger.debug(
                    f"File chunk {chunk_count} | "
                    f"rms={compute_rms(audio):.4f} "
                    f"frames_read={frames_read}/{total_frames}"
                )

                yield audio

                # Pace output to simulate realtime streaming
                # Without this, we'd stream the whole file in milliseconds
                # and the pipeline wouldn't behave like a real microphone
                if self.realtime:
                    time.sleep(chunk_duration_s)

    def _bytes_to_array(self, raw_bytes: bytes, sampwidth: int) -> np.ndarray:
        """
        Convert WAV raw bytes to numpy array with correct dtype.

        WAV files store samples as integers. The sample width (bytes per sample)
        determines the range:
            1 byte (uint8):  0 to 255
            2 bytes (int16): -32768 to 32767  ← most common
            4 bytes (int32): -2147483648 to 2147483647
        """
        if sampwidth == 1:
            return np.frombuffer(raw_bytes, dtype=np.uint8)
        elif sampwidth == 2:
            return np.frombuffer(raw_bytes, dtype=np.int16)
        elif sampwidth == 4:
            return np.frombuffer(raw_bytes, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    def _simple_resample(
        self,
        audio: np.ndarray,
        from_rate: int,
        to_rate: int,
    ) -> np.ndarray:
        """
        Simple linear interpolation resampling.

        This is NOT production-quality resampling.
        Production systems use scipy.signal.resample or soxr
        which use high-quality anti-aliasing filters.

        We use linear interpolation here because:
            1. It has zero dependencies
            2. It's easy to understand
            3. For speech recognition, quality doesn't need to be perfect
            4. We'll mention the production alternative

        Production alternative:
            import soxr
            audio = soxr.resample(audio, from_rate, to_rate)

        Args:
            audio: Input audio at from_rate.
            from_rate: Source sample rate.
            to_rate: Target sample rate.

        Returns:
            Resampled audio at to_rate.
        """
        if from_rate == to_rate:
            return audio

        # Number of samples in the output
        output_length = int(len(audio) * to_rate / from_rate)

        # Create evenly-spaced indices in the INPUT array
        # corresponding to each OUTPUT sample position
        input_indices = np.linspace(0, len(audio) - 1, output_length)

        # Linear interpolation: for each output sample, interpolate
        # between the two nearest input samples
        resampled = np.interp(
            input_indices,
            np.arange(len(audio)),
            audio.astype(np.float64),
        ).astype(np.float32)

        logger.debug(
            f"Resampled | {from_rate}Hz→{to_rate}Hz | "
            f"{len(audio)}→{len(resampled)} samples"
        )

        return resampled

    def _pad_to_chunk_size(
        self, audio: np.ndarray, target_size: int
    ) -> np.ndarray:
        """
        Pad a short final chunk with silence to reach target_size.

        The last chunk of a file is almost never exactly chunk_size samples.
        We pad it so downstream modules always receive exactly what they expect.
        """
        pad_length = target_size - len(audio)
        silence = np.zeros(pad_length, dtype=np.float32)
        return np.concatenate([audio, silence])

    def _log_file_info(self, wav_file: wave.Wave_read) -> dict:
        """Log WAV file metadata — essential for debugging."""
        info = {
            "sample_rate": wav_file.getframerate(),
            "channels": wav_file.getnchannels(),
            "sample_width_bytes": wav_file.getsampwidth(),
            "total_frames": wav_file.getnframes(),
            "duration_s": wav_file.getnframes() / wav_file.getframerate(),
        }
        logger.info(
            f"WAV file info | "
            f"rate={info['sample_rate']}Hz "
            f"channels={info['channels']} "
            f"depth={info['sample_width_bytes'] * 8}bit "
            f"duration={info['duration_s']:.2f}s"
        )
        return info