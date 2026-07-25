# src/audio/stream_source.py

import time
import logging
from typing import Generator
from config.settings import AudioSettings

logger = logging.getLogger("StreamSource")

class SimulatedMicStream:
    """
    Simulates a live microphone stream by generating silent audio chunks in real-time.
    This provides an infrastructure harness that runs reliably on any machine without
    requiring physical microphone access or hardware drivers.
    """
    def __init__(self, settings: AudioSettings):
        self.settings = settings
        self.is_running = False
        
        # Pre-generate one static chunk of silence (zeros in bytes)
        self._silence_chunk = b"\x00" * self.settings.chunk_bytes

    def start(self) -> None:
        """Starts the audio generation lifecycle."""
        self.is_running = True
        logger.info("Simulated microphone stream initialized.")

    def stop(self) -> None:
        """Stops the audio generation lifecycle cleanly."""
        self.is_running = False
        logger.info("Simulated microphone stream stopped.")

    def yield_chunks(self) -> Generator[bytes, None, None]:
        """
        Generates stream chunks at a real-time cadence.
        
        Yields:
            bytes: A raw audio frame matching the configured chunk size.
        """
        if not self.is_running:
            raise RuntimeError("Stream must be started before yielding chunks.")
            
        interval = self.settings.CHUNK_DURATION_MS / 1000.0
        
        while self.is_running:
            start_time = time.time()
            
            yield self._silence_chunk
            
            # Real-time pacing: sleep to simulate the elapsed real-world time
            elapsed = time.time() - start_time
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)