# utils/logging_config.py

"""
Centralized logging configuration for the realtime ASR system.

Engineering principle: Configure logging ONCE at startup, use it everywhere.

Why centralize?
    If every module sets up its own logging configuration, they'll conflict.
    You'll get duplicate messages, inconsistent formats, and chaos.
    One place sets the rules. Every module just says "give me a logger" and uses it.

Usage in any module:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"key": "value"})
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# LOG FORMAT
# ─────────────────────────────────────────────────────────────────

# This is what every log line looks like.
# %(name)-20s means: the logger name, padded to 20 chars (so columns align).
# Aligned columns make logs scannable at a glance.

CONSOLE_FORMAT = (
    "%(asctime)s.%(msecs)03d | "   # Timestamp with milliseconds
    "%(levelname)-8s | "           # Level (padded to 8 chars)
    "%(name)-25s | "               # Module name (padded to 25 chars)
    "%(message)s"                  # The actual message
)

# File logs get more detail: thread name helps debug multithreading issues
FILE_FORMAT = (
    "%(asctime)s.%(msecs)03d | "
    "%(levelname)-8s | "
    "%(name)-25s | "
    "%(threadName)-15s | "         # Which thread produced this log
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ─────────────────────────────────────────────────────────────────
# SETUP FUNCTION — called once at startup in main.py
# ─────────────────────────────────────────────────────────────────

def setup_logging(
    console_level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str = "logs",
) -> None:
    """
    Configure the root logger for the entire application.

    Call this ONCE at the very start of main.py, before anything else.
    After this runs, every module that calls get_logger() will
    automatically inherit this configuration.

    Args:
        console_level: Minimum severity to print to terminal.
            logging.DEBUG    → see everything (noisy but useful for dev)
            logging.INFO     → see important events only
            logging.WARNING  → see only problems
        log_to_file: Whether to also write logs to a file.
            Useful for post-mortem debugging of realtime sessions.
        log_dir: Directory for log files. Created if it doesn't exist.

    Why configure the ROOT logger?
        Python logging is hierarchical. The root logger is the ancestor
        of all loggers. Setting it up here means every child logger
        (vad.engine, asr.buffer, etc.) inherits the handlers and format.
        We can still override individual loggers later if needed.
    """
    # Get the root logger — this is the parent of ALL loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Root accepts everything; handlers filter

    # Remove any existing handlers (important if setup is called multiple times,
    # e.g., during testing)
    root_logger.handlers.clear()

    # ── Console Handler ──────────────────────────────────────────
    # This is what you see in your terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    )
    root_logger.addHandler(console_handler)

    # ── File Handler (optional) ──────────────────────────────────
    # Writes to a timestamped file so you can review sessions later
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"asr_session_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File always gets everything
        file_handler.setFormatter(
            logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
        )
        root_logger.addHandler(file_handler)

        # Log the file location so you can find it later
        root_logger.info(f"Logging to file: {log_file}")

    # Silence noisy third-party libraries
    # These libraries log at DEBUG constantly and would drown out your logs
    _silence_noisy_libraries()


def _silence_noisy_libraries() -> None:
    """
    Third-party libraries are often very chatty at DEBUG level.
    We silence them here so our logs stay readable.

    This is standard practice. You'll do this in every production system.
    """
    noisy_libraries = [
        "urllib3",          # HTTP library — logs every request/response
        "httpx",            # Another HTTP library
        "httpcore",
        "multipart",
        "filelock",         # Model download locking
        "transformers",     # Hugging Face — logs model loading details
        "torch",            # PyTorch — logs CUDA initialization
        "numba",            # JIT compiler — logs compilation steps
    ]
    for library_name in noisy_libraries:
        logging.getLogger(library_name).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────
# GET LOGGER — call this in every module
# ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Call at the TOP of every module file, like this:
        logger = get_logger(__name__)

    Why __name__?
        Python sets __name__ to the module's full dotted path automatically.
        In vad/engine.py, __name__ is "vad.engine".
        In asr/buffer.py, __name__ is "asr.buffer".
        This gives every logger a unique, meaningful name for free.

    Args:
        name: The logger name. Always pass __name__.

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────
# PERFORMANCE LOGGING HELPER
# ─────────────────────────────────────────────────────────────────

class TimingLogger:
    """
    Context manager for measuring and logging how long something takes.

    In realtime systems, latency is everything.
    You need to know: how long does ASR take per utterance?
    How long does VAD take per chunk?

    Usage:
        with TimingLogger(logger, "ASR transcription", level=logging.INFO):
            result = model.transcribe(audio)
        # Automatically logs: "ASR transcription completed in 342.1ms"

    Why a context manager?
        It guarantees the timing is logged even if an exception occurs.
        The 'with' block handles setup and teardown automatically.
        It's cleaner than try/finally everywhere.
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation_name: str,
        level: int = logging.DEBUG,
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.level = level
        self._start_time: float = 0.0

    def __enter__(self) -> "TimingLogger":
        import time
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import time
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000

        if exc_type is not None:
            # An exception occurred inside the 'with' block
            self.logger.error(
                f"{self.operation_name} FAILED after {elapsed_ms:.1f}ms | "
                f"error={exc_type.__name__}: {exc_val}"
            )
        else:
            self.logger.log(
                self.level,
                f"{self.operation_name} completed | duration_ms={elapsed_ms:.1f}"
            )

        # Return False: don't suppress the exception if one occurred
        return False