# tests/test_logging_config.py

"""
Testing the logging configuration.

What we're testing:
    1. setup_logging() runs without errors
    2. get_logger() returns a properly named logger
    3. TimingLogger measures time and logs correctly
    4. TimingLogger handles exceptions gracefully

What failures would mean:
    - If get_logger() returns an unnamed logger, all log lines say "root"
      and you can't tell which module produced them.
    - If TimingLogger swallows exceptions, bugs disappear silently.
    - If setup_logging() fails, the whole system starts without logging
      and you're debugging blind.

Engineering note on testing logging:
    We don't test the exact log output format — that's fragile and couples
    tests to presentation details. We test BEHAVIOR:
    - Did a log record get produced?
    - Is it at the right level?
    - Does it have the right name?
"""

import logging
import pytest
import time
from utils.logging_config import setup_logging, get_logger, TimingLogger


class TestSetupLogging:

    def test_setup_runs_without_error(self):
        """Basic smoke test — setup should not raise."""
        setup_logging(console_level=logging.WARNING)  # Quiet during tests

    def test_setup_clears_existing_handlers(self):
        """
        Calling setup twice shouldn't double up log output.
        This matters for tests that call setup_logging multiple times.
        """
        setup_logging(console_level=logging.WARNING)
        root = logging.getLogger()
        handler_count_after_first = len(root.handlers)

        setup_logging(console_level=logging.WARNING)
        handler_count_after_second = len(root.handlers)

        assert handler_count_after_first == handler_count_after_second

    def test_log_to_file_creates_file(self, tmp_path):
        """
        When log_to_file=True, a log file should be created.
        tmp_path is a pytest fixture that gives us a temporary directory.
        """
        setup_logging(
            console_level=logging.WARNING,
            log_to_file=True,
            log_dir=str(tmp_path / "logs")
        )
        log_files = list((tmp_path / "logs").glob("*.log"))
        assert len(log_files) == 1, f"Expected 1 log file, got {len(log_files)}"


class TestGetLogger:

    def test_logger_name_matches_module(self):
        """
        The logger name should be exactly what we pass in.
        When modules pass __name__, this becomes their dotted path.
        """
        logger = get_logger("vad.engine")
        assert logger.name == "vad.engine"

    def test_logger_is_logging_logger_instance(self):
        """Should return a standard Python logger, not some custom object."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_different_names_give_different_loggers(self):
        """Each module should have its own logger — not sharing state."""
        logger_a = get_logger("module.a")
        logger_b = get_logger("module.b")
        assert logger_a is not logger_b
        assert logger_a.name != logger_b.name

    def test_same_name_gives_same_logger(self):
        """
        Python loggers are cached by name — same name = same object.
        This is important: if two files in the same module call
        get_logger(__name__), they should get the SAME logger.
        """
        logger_1 = get_logger("vad.engine")
        logger_2 = get_logger("vad.engine")
        assert logger_1 is logger_2


class TestTimingLogger:

    def setup_method(self):
        """Run before each test — set up fresh logging."""
        setup_logging(console_level=logging.WARNING)

    def test_timing_logger_measures_time(self):
        """
        The timing logger should produce a log record when the block exits.
        We capture log records using a custom handler.
        """
        captured_records = []

        # A simple handler that captures log records instead of printing them
        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)

        logger = get_logger("test.timing")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)

        try:
            sleep_ms = 50
            with TimingLogger(logger, "test operation", level=logging.DEBUG):
                time.sleep(sleep_ms / 1000)

            assert len(captured_records) == 1
            message = captured_records[0].getMessage()
            assert "test operation" in message
            assert "completed" in message
            assert "duration_ms" in message
        finally:
            logger.removeHandler(handler)

    def test_timing_logger_does_not_swallow_exceptions(self):
        """
        CRITICAL: If TimingLogger swallows exceptions, bugs disappear silently.
        The context manager must let exceptions propagate.
        """
        logger = get_logger("test.timing.exceptions")

        with pytest.raises(ValueError, match="intentional test error"):
            with TimingLogger(logger, "failing operation"):
                raise ValueError("intentional test error")

    def test_timing_logger_logs_failure_on_exception(self):
        """
        When an exception occurs, TimingLogger should log an error
        (not a success), then let the exception propagate.
        """
        captured_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)

        logger = get_logger("test.timing.failure")
        logger.setLevel(logging.DEBUG)
        handler = CapturingHandler()
        logger.addHandler(handler)

        try:
            with pytest.raises(RuntimeError):
                with TimingLogger(logger, "bad operation"):
                    raise RuntimeError("boom")

            assert len(captured_records) == 1
            assert captured_records[0].levelno == logging.ERROR
            assert "FAILED" in captured_records[0].getMessage()
        finally:
            logger.removeHandler(handler)