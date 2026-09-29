"""One structured logging surface for rp1 and its dependencies."""

from __future__ import annotations

import contextlib
import io
import logging as stdlib_logging
import sys
import traceback
import warnings
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

from loguru import logger


class _StructuredSink:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self.lock = Lock()

    def __call__(self, message: Any) -> None:
        record = message.record
        lines = str(record["message"]).splitlines() or [""]
        exception = record["exception"]
        if exception is not None:
            lines.extend(
                line
                for line in "".join(traceback.format_exception(exception.type, exception.value, exception.traceback))
                .rstrip()
                .splitlines()
            )
        self.write_lines(record["level"].name, lines, timestamp=record["time"])

    def write_lines(self, level: str, lines: list[str], *, timestamp: Any | None = None) -> None:
        moment = timestamp or datetime.now().astimezone()
        prefix = f"{moment.strftime('%Y-%m-%d %H:%M:%S')} [{level}] "
        with self.lock:
            for line in lines:
                self.stream.write(prefix + line + "\n")
            self.stream.flush()


class _InterceptHandler(stdlib_logging.Handler):
    def emit(self, record: stdlib_logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(exception=record.exc_info, depth=6).log(level, record.getMessage())


class _StreamLogger(io.TextIOBase):
    def __init__(self, level: str, original: TextIO, sinks: tuple[_StructuredSink, ...]):
        self.level = level
        self.original = original
        self.sinks = sinks
        self._buffer = ""

    def write(self, value: str) -> int:
        self._buffer += value
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._emit(line.rstrip())
        return len(value)

    def flush(self) -> None:
        if self._buffer.strip():
            self._emit(self._buffer.rstrip())
        self._buffer = ""

    def _emit(self, line: str) -> None:
        # Write directly to the configured destinations instead of calling
        # Loguru.  Dependencies are allowed to use sys.stdout as a Loguru sink;
        # forwarding that write back into Loguru would make the handler invoke
        # itself and trigger Loguru's deadlock protection.
        for sink in self.sinks:
            sink.write_lines(self.level, [line])

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self.original.fileno()


@contextlib.contextmanager
def configured_logging(log_path: str | Path, level: str) -> Iterator[Any]:
    """Log to the terminal and file while normalizing Python output streams."""
    logger.remove()
    terminal = sys.__stderr__ or sys.stderr
    original_stdout = sys.__stdout__ or sys.stdout
    with Path(log_path).open("a", buffering=1) as log_file:
        terminal_sink = _StructuredSink(terminal)
        file_sink = _StructuredSink(log_file)
        logger.add(terminal_sink, level=level, format="{message}", catch=True)
        logger.add(file_sink, level=level, format="{message}", catch=True)

        root = stdlib_logging.getLogger()
        previous_handlers = root.handlers[:]
        previous_level = root.level
        root.handlers = [_InterceptHandler()]
        root.setLevel(level)

        sinks = (terminal_sink, file_sink)
        stdout = _StreamLogger("INFO", original_stdout, sinks)
        stderr = _StreamLogger("ERROR", terminal, sinks)
        previous_showwarning = warnings.showwarning

        def showwarning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: TextIO | None = None,
            line: str | None = None,
        ) -> None:
            del file
            lines = warnings.formatwarning(message, category, filename, lineno, line).rstrip().splitlines()
            for sink in sinks:
                sink.write_lines("WARNING", lines)

        warnings.showwarning = showwarning  # ty: ignore[invalid-assignment]  # Typeshed models equivalent warning hooks as distinct callables.
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                yield logger
        finally:
            warnings.showwarning = previous_showwarning
            stdout.flush()
            stderr.flush()
            root.handlers = previous_handlers
            root.setLevel(previous_level)
            # A dependency may have replaced the handlers after this context
            # started (stable-pretraining does this at import time), so the
            # original numeric IDs may no longer exist.
            logger.remove()


__all__ = ["configured_logging", "logger"]
