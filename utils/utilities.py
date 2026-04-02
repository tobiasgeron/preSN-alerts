"""
Generic helpers: logging setup, paths, and log-level parsing.

This module centralizes configuration so the main script can focus on domain logic.
All user-visible diagnostics from the CLI entry point should go through the
application logger configured by :func:`configure_application_logging`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

APP_LOGGER_NAME = "pre_sn_alerts"

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Create the parent directory for a file path if it does not exist.

    Parameters
    ----------
    path : str or pathlib.Path
        File path whose parent directory should exist.

    Returns
    -------
    pathlib.Path
        Resolved ``Path`` for ``path``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def parse_loglevel_name(name: str) -> int:
    """
    Map a string such as ``\"INFO\"`` to a ``logging`` level constant.

    Parameters
    ----------
    name : str
        Level name (case-insensitive), e.g. ``DEBUG``, ``INFO``, ``WARNING``.

    Returns
    -------
    int
        Numeric level (e.g. ``logging.INFO``).

    Raises
    ------
    ValueError
        If ``name`` is not a valid level.
    """
    level = getattr(logging, str(name).upper(), None)
    if not isinstance(level, int):
        raise ValueError(
            f"Invalid log level {name!r}; use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )
    return level


class _StreamToLogger(TextIO):
    """
    File-like object that forwards each line to a ``logging.Logger``.

    Used so third-party code writing to a stream (e.g. progress text) can be
    captured in the log file.
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            text = line.strip()
            if text and not text.startswith("\r"):
                self._logger.log(self._level, text)
        return len(s)

    def flush(self) -> None:
        text = self._buf.strip()
        if text:
            self._logger.log(self._level, text)
        self._buf = ""


def configure_application_logging(
    log_file: str | Path,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> logging.Logger:
    """
    Configure the application logger with a file handler and a console handler.

    All messages at or above ``file_level`` are written to ``log_file``.
    All messages at or above ``console_level`` are echoed to stderr.

    Parameters
    ----------
    log_file : str or pathlib.Path
        Path to the log file (parent directories are created).
    console_level : int, optional
        Minimum level for the stderr :class:`logging.StreamHandler`.
    file_level : int, optional
        Minimum level for the :class:`logging.FileHandler`.
    fmt : str, optional
        ``logging.Formatter`` format string.
    datefmt : str, optional
        ``strftime``-style date format for the formatter.

    Returns
    -------
    logging.Logger
        The named application logger (``APP_LOGGER_NAME``).

    Notes
    -----
    Existing handlers on this logger are removed to avoid duplicate lines when
    re-running in the same interpreter (e.g. tests).
    """
    ensure_parent_dir(log_file)
    log = logging.getLogger(APP_LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(ch)

    log.debug(
        "Logging initialized: file=%r file_level=%s console_level=%s",
        str(Path(log_file).resolve()),
        logging.getLevelName(file_level),
        logging.getLevelName(console_level),
    )
    return log


def get_app_logger() -> logging.Logger:
    """
    Return the application logger (may have no handlers until configured).

    Returns
    -------
    logging.Logger
        Logger named :data:`APP_LOGGER_NAME`.
    """
    return logging.getLogger(APP_LOGGER_NAME)


def tqdm_log_stream(level: int = logging.INFO) -> TextIO:
    """
    Build a text stream suitable for ``tqdm(..., file=...)`` that logs lines.

    Parameters
    ----------
    level : int, optional
        Log level for forwarded lines.

    Returns
    -------
    TextIO
        Writable stream; flush periodically so partial lines are logged.
    """
    return _StreamToLogger(get_app_logger(), level)
