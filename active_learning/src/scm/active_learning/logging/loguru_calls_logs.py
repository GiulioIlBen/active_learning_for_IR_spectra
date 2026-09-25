from __future__ import annotations

from pathlib import Path
from typing import Any, TextIO

from loguru import logger
from typing_extensions import Literal, TypeAlias

LoguruNamedLevel: TypeAlias = Literal[
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
]
LoguruLevel: TypeAlias = LoguruNamedLevel | int
FormatOptions: TypeAlias = str | None | Literal["verbose", "simple"]


def log_level(message: Any, level: LoguruLevel = "INFO", depth: int = 1, **kwargs: Any) -> None:
    logger.opt(depth=depth).log(level, message, **kwargs)


def remove_sinks(handler_id: int | None = None) -> None:
    logger.remove(handler_id=handler_id)


def add_sink(
    file_path: str | Path | TextIO, level: LoguruLevel = "INFO", format: FormatOptions = None, **kwargs: Any
) -> int:
    if format == "verbose":
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name:<60}</cyan>:<cyan>{function:<20}</cyan>:<cyan>{line:<4}</cyan> - "
            "<level>{message}</level>"
        )
    elif format == "simple":
        format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    return logger.add(sink=file_path, level=level, format=format, **kwargs)
