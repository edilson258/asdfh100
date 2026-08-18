"""Logging initialization module integrating Loguru with Rich terminal rendering."""

import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.traceback import install as rich_traceback_install

console = Console()
rich_traceback_install(show_locals=False)


def setup_console_handler(verbose: bool) -> None:
    """Attach stderr console log sink formatted for Rich display.

    Example:
        >>> setup_console_handler(verbose=True)
    """
    log_level = "DEBUG" if verbose else "INFO"
    log_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=log_level, colorize=True, format=log_format)


def setup_file_handler(log_dir: Path) -> None:
    """Attach daily rotating file log sink with 14-day retention.

    Example:
        >>> setup_file_handler(Path("./logs"))
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_pattern = str(log_dir / "pipeline_{time:YYYY-MM-DD}.log")
    logger.add(
        log_file_pattern,
        level="DEBUG",
        rotation="00:00",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


def initialize_pipeline_logger(verbose: bool, log_dir: Path) -> None:
    """Configure Loguru loggers for console and file sinks.

    Example:
        >>> initialize_pipeline_logger(verbose=False, log_dir=Path("./logs"))
    """
    logger.remove()
    setup_console_handler(verbose=verbose)
    setup_file_handler(log_dir=log_dir)
