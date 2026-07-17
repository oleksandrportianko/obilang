"""Structured JSON logging for CLI, training runs, and the monitoring UI."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format standard log records as one machine-readable JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON line containing context attached through ``extra``."""
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        for field in ("experiment_id", "model_version", "epoch", "step", "context"):
            value = getattr(record, field, None)
            if value is not None:
                event[field] = value
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False, sort_keys=True)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure deterministic console and optional file logging.

    Args:
        level: Standard logging level name.
        log_file: Optional JSON Lines log destination.

    Returns:
        None. Existing root handlers are replaced to avoid duplicate events.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    formatter = JsonFormatter()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
