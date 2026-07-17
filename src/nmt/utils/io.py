"""Safe local filesystem helpers used by all artifact stores."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a file with bytes.

    Args:
        path: Destination inside a platform-managed artifact directory.
        content: Complete new file contents.

    Returns:
        None. The destination is durable when the function returns.

    Raises:
        OSError: If the parent cannot be created, data cannot be flushed, or the
            operating system cannot atomically replace the destination.

    Side effects:
        Creates parent directories and a short-lived temporary file beside the
        destination. Keeping both files on one filesystem makes replacement atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    """Serialize JSON deterministically and atomically replace ``path``.

    Args:
        path: JSON destination path.
        value: JSON-serializable value.

    Returns:
        None.

    Raises:
        TypeError: If ``value`` cannot be represented as JSON.
        OSError: If the destination cannot be written.
    """
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, encoded + b"\n")


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a large file to a temporary sibling and atomically replace its anchor.

    Args:
        source: Complete source artifact, such as a just-fsynced periodic checkpoint.
        destination: Named anchor such as `latest.pt`.

    Returns:
        None. The old destination remains valid until replacement completes.

    Raises:
        OSError: If source reading, destination writing, fsync, or replacement fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as source_file, os.fdopen(descriptor, "wb") as destination_file:
            shutil.copyfileobj(source_file, destination_file, length=1024 * 1024)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_json(path: Path, default: Any | None = None) -> Any:
    """Load a UTF-8 JSON document, optionally returning a missing-file default.

    Args:
        path: JSON file to read.
        default: Value returned only when the path does not exist.

    Returns:
        Parsed JSON content.

    Raises:
        UnicodeDecodeError: If the file is not UTF-8.
        json.JSONDecodeError: If the document is malformed.
    """
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one structured event to a UTF-8 JSON Lines file.

    Args:
        path: Event stream destination.
        value: JSON object for one line.

    Returns:
        None.

    Side effects:
        Creates the parent directory and appends to the destination. Callers that
        share a file across processes must provide their own lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
