"""File contracts shared by the public release commands."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document; propagate missing-file and parsing errors."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write a finite, indented JSON document, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a comma-separated UTF-8 table while accepting its optional BOM."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write RFC 4180 CSV with a UTF-8 BOM and CRLF, including embedded newlines."""
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write an empty table without a schema: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        for row in rows:
            normalized = {key: value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
                          if isinstance(value, str) else value for key, value in row.items()}
            writer.writerow(normalized)


def sha256(path: Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_samples(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines and require nonempty, unique sample identifiers."""
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [row.get("sample_id") for row in rows]
    if not rows or any(not isinstance(s, str) or not s for s in ids) or len(ids) != len(set(ids)):
        raise ValueError("Input must contain unique, nonempty string sample_id values.")
    return rows


def require_new(path: Path) -> None:
    """Reject an existing output so a command cannot replace earlier results."""
    if path.exists():
        raise FileExistsError(f"Output exists; choose a new path: {path}")


def schema_path() -> Path:
    """Locate the graph schema shipped beside the installed model package."""
    import coarsepgnn
    return Path(coarsepgnn.__file__).resolve().parent / "sample_schema.json"
