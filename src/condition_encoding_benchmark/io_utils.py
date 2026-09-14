"""稳定文件读写、哈希和运行环境记录工具。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def utc_now() -> str:
    """返回秒精度、带 Z 后缀的 UTC 时间。"""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(value: Any) -> str:
    """生成稳定紧凑 JSON，用于哈希和 JSONL。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    """计算字节串 SHA-256。"""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，避免把模型文件整体载入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """原子写入带末尾换行的 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """原子写入稳定 UTF-8 JSONL，返回记录数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并拒绝空行及非对象记录。"""

    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"JSONL 存在空行: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL 行不是对象: {path}:{line_number}")
            output.append(value)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """按 RFC 4180、UTF-8 BOM 和 CRLF 原子写入 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\r\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 RFC 4180 CSV，并保留所有字段为字符串。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def runtime_snapshot() -> dict[str, Any]:
    """记录不改变环境的 Python、平台和关键依赖版本。"""

    versions: dict[str, str] = {}
    for name in ("numpy", "torch", "transformers", "pandas", "sklearn"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not_installed"
    return {
        "created_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "versions": versions,
    }


def file_hash_rows(root: Path, *, excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
    """生成目录内普通文件的稳定哈希清单。"""

    excluded_set = set(excluded)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set or relative.startswith("."):
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_hash_manifest(root: Path) -> Path:
    """写入不自我包含的 SHA-256 CSV 清单。"""

    output = root / "SHA256SUMS.csv"
    rows = file_hash_rows(root, excluded=("SHA256SUMS.csv",))
    write_csv(output, rows, ("relative_path", "size_bytes", "sha256"))
    return output
