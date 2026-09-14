"""Chain100 harness 的文件、哈希和运行时审计工具。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CSV_ENCODING = "utf-8-sig"
CSV_LINE_TERMINATOR = "\r\n"


def jsonable(value: Any) -> Any:
    """递归转换路径、数组标量和非有限值，保证审计 JSON 可序列化。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    """以稳定顺序编码对象，用于适配输入哈希。"""

    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    """原子写入 UTF-8 JSON，避免中断时留下半成品。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _csv_value(value: Any) -> Any:
    """把嵌套值和换行稳定转换为 RFC 4180 字段。"""

    if isinstance(value, (dict, list, tuple)):
        value = canonical_json(value)
    value = jsonable(value)
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> int:
    """写入 UTF-8 BOM、逗号分隔、CRLF 的 RFC 4180 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator=CSV_LINE_TERMINATOR,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
            count += 1
    os.replace(temporary, path)
    return count


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带 BOM 的 CSV，并保留字段文本精度。"""

    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """计算 UTF-8 文本的 SHA-256。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finite_float(value: Any) -> float | None:
    """解析有限浮点；空值、非法值和非有限值返回 ``None``。"""

    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def command_output(command: Sequence[str]) -> str:
    """采集版本命令输出；失败时保留可审计错误文本。"""

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"
    return result.stdout.strip()


def runtime_snapshot() -> dict[str, Any]:
    """记录主机、解释器和关键包版本。"""

    packages: dict[str, str] = {}
    for name in ("numpy", "rdkit", "torch", "torch_geometric", "scipy"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "available"))
        except Exception as error:
            packages[name] = f"ERROR: {type(error).__name__}: {error}"
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "packages": packages,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
    }


def finite_array(values: Any) -> bool:
    """判断数组或张量是否全部有限。"""

    try:
        import numpy as np

        return bool(np.isfinite(np.asarray(values)).all())
    except Exception:
        return False
