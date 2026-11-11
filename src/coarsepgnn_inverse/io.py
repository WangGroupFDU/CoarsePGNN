"""输入边界、稳定散列及符合项目约定的结果文件。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    """将 JSON 兼容输入稳定序列化用于身份散列；拒绝 NaN 和无穷大。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    """返回规范 JSON 的 SHA-256；键顺序不影响散列，数值不做舍入。"""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    """流式读取文件并计算 SHA-256；文件不可读时保留原 I/O 错误。"""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON；拒绝非标准常量，避免将坏标签传入训练。"""
    def invalid(value: str) -> None:
        """把非标准 JSON 常量转换为可追溯的输入错误。"""
        raise ValueError(f"{path}: 非法 JSON 常量 {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)


def write_json(path: Path, value: Any) -> None:
    """独占写入新 JSON 产物；拒绝覆盖历史结果，先校验数据再创建文件。"""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """以 UTF-8 BOM、逗号及 CRLF 独占导出 CSV；空表仍写入明确列名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)

