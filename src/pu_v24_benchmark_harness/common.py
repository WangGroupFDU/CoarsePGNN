"""PU V2.4 统一训练流程共享的稳定文件格式、目标变换与指标函数。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BASE_SEED = 20260901
FOLD_COUNT = 5
PRIMARY_STRATEGY = "record_random_interpolation"
GROUPED_STRATEGY = "formulation_graph_grouped"

COARSE_MODELS = ("CoarsePGNN-structure-only", "CoarsePGNN-FULL")
ALL_MODELS = COARSE_MODELS

TARGETS: dict[str, dict[str, Any]] = {
    "tg": {
        "source_count": 803,
        "unit": "°C",
        "kelvin_offset": 273.15,
        "transform": "standardize",
        "display_name_zh": "玻璃化转变温度",
    },
    "tm": {
        "source_count": 954,
        "unit": "°C",
        "kelvin_offset": 273.15,
        "transform": "standardize",
        "display_name_zh": "熔融/软化转变温度",
    },
    "elongation_at_break": {
        "source_count": 1032,
        "unit": "%",
        "kelvin_offset": None,
        "transform": "log1p",
        "display_name_zh": "断裂伸长率",
    },
    "tensile_strength_ultimate": {
        "source_count": 868,
        "unit": "MPa",
        "kelvin_offset": None,
        "transform": "log1p",
        "display_name_zh": "极限拉伸强度",
    },
}

CSV_ENCODING = "utf-8-sig"
CSV_LINE_TERMINATOR = "\r\n"


def canonical_json(value: Any) -> str:
    """将 JSON 对象稳定序列化，作为哈希与缓存身份的唯一文本表示。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """计算字节串 SHA-256。"""

    return hashlib.sha256(value).hexdigest()


def jsonable(value: Any) -> Any:
    """递归转换路径和 NumPy 标量，并把非有限浮点数变为 JSON null。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    """通过同目录临时文件原子写入稳定 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    """把嵌套字段稳定序列化，交由 csv 模块执行 RFC 4180 转义。"""

    if isinstance(value, (dict, list, tuple)):
        value = canonical_json(value)
    value = jsonable(value)
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> int:
    """写 UTF-8 BOM、逗号分隔、CRLF 的 RFC 4180 CSV，并返回数据行数。"""

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
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
            count += 1
    os.replace(temporary, path)
    return count


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 BOM CSV，返回字符串字典行。"""

    with path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """以稳定 JSON Lines 原子写入缓存输入，并返回记录数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(jsonable(row)) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSON Lines；空行被忽略，坏行保留明确解析异常。"""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def command_output(command: Sequence[str]) -> str:
    """采集只读版本命令输出，失败时写入可审计错误而不打断主流程。"""

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"
    return result.stdout.strip()


def runtime_snapshot() -> dict[str, Any]:
    """记录主机、Python 和 Git 状态，供结果包追溯。"""

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
    }


def parse_json_field(value: Any, default: Any = None) -> Any:
    """解析数据库 JSON 字段；缺失返回默认值，畸形文本抛出 ValueError。"""

    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON 字段无法解析: {value!r}") from error


@dataclass(frozen=True)
class TargetTransform:
    """仅由训练集拟合的标签变换；力学目标在反变换时执行非负约束。"""

    kind: str
    mean: float
    std: float

    @classmethod
    def fit(cls, target: str, values: Sequence[float]) -> "TargetTransform":
        """根据目标和训练标签拟合变换参数，不读取验证或测试统计量。"""

        if target not in TARGETS:
            raise KeyError(f"未知目标: {target}")
        array = np.asarray(values, dtype=np.float64)
        if len(array) < 2 or not np.isfinite(array).all():
            raise ValueError(f"{target} 训练标签不足或包含非有限值")
        kind = str(TARGETS[target]["transform"])
        if kind == "log1p":
            if np.any(array < 0):
                raise ValueError(f"{target} 不能对负值执行 log1p")
            return cls(kind=kind, mean=0.0, std=1.0)
        std = float(array.std(ddof=0))
        return cls(kind=kind, mean=float(array.mean()), std=std if std > 0 else 1.0)

    def forward(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """将原始单位标签变换到模型训练空间。"""

        array = np.asarray(values, dtype=np.float64)
        if self.kind == "log1p":
            if np.any(array < 0):
                raise ValueError("力学目标不能对负值执行 log1p")
            return np.log1p(array)
        return (array - self.mean) / self.std

    def inverse(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """将模型输出恢复到原始单位；力学预测显式限制为非负。"""

        array = np.asarray(values, dtype=np.float64)
        if self.kind == "log1p":
            return np.expm1(np.maximum(array, 0.0))
        return array * self.std + self.mean

    def as_dict(self) -> dict[str, float | str]:
        """返回可写入训练配置的稳定参数字典。"""

        return {"kind": self.kind, "mean": self.mean, "std": self.std}


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    """计算 MAE、RMSE、R² 和 Spearman；负 R² 被原样保留。"""

    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError(f"预测与真值形状不一致: {truth.shape} vs {pred.shape}")
    if len(truth) == 0 or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("指标输入为空或包含非有限值")
    error = pred - truth
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    try:
        from scipy.stats import spearmanr

        rho = float(spearmanr(truth, pred).statistic)
    except Exception:
        rho = float("nan")
    return {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "spearman": rho if math.isfinite(rho) else None,
    }


def refresh_hash_manifest(root: Path, output_name: str = "sha256_manifest.csv") -> Path:
    """为结果目录内全部普通文件生成稳定哈希清单，清单本身不自包含。"""

    output = root / output_name
    rows = [
        {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output
    ]
    write_csv(output, rows, ["relative_path", "size_bytes", "sha256"])
    return output
