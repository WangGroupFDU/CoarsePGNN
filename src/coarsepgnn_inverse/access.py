"""研究专用的标签访问边界；开发入口没有测试数据的加载分支。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .io import read_json, sha256


def load_development(manifest: dict, reader: Callable = read_json) -> dict[str, list[dict]]:
    """仅加载固定 train/validation 文件并验证身份，不访问 test 文件或其内容。

    manifest 的两个开发条目必须有 path/sha256；reader 参数用于真实拒绝访问测试。
    文件变动、空集合、错误集合及重复观察均明确失败，避免默默重新划分。
    """
    result = {}
    seen = set()
    # 仅读取封存路径元数据；在任何hash/JSON内容读取之前检查两个开发路径。
    sealed = manifest.get("test", {}).get("path")
    if not sealed:
        raise ValueError("开发清单必须声明封存测试路径元数据")
    sealed_path = Path(sealed).resolve()
    paths = {s: Path(manifest[s]["path"]).resolve() for s in ("train", "validation")}
    if sealed_path in paths.values():
        raise PermissionError("开发条目误指向封存测试文件，在散列读取前拒绝")
    if paths["train"] == paths["validation"]:
        raise ValueError("训练和验证路径必须互相独立")
    for split in ("train", "validation"):
        entry = manifest[split]
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"开发标签文件散列不符: {split}")
        rows = reader(path)
        if not rows or any(row["split"] != split for row in rows):
            raise ValueError(f"开发集合为空或包含错误归属: {split}")
        for row in rows:
            if row["observation_id"] in seen:
                raise ValueError("开发集合存在重复观察")
            seen.add(row["observation_id"])
        result[split] = rows
    return result


def load_final_test(manifest: dict, lock: dict, contract_path: Path,
                    frozen_artifacts: dict[str, Path]) -> list[dict]:
    """独立最终入口验证冻结合同及全部锁定对象后，才读取测试标签。

    lock 必须标明一次性最终评价窗口，并列出实际合同、代码清单、策略和三种子权重散列。
    调用方负责在不可覆盖的结果目录登记窗口；函数不启动训练或放宽失败条件。
    """
    contract = read_json(contract_path)
    if contract.get("status") != "FROZEN" or not contract.get("frozen"):
        raise ValueError("最终评价要求已冻结成功合同")
    if lock.get("status") != "FINAL_EVALUATION_LOCKED":
        raise ValueError("最终测试窗口未锁定")
    if sha256(contract_path) != lock["contract_sha256"]:
        raise ValueError("冻结合同散列不一致")
    required = {"code_manifest", "data_manifest", "candidate_policy", "baseline_policy",
                "scorer", "checkpoint_20261111", "checkpoint_20261112", "checkpoint_20261113"}
    if set(frozen_artifacts) != required or set(lock["artifacts"]) != required:
        raise ValueError("最终锁缺少必需版本对象")
    for key, path in frozen_artifacts.items():
        if sha256(path) != lock["artifacts"][key]:
            raise ValueError(f"最终冻结对象变动: {key}")
    entry = manifest["test"]
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("最终测试文件散列不一致")
    rows = read_json(path)
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError("最终测试为空或归属错误")
    return rows
