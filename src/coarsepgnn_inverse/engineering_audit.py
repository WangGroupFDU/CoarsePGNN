"""审计专用真实批次接线与独立回读；禁止优化器更新或性能评价。"""
from __future__ import annotations

import argparse
import collections
import json
import platform
import time
from pathlib import Path

from .access import load_development
from .development import build_development_loaders
from .graph import recipe_to_complete_graph
from .io import read_json, sha256, write_json


def run(directory: Path) -> dict:
    """独立读取审核产物，检查谱系、访问隔离和实际图前向；返回工程事实而非性能。

    输入为一次新审计目录。随机初始化结构模型只进行无梯度前向，权重不复用。
    测试标签路径被拒绝访问代理拦截；正负对照分别计数并保留回执。
    """
    import torch
    import rdkit
    import torch_geometric
    from coarsepgnn.data import GraphFeaturizer
    from coarsepgnn.schema import package_schema
    from coarsepgnn_hybrid_v3.model import StructureOnlyRegressor
    from coarsepgnn_hybrid_v3.train import _move

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(20261111)
    manifest = read_json(directory/"split_labels_manifest.json")
    test_path = Path(manifest["test"]["path"])
    attempted = []

    def guarded_reader(path: Path):
        """记录真实I/O请求；测试文件即使存在也禁止打开。"""
        attempted.append(str(path))
        if path.resolve() == test_path.resolve():
            raise PermissionError("最终测试标签访问被审计代理拒绝")
        return read_json(path)

    loaded = load_development(manifest, guarded_reader)
    finite = {}
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("本轮指定cn2 CUDA，当前不可用")
    batch_times = {}
    for target in ("tg", "tm", "tensile_strength_ultimate", "elongation_at_break"):
        wired = build_development_loaders(manifest, target, reader=guarded_reader)
        cpu_batch = next(iter(wired["loaders"]["train"]))
        batch = _move(cpu_batch, device)
        model = StructureOnlyRegressor(target=target, schema=package_schema()).to(device).eval()
        with torch.no_grad():
            model(batch["graph"])
            torch.cuda.synchronize()
            begin = time.monotonic()
            prediction = model(batch["graph"])
            torch.cuda.synchronize()
        batch_times[target] = {"forward_seconds": time.monotonic()-begin,
                               "batch_size": int(prediction.numel()),
                               "train_observations": len(wired["loaders"]["train"].dataset)}
        finite[target] = bool(torch.isfinite(prediction).all())
        if not finite[target]:
            raise ValueError("随机初始化真实批次前向出现非有限值")
        del model, wired, batch
    normal_test_attempts = sum(p == str(test_path) for p in attempted)
    refused = False
    try:
        guarded_reader(test_path)
    except PermissionError:
        refused = True
    if normal_test_attempts != 0 or not refused:
        raise AssertionError("开发测试访问隔离未成立")
    # 从真实来源图而非合成夹具重建D/P/E配方，逐张量核对训练兼容性。
    formulations = {f["material_id"]: f for f in read_json(directory/"approved_formulations.json")}
    sample = next(r for r in loaded["train"] if r["material_id"] in formulations)
    raw = read_json(Path(sample["graph_relpath"]))
    formula = formulations[sample["material_id"]]
    rebuilt = recipe_to_complete_graph(raw["components"], formula["nco_oh_ratio"], formula["extender_oh_fraction"])
    featurizer = GraphFeaturizer.from_sample_schema(package_schema())
    before = featurizer.prepare_graph(raw).data.to_dict()
    after = featurizer.prepare_graph(rebuilt).data.to_dict()
    mismatched = [key for key, value in before.items() if torch.is_tensor(value) and
                  (key not in after or not torch.equal(value, after[key]))]
    graph_result = {"material_id": sample["material_id"], "mismatched_tensor_fields": mismatched,
                    "exact_model_input_replay": not mismatched}
    if mismatched:
        raise ValueError(f"配方构图与来源张量未一致: {mismatched}")
    counts = collections.Counter(r["restored_from_mean"] for r in loaded["train"] if r.get("restored_from_mean"))
    report = {"status":"ENGINEERING_VERIFIED_NO_TRAINING", "formal_training_started":False,
              "optimizer_updates":0, "normal_development_test_read_attempts":normal_test_attempts,
              "negative_control_test_read_refused":refused, "finite_forward_by_task":finite,
              "graph_replay":graph_result, "restored_training_parent_count":len(counts),
              "batch_forward_timing":batch_times, "elapsed_seconds":time.monotonic()-started,
              "environment":{"python":platform.python_version(),"torch":torch.__version__,
                 "pyg":torch_geometric.__version__,"rdkit":rdkit.__version__,
                 "device":torch.cuda.get_device_name(0)},
              "runtime_limit":"仅实测无梯度前向。反向/优化器、搜索和全三种子耗时尚未实测；不能据此保证72小时可完成。"}
    write_json(directory/"test_access_guard_results.json",report)
    write_json(directory/"checkpoint_lineage.json", {
        "old_supervised_weights_used":False,
        "must_retrain":["four_structure_predictors","four_condition_residual_heads",
                        "supervised_condition_projection","PLE_train_statistics","calibration"],
        "reason":"旧监督资产无法证明与新闭包留出集合隔离；冻结权重并不能消除泄漏。",
        "reusable":["deterministic_graph_fingerprints","fixed_unsupervised_text_backbone_after_version_and_prompt_check"]})
    write_json(directory/"runtime_estimate.json", {"observed":batch_times,
        "full_runtime_known":False,"reserved_budget_seconds":259200,
        "pending":"冻结合同后的首个真实epoch测量训练吞吐；不能用本轮前向计时伪称完整预算已验证。"})
    write_json(directory/"split_access_policy.json", {"development_entry":"coarsepgnn_inverse.development.build_development_loaders",
        "legacy_train_fold_called":False,"development_label_splits":["train","validation"],
        "final_entry":"coarsepgnn_inverse.access.load_final_test",
        "final_window_opened":False,"training_entry_complete":False})
    write_json(directory/"engineering_manifest.json", {p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}
                for p in sorted(directory.iterdir()) if p.is_file() and p.name not in {"test_labels.json","validation_labels.json"}})
    return report


def main() -> None:
    """只接受一次已完成的审计目录；输出回执已存在时不覆盖。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.directory),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
