"""单一配置、单一预算的一键训练流水线；折外评价与全量部署明确分离。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sources():
    # 从实际导入包定位源码；wheel 的 site-packages 不能反推成 repo/src。
    package = Path(__file__).resolve().parent
    repository_layout = package.parent.name == "src"
    root = package.parent.parent if repository_layout else package.parent
    paths = sorted(package.rglob("*.py"))
    if not paths or Path(__file__).resolve() not in paths:
        raise RuntimeError("cannot bind the actual imported Python package sources")
    if repository_layout:
        script = root / "scripts/train_pu_all_data.py"
        if script.is_file():
            paths.append(script)
    return {path.relative_to(root).as_posix(): _sha(path.read_bytes()) for path in paths}


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("shared workflow deadline exceeded")
    return seconds


def _configure_runtime(device, threads):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no available GPU; CPU fallback is forbidden")
    torch.set_num_threads(threads)


def _worker(output_dir, deadline):
    """独立进程只执行锁定计划；任一步失败立即终止，不重试或换协议。"""
    output = Path(output_dir)
    lock = json.loads((output / "workflow_lock.json").read_text())
    if lock["deadline_monotonic"] != deadline:
        raise ValueError("worker deadline differs from the immutable workflow lock")
    started = time.monotonic()
    completed, active = [], "input_verification"

    def event(kind, **fields):
        with (output / "workflow_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "utc": _utc(), "monotonic": time.monotonic(),
                                     "pid": os.getpid(), **fields}, ensure_ascii=False, allow_nan=False) + "\n")

    def execute(name, operation):
        nonlocal active
        active = name
        if _sources() != lock["source_sha256"]:
            raise ValueError("source changed after the workflow lock")
        remaining = _remaining(deadline)
        event("stage_started", stage=name, remaining_seconds=remaining)
        stage_started = time.monotonic()
        result = operation(remaining)
        _remaining(deadline)
        if isinstance(result, dict) and result.get("status", "completed") != "completed":
            raise RuntimeError(f"stage {name} did not complete")
        receipt = {"stage": name, "status": "completed", "pid": os.getpid(),
                   "elapsed_seconds": time.monotonic() - stage_started}
        completed.append(receipt)
        event("stage_completed", **receipt)
        return result

    try:
        _write(output / "workflow_worker_start.json", {"pid": os.getpid(), "parent_pid": os.getppid(),
                    "started_utc": _utc(), "started_monotonic": started, "deadline_monotonic": deadline})
        _remaining(deadline)
        dataset_bytes = (output / "inputs/dataset.jsonl").read_bytes()
        config_bytes = (output / "inputs/config.json").read_bytes()
        if _sha(dataset_bytes) != lock["dataset_file_sha256"] or _sha(config_bytes) != lock["config_file_sha256"]:
            raise ValueError("locked input snapshot SHA mismatch")
        rows = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines() if line.strip()]
        config = json.loads(config_bytes)
        if not rows:
            raise ValueError("empty dataset")
        from .champion import FORWARD_FAMILY_POLICY, select_forward_champion
        from .baseline import run_baselines, refit_full_baseline
        from .features import MAIN_TARGETS, FoldFeatures
        from .splits import make_nested_splits
        from .training import run_nested, refit_full, runtime_provenance

        if config.get("forward_family_selection") != FORWARD_FAMILY_POLICY:
            raise ValueError("forward family selection policy must be declared before training")
        seeds = config.get("seeds", [])
        if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("configuration must declare unique integer seeds")
        _configure_runtime(lock["device"], lock["threads"])
        _write(output / "runtime.json", runtime_provenance())
        _write(output / "dataset_identity.json", {
            "dataset_sha256": _sha(json.dumps(rows, sort_keys=True).encode()),
            "retained_observations": len(rows),
            "supervised_observations": sum(row.get("shared_trainable", False) for row in rows),
        })
        tasks = list(MAIN_TARGETS) + sorted({row["target"] for row in rows if row.get("shared_trainable")} - set(MAIN_TARGETS))
        features = lambda: FoldFeatures(tasks=tasks)
        splits = execute("split", lambda _: make_nested_splits(
            rows, outer_folds=config.get("outer_folds", 5), inner_folds=config.get("inner_folds", 3), seed=seeds[0]))
        _write(output / "splits.json", splits)
        execute("baselines", lambda budget: run_baselines(rows, splits, features, output / "evaluation/baselines",
            seed=seeds[0], candidates=config.get("baseline_candidates"), max_seconds=budget))
        execute("neural_nested", lambda budget: run_nested(rows, splits, config, output / "evaluation/neural",
            device=lock["device"], max_seconds=budget))
        # 家族选择不拟合；父监督器仍对其汇总/自举阶段执行同一硬截止。
        execute("forward_champion", lambda _: select_forward_champion(rows, splits,
            output / "evaluation/baselines", output / "evaluation/neural", output / "evaluation/champion",
            bootstrap_replicates=config.get("bootstrap_replicates", 2000)))
        if not lock["evaluation_only"]:
            execute("neural_full_refit", lambda budget: refit_full(rows, splits, config,
                output / "evaluation/neural", output / "deployment/neural", device=lock["device"], max_seconds=budget))
            execute("baseline_full_refit", lambda budget: refit_full_baseline(rows, splits,
                output / "evaluation/baselines", output / "deployment/baseline", seed=seeds[0],
                feature_factory=features, max_seconds=budget))
        result = {"status": "completed", "pid": os.getpid(), "stages": completed,
                  "retained_observations": len(rows), "supervised_observations": sum(row["shared_trainable"] for row in rows),
                  "evaluation_only": lock["evaluation_only"], "evaluation_scope": "retrospective_nested_grouped_oof",
                  "deployment_scope": "not_run" if lock["evaluation_only"] else "full_data_refit_without_independent_test",
                  "new_blind_test_claim": False, "elapsed_seconds": time.monotonic() - started}
        _write(output / "workflow_worker_result.json", result)
        event("worker_completed")
        return result
    except BaseException as error:
        _write(output / "workflow_worker_failure.json", {"status": "failed", "stage": active,
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "completed_stages": completed, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started})
        event("worker_failed", stage=active, error_type=type(error).__name__, error=str(error))
        raise


def run_workflow(dataset, configuration, output_dir, *, device="cpu", threads=1,
                 max_seconds=14400, evaluation_only=False):
    """监督独立 worker，全部步骤共享一个硬截止；不复用或覆盖既有输出目录。"""
    if device not in {"cpu", "cuda"} or type(threads) is not int or threads < 1:
        raise ValueError("device must be cpu/cuda and threads must be a positive integer")
    if isinstance(max_seconds, bool) or not math.isfinite(float(max_seconds)) or float(max_seconds) <= 0:
        raise ValueError("max_seconds must be finite and positive")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + float(max_seconds)
    process, result = None, None
    audit = {"supervisor_pid": os.getpid(), "started_utc": _utc(), "started_monotonic": started,
             "deadline_monotonic": deadline, "max_seconds": float(max_seconds),
             "timeout_scope": "input locking, splitting, evaluation, selection and deployment share one deadline",
             "retries": 0, "automatic_protocol_changes": 0}
    try:
        (output / "inputs").mkdir()
        dataset_bytes = Path(dataset).read_bytes()
        config_bytes = Path(configuration).read_bytes()
        (output / "inputs/dataset.jsonl").write_bytes(dataset_bytes)
        (output / "inputs/config.json").write_bytes(config_bytes)
        lock = {"schema_version": "pu-all-data-workflow-v1", "created_utc": _utc(),
                "dataset_file_sha256": _sha(dataset_bytes), "config_file_sha256": _sha(config_bytes),
                "dataset_original_path": str(Path(dataset).resolve()), "config_original_path": str(Path(configuration).resolve()),
                "source_sha256": _sources(), "configuration": json.loads(config_bytes),
                "device": device, "threads": threads, "evaluation_only": bool(evaluation_only),
                "deadline_monotonic": deadline, "max_seconds": float(max_seconds),
                "new_blind_test_claim": False, "deployment_has_independent_test": False}
        _write(output / "workflow_lock.json", lock)
        _write(output / "workflow_supervisor_start.json", audit)
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = str(threads)
        source_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        command = [sys.executable, "-m", "coarsepgnn_retrain.workflow", "--worker", str(output), "--deadline", str(deadline)]
        audit["command"] = command
        _remaining(deadline)
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            audit["worker_pid"] = process.pid
            try:
                returncode = process.wait(timeout=_remaining(deadline))
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("shared workflow hard deadline exceeded") from error
        audit["worker_returncode"] = returncode
        _remaining(deadline)
        if returncode != 0:
            raise RuntimeError(f"workflow worker failed with return code {returncode}; see worker failure and log")
        result = json.loads((output / "workflow_worker_result.json").read_text())
        if result.get("status") != "completed":
            raise RuntimeError("workflow worker did not report completion")
        audit["status"] = "completed"
    except BaseException as error:
        audit["status"] = "failed"
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                  "existing_outputs_preserved": True, "automatic_retry": False, "new_blind_test_claim": False}
        _write(output / "workflow_failure.json", {**result, "traceback": traceback.format_exc()})
        raise
    finally:
        # 终止整个进程组；清理宽限期只用于回收，不授权继续科学计算。
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            audit["worker_returncode"] = process.returncode
            audit["worker_terminated"] = True
        audit["ended_utc"] = _utc()
        audit["elapsed_seconds"] = time.monotonic() - started
        _write(output / "workflow_audit.json", audit)
        _write(output / "workflow_result.json", {**result, "output_dir": str(output),
                    "wall_seconds": audit["elapsed_seconds"], "supervisor_pid": os.getpid(),
                    "worker_pid": audit.get("worker_pid")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train all traceable PU observations with one grouped nested protocol and one total budget")
    parser.add_argument("--dataset")
    parser.add_argument("--config", default="configs/retrain/pu_all_data_v2.json")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=14400)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.deadline is None:
            parser.error("worker requires a locked absolute deadline")
        _worker(args.worker, args.deadline)
    else:
        if not args.dataset or not args.output:
            parser.error("--dataset and --output are required")
        result = run_workflow(args.dataset, args.config, args.output, device=args.device, threads=args.threads,
                              max_seconds=args.max_seconds, evaluation_only=args.evaluation_only)
        print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
