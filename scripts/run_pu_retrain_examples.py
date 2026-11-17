#!/usr/bin/env python3
"""对完成的全量部署模型运行三个示例；只推理、保留原输出、不报告逆向命中率。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


SEED = 20261117
EXAMPLES = {"forward": "forward_request.json", "inverse_single": "inverse_single_request.json",
            "inverse_multiple": "inverse_multiple_request.json"}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("bound file path escapes its run or repository")
    return path


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def verify_model_files(directory, family, expected_ids):
    """只校验字节和 JSON 元数据；在全部清单通过之前不反序列化权重。"""
    directory = Path(directory)
    if family not in {"classical", "neural"}:
        raise ValueError("unknown deployment model family")
    name = "manifest.json" if family == "classical" else "checkpoint_manifest.json"
    manifest = _json(directory / name)
    if family == "classical":
        required = {"selection_lock.json", "features.json", "models.joblib", "audit.json"}
        if manifest.get("schema_version") != "pu-baseline-deployment-files-v1":
            raise ValueError("incorrect classical deployment manifest schema")
        entries = manifest["files"]
        if len(entries) != len(required) or {item["path"] for item in entries} != required:
            raise ValueError("incomplete classical model manifest")
    else:
        required = {"forward.pt", "inverse.pt", "features.json"}
        if set(manifest["files"]) != required:
            raise ValueError("incomplete neural checkpoint manifest")
        entries = [{"path": key, "sha256": value} for key, value in manifest["files"].items()]
    files = {}
    for item in entries:
        path = _inside(directory, item["path"])
        if _sha(path) != item["sha256"] or ("bytes" in item and path.stat().st_size != item["bytes"]):
            raise ValueError("deployment model SHA/size mismatch: " + item["path"])
        files[item["path"]] = {"sha256": item["sha256"], "bytes": path.stat().st_size}
    wrapper = _json(directory / "features.json")
    state = wrapper["state"] if family == "classical" else wrapper
    ids = state["fit_observation_ids"]
    identity = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(expected_ids) or state["fit_ids_sha256"] != identity:
        raise ValueError("deployment model does not bind all shared training observations")
    if family == "classical":
        audit = _json(directory / "audit.json")
        if audit.get("status") != "completed" or audit.get("seed") != SEED:
            raise ValueError("classical deployment audit is not complete")
        fit_receipt = "audit.json"
    else:
        fit_receipt = "fit_result.json"
        audit = _json(directory / fit_receipt)
        if (audit.get("status") != "completed" or audit.get("seed") != SEED
                or audit.get("validation_observations") != 0 or audit.get("training_observations") != len(ids)
                or audit.get("shared_training_ids_sha256") != identity):
            raise ValueError("neural deployment receipt does not match the fixed seed/full-data refit")
    return {"directory": str(directory.resolve()), "family": family, "manifest_sha256": _sha(directory / name),
            "fit_receipt_sha256": _sha(directory / fit_receipt),
            "files": files, "fit_ids_sha256": identity, "shared_training_observations": len(ids)}


def verify_output(kind, request, result, *, count):
    """仅核对调用、数值与组成接口；这些检查不构成化学或实验验证。"""
    if result.get("fitting_performed") is not False:
        raise ValueError("inference output must explicitly exclude fitting")
    if kind == "forward":
        requests = request if isinstance(request, list) else [request]
        if any(set(row) & {"value", "y_true", "observed_value", "target_value", "targets"} for row in requests):
            raise ValueError("forward example must not include observed labels")
        outputs = result["rows"]
        if len(outputs) != len(requests) or result.get("observed_labels_required") is not False:
            raise ValueError("forward output changed request coverage")
        for index, (source, row) in enumerate(zip(requests, outputs)):
            if row["request_id"] != source.get("request_id", f"query_{index}") or row["target"] != source["target"]:
                raise ValueError("forward output request identity mismatch")
            if row["status"] not in {"predicted", "unsupported"}:
                raise ValueError("unknown forward status")
            if (row["status"] == "predicted" and not _finite(row["prediction"])) or (
                    row["status"] == "unsupported" and row["prediction"] is not None):
                raise ValueError("forward status and prediction disagree")
            if row.get("experimental_success") is not None or row.get("chemical_validation") != "not_performed":
                raise ValueError("forward smoke example must not claim experimental success")
        return {"request_count": len(outputs), "statuses": dict(Counter(row["status"] for row in outputs))}
    expected_scope = "single_target_query" if len(request["targets"]) == 1 else "compositional_query_unvalidated"
    if (result["scope"] != expected_scope or result["joint_target_training_verified"] is not False
            or result["targets"] != request["targets"] or result["seed"] != SEED
            or result["request_id"] != request.get("request_id")):
        raise ValueError("inverse request identity, scope or fixed seed mismatch")
    limits = {"D": [1, 3], "P": [1, 3], "E": [0, 3], "OTHER": [0, 0]}
    limits.update(request.get("role_limits", {}))
    if result["role_limits"] != limits or result["conditions"] != request.get("conditions", {}):
        raise ValueError("inverse output changed the requested conditions or role constraints")
    if any(result[key] != count for key in ("requested_count", "candidate_count", "proposal_budget_consumed")):
        raise ValueError("raw proposal budget changed")
    if (result["hidden_retries"] != 0 or result["forward_ranking_performed"] is not False
            or result["chemical_validation"] != "not_performed" or result["experimental_success"] is not None
            or result["ratio_basis"] != "molecular_mole_fraction" or len(result["candidates"]) != count):
        raise ValueError("inverse output overstates validation or changes the proposal budget")
    for index, candidate in enumerate(result["candidates"]):
        if (candidate["proposal_index"] != index or candidate["chemical_validation"] != "not_performed"
                or candidate["experimental_success"] is not None):
            raise ValueError("candidate identity or validation status mismatch")
        components = candidate["components"]
        ratios = [item["molecule_ratio"] for item in components]
        if not ratios or any(not _finite(value) or value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1., abs_tol=1e-6):
            raise ValueError("candidate mole fractions must be finite, positive and sum to one")
        roles = Counter(item["role"] for item in components)
        if set(roles) - {"D", "P", "E", "OTHER"} or roles["D"] < 1 or roles["P"] < 1:
            raise ValueError("candidate role composition is invalid")
        for role, (lower, upper) in result["role_limits"].items():
            if not lower <= roles[role] <= upper:
                raise ValueError("candidate violates a requested role-count limit")
        for item in components:
            if item.get("ratio_mask") is not True or item.get("ratio_origin") != "model_sample" or not item.get("catalog_key"):
                raise ValueError("candidate lacks catalog/continuous-ratio provenance")
        if candidate["extender_state"] != ("PRESENT" if roles["E"] else "NONE"):
            raise ValueError("candidate extender state disagrees with its components")
    return {"candidate_count": count, "scope": expected_scope,
            "positive_normalized_composition": True, "chemical_validation": "not_performed",
            "experimental_success": None, "hit_rate_computed": False}


def _preflight(run, repository):
    workflow = run / "workflow"
    if _json(workflow / "workflow_result.json")["status"] != "completed":
        raise ValueError("examples require a fully completed workflow")
    lock = _json(workflow / "workflow_lock.json")
    if not lock["source_sha256"]:
        raise ValueError("workflow must bind its actual source files")
    for name, expected in lock["source_sha256"].items():
        if _sha(_inside(repository, name)) != expected:
            raise ValueError("frozen inference package source changed: " + name)
    config = _json(workflow / "inputs/config.json")
    if _sha(workflow / "inputs/config.json") != lock["config_file_sha256"]:
        raise ValueError("workflow configuration SHA mismatch")
    route_path = run / "delivery_summary/deployment_route.json"
    route = _json(route_path)
    if not route["seeds"] or route["seeds"][0] != SEED or route["seeds"] != config["seeds"] or route["outer_metrics_used"] is not False:
        raise ValueError("deployment route must retain the predeclared first seed and inner-only selection")
    neural_root, classical_root = workflow / "deployment/neural", workflow / "deployment/baseline"
    nlock, block = _json(neural_root / "selection_lock.json"), _json(classical_root / "selection_lock.json")
    expected_selection = {"neural_selection": _sha(neural_root / "selection_lock.json"),
                          "classical_selection": _sha(classical_root / "selection_lock.json")}
    if route["source_sha256"] != expected_selection:
        raise ValueError("deployment route selection SHA mismatch")
    nscore = nlock["inner_scores"]["forward"][nlock["chosen"]["forward"]]
    selected = min(block["candidate_scores"], key=lambda item: (item["mean_inner_score"], item["order"]))
    expected_family = "classical" if selected["mean_inner_score"] <= nscore else "neural"
    if (route["selected_forward_family"] != expected_family or route["classical_candidate"] != selected["candidate"]
            or route["neural_forward_candidate"] != nlock["chosen"]["forward"]
            or route["neural_inverse_candidate"] != nlock["chosen"]["inverse"]):
        raise ValueError("deployment route differs from the recorded inner-only rule")
    expected_paths = {direction: neural_root / f"seed_{SEED}_trial_{nlock['chosen'][direction]}"
                      for direction in ("forward", "inverse")}
    if (_inside(run, route["classical_path"]) != classical_root
            or any(_inside(run, route[f"neural_{direction}_paths"][str(SEED)]) != expected_paths[direction]
                   for direction in expected_paths)):
        raise ValueError("deployment route references a different seed or a nondeployment model")
    audit = _json(run / "dataset/audit.json")
    if audit["dataset_sha256"] != lock["dataset_file_sha256"]:
        raise ValueError("dataset audit does not describe the workflow input bytes")
    if audit["forward_observation_ids"] != audit["inverse_observation_ids"]:
        raise ValueError("shared supervision identities differ")
    expected_ids = audit["forward_observation_ids"]
    forward = classical_root if expected_family == "classical" else expected_paths["forward"]
    bindings = {"forward": verify_model_files(forward, expected_family, expected_ids),
                "inverse": verify_model_files(expected_paths["inverse"], "neural", expected_ids)}
    return route, bindings, {"workflow_lock": _sha(workflow / "workflow_lock.json"),
                            "workflow_result": _sha(workflow / "workflow_result.json"), "deployment_route": _sha(route_path)}, expected_ids


def _worker(plan_path, case_name, expected_plan_sha):
    if _sha(plan_path) != expected_plan_sha:
        raise ValueError("example plan SHA changed before worker startup")
    plan = _json(plan_path)
    case = plan["cases"][case_name]
    directory = Path(plan_path).parent / case_name
    binding = plan["models"]["forward" if case_name == "forward" else "inverse"]
    if verify_model_files(binding["directory"], binding["family"], plan["shared_observation_ids"]) != binding:
        raise ValueError("model bytes changed before inference worker startup")
    if _sha(directory / "input.json") != case["input_sha256"]:
        raise ValueError("example input changed after plan lock")
    guard_script = Path(__file__).resolve().parent / "verify_pu_retrain_oof.py"
    if _sha(__file__) != plan["script_sha256"] or _sha(guard_script) != plan["guard_script_sha256"]:
        raise ValueError("example wrapper or no-training guard changed after plan lock")
    workflow_lock = Path(plan["run"]) / "workflow/workflow_lock.json"
    if _sha(workflow_lock) != plan["source_sha256"]["workflow_lock"]:
        raise ValueError("workflow lock changed after example plan creation")
    for name, expected in _json(workflow_lock)["source_sha256"].items():
        if _sha(_inside(Path(__file__).resolve().parents[1], name)) != expected:
            raise ValueError("frozen source changed before inference: " + name)
    # 复用包外核验工具的禁止拟合守卫，所有模型读取与 CLI 调用都在其作用域内。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_pu_retrain_oof import _no_training
    from coarsepgnn_retrain.cli import main as cli_main
    started = time.monotonic()
    receipt = {"pid": os.getpid(), "parent_pid": os.getppid(), "started_utc": _utc(),
               "case": case_name, "cli_arguments": case["cli_arguments"]}
    try:
        with _no_training(directory / "no_training_audit.json"):
            cli_main(case["cli_arguments"])
        result = _json(directory / "output.json")
        validation = verify_output(case_name, _json(directory / "input.json"), result, count=plan["raw_proposal_count"])
        expected_files = {name: item["sha256"] for name, item in binding["files"].items()}
        if result["model_identity"]["files_sha256"] != expected_files or result["model_identity"]["fit_ids_sha256"] != binding["fit_ids_sha256"]:
            raise ValueError("inference output model identity mismatch")
        receipt.update(status="completed", validation=validation, output_sha256=_sha(directory / "output.json"))
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(finished_utc=_utc(), elapsed_seconds=time.monotonic() - started)
        _write(directory / "worker_receipt.json", receipt)


def run_examples(run_dir, output_dir, *, examples_dir=None, count=20, device="cpu", threads=1, max_seconds=300):
    if type(count) is not int or count < 1 or type(threads) is not int or threads < 1:
        raise ValueError("proposal count and thread count must be positive integers")
    if not _finite(max_seconds) or max_seconds <= 0:
        raise ValueError("example deadline must be finite and positive")
    started, started_utc = time.monotonic(), _utc()
    deadline = started + max_seconds
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[1]
    examples = Path(examples_dir).resolve() if examples_dir else repository / "examples/retrain"
    if output.exists():
        raise FileExistsError("example output must be a new directory")
    route, bindings, sources, shared_ids = _preflight(run, repository)
    output.mkdir(parents=True)
    cases = {}
    for name, filename in EXAMPLES.items():
        directory = output / name
        directory.mkdir()
        source = examples / filename
        (directory / "input.json").write_bytes(source.read_bytes())
        binding = bindings["forward" if name == "forward" else "inverse"]
        arguments = ["predict" if name == "forward" else "propose", "--model-dir", binding["directory"],
                     "--input", str(directory / "input.json"), "--output", str(directory / "output.json"),
                     "--device", "cpu" if name == "forward" and binding["family"] == "classical" else device,
                     "--threads", str(threads)]
        if name == "forward":
            arguments += ["--family", binding["family"]]
            if binding["family"] == "classical":
                arguments.append("--trusted-generated-artifact")
        else:
            arguments += ["--count", str(count), "--seed", str(SEED)]
        cases[name] = {"source_input": str(source), "input_sha256": _sha(source), "cli_arguments": arguments}
    plan = {"schema_version": "pu-deployment-examples-v1", "run": str(run), "created_utc": _utc(),
            "seed": SEED, "seed_rule": "predeclared first seed; never selected by outer-fold performance",
            "raw_proposal_count": count, "models": bindings, "source_sha256": sources,
            "script_sha256": _sha(__file__), "guard_script_sha256": _sha(repository / "scripts/verify_pu_retrain_oof.py"),
            "shared_observation_ids": shared_ids, "cases": cases, "max_seconds": max_seconds,
            "selected_forward_family": route["selected_forward_family"]}
    _write(output / "example_lock.json", plan)
    completed = []
    result = {"status": "running", "parent_pid": os.getpid(), "started_utc": started_utc,
              "seed": SEED, "new_fitting": False, "forward_ranking_performed": False,
              "chemical_validation": "not_performed", "experimental_success": None, "hit_rate_computed": False}
    try:
        for name in cases:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("shared example deadline exceeded")
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(output / "example_lock.json"),
                       "--worker-case", name, "--worker-plan-sha", _sha(output / "example_lock.json")]
            call_started = time.monotonic()
            _write(output / name / "call_start.json", {"command": command, "started_utc": _utc(),
                   "parent_pid": os.getpid(), "remaining_shared_seconds": remaining})
            try:
                process = subprocess.run(command, cwd=repository,
                                         env={**os.environ, "PYTHONPATH": str(repository / "src")},
                                         capture_output=True, text=True, timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                for filename, content in (("stdout.log", exc.stdout), ("stderr.log", exc.stderr)):
                    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content or ""
                    (output / name / filename).write_text(text, encoding="utf-8")
                _write(output / name / "call_end.json", {"status": "timeout", "returncode": None,
                       "finished_utc": _utc(), "elapsed_seconds": time.monotonic() - call_started})
                raise
            (output / name / "stdout.log").write_text(process.stdout, encoding="utf-8")
            (output / name / "stderr.log").write_text(process.stderr, encoding="utf-8")
            _write(output / name / "call_end.json", {"status": "exited", "returncode": process.returncode,
                   "finished_utc": _utc(), "elapsed_seconds": time.monotonic() - call_started,
                   "stdout_sha256": _sha(output / name / "stdout.log"),
                   "stderr_sha256": _sha(output / name / "stderr.log")})
            if process.returncode != 0:
                raise RuntimeError(f"example {name} exited with code {process.returncode}; no retry performed")
            audit = _json(output / name / "no_training_audit.json")
            if audit["status"] != "passed" or audit["optimizer_updates"] != 0 or audit["blocked_calls"]:
                raise ValueError("example did not satisfy its actual no-training audit")
            binding = bindings["forward" if name == "forward" else "inverse"]
            if verify_model_files(binding["directory"], binding["family"], shared_ids) != binding:
                raise ValueError("deployment model bytes changed during example inference")
            if time.monotonic() > deadline:
                raise TimeoutError("shared example deadline exceeded after call verification")
            completed.append({"case": name, "command": command, "returncode": process.returncode,
                              "worker_receipt_sha256": _sha(output / name / "worker_receipt.json"),
                              "no_training_audit_sha256": _sha(output / name / "no_training_audit.json")})
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=_utc(), elapsed_seconds=time.monotonic() - started, completed_calls=completed,
                      completed_raw_proposal_count=sum(count for call in completed if call["case"].startswith("inverse_")),
                      files_sha256={path.relative_to(output).as_posix(): _sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
        _write(output / "example_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--output")
    parser.add_argument("--examples")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--worker-plan", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", choices=tuple(EXAMPLES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-plan-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        _worker(args.worker_plan, args.worker_case, args.worker_plan_sha)
    else:
        if not args.run or not args.output:
            parser.error("--run and --output are required")
        print(json.dumps(run_examples(args.run, args.output, examples_dir=args.examples, count=args.count,
                                      device=args.device, threads=args.threads, max_seconds=args.max_seconds), ensure_ascii=False))
