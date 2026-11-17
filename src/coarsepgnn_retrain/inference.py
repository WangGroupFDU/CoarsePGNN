"""加载已校验权重的无标签推理入口；不声称完成化学或实验验证。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .features import CATEGORICAL_CONDITIONS, NUMERIC_CONDITIONS, ROLES, FoldFeatures
from .training import load_pair


TASK_UNITS = {"tg": "degC", "tm": "degC", "tensile_strength_ultimate": "MPa", "elongation_at_break": "%"}
SHARED_METADATA = {"transition_criterion", "transition_scope", "architecture"}
COMPONENT_IDENTITY = ("component_id", "role", "name", "smiles", "psmiles", "structure_psmiles",
                      "component_parent_code", "mn_g_mol", "molecular_weight_g_mol", "functionality",
                      "mixture_fraction", "identity_mask")


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _conditions(fields):
    """只接受模型确实编码的共同条件，避免静默丢弃用户指定的协议。"""
    if not isinstance(fields, Mapping):
        raise ValueError("conditions must be one shared structured-condition mapping")
    allowed = set(NUMERIC_CONDITIONS) | set(CATEGORICAL_CONDITIONS)
    if set(fields) - allowed:
        raise ValueError(f"unsupported condition fields: {sorted(set(fields) - allowed)}")
    for name, field in fields.items():
        if not isinstance(field, Mapping) or type(field.get("mask")) is not bool:
            raise ValueError(f"condition {name} requires explicit value/mask fields")
        if set(field) - {"value", "mask", "status", "source"}:
            raise ValueError(f"unsupported nested condition/protocol fields: {name}")
        if not field["mask"]:
            continue
        if name in NUMERIC_CONDITIONS:
            _number(field.get("value"), f"condition {name}")
        elif not isinstance(field.get("value"), (str, int, float)) or isinstance(field.get("value"), bool):
            raise ValueError(f"condition {name} must contain a scalar category")
    return copy.deepcopy(dict(fields))


def _shared_context(request):
    context = {"conditions": _conditions(request.get("conditions", {}))}
    for key in SHARED_METADATA:
        if key in request:
            if not isinstance(request[key], str):
                raise ValueError(f"{key} must be one shared string")
            context[key] = request[key]
    # 当前模型每条观察只有一套条件，不支持每个性质各带协议。
    protocol_keys = {"protocol", "protocols", "assays", "conditions_by_target", "protocol_by_target", "constraints"}
    if set(request) & protocol_keys:
        raise ValueError("property-specific or multiple protocols are unsupported; provide one shared conditions mapping")
    return context


def _recipe(row):
    if not isinstance(row, Mapping) or not isinstance(row.get("target"), str):
        raise ValueError("forward requests require a target and normalized recipe")
    if set(row) & {"value", "y_true", "observed_value", "target_value", "targets"}:
        raise ValueError("forward prediction accepts no observed property labels")
    components = row.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)) or not components:
        raise ValueError("a recipe requires at least one component")
    nonzero_or_unknown = 0
    for component in components:
        if not isinstance(component, Mapping) or component.get("role") not in ROLES:
            raise ValueError("each component requires a recognized D/P/E/OTHER role")
        if type(component.get("ratio_mask")) is not bool:
            raise ValueError("each component requires an explicit ratio_mask")
        if component["ratio_mask"]:
            ratio = _number(component.get("molecule_ratio"), "molecule_ratio")
            if ratio < 0:
                raise ValueError("molecule_ratio cannot be negative")
            nonzero_or_unknown += ratio > 0
        else:
            nonzero_or_unknown += 1
    if not nonzero_or_unknown:
        raise ValueError("all components have zero loading")
    # transform 的监督槽只作内部张量接口占位；正向网络只读取配方、条件与任务。
    return {"target": row["target"], "value": 0.0, "components": copy.deepcopy(list(components)),
            **_shared_context(row)}


def _identity(directory, features):
    manifest = json.loads((Path(directory) / "checkpoint_manifest.json").read_text())
    return {"files_sha256": manifest["files"], "fit_ids_sha256": features.state["fit_ids_sha256"],
            "label_transforms": copy.deepcopy(features.state["targets"])}


@torch.no_grad()
def predict_forward(model_dir, recipe_rows, device="cpu"):
    """无性质标签的正向推理；未知任务明确返回 unsupported，并保留所有请求。

    recipe_rows 每项包含 target、components、conditions，可带共同的相态/结构定义。
    components 使用训练母表的规范化字段；输入比例不被自动补全或重新归一化。
    """
    if not isinstance(recipe_rows, Sequence) or isinstance(recipe_rows, (str, bytes)) or not recipe_rows:
        raise ValueError("recipe_rows must be a nonempty sequence")
    prepared = [_recipe(row) for row in recipe_rows]
    forward, _, features = load_pair(model_dir, device=device)
    outputs = []
    for i, (request, row) in enumerate(zip(recipe_rows, prepared)):
        task = row["target"]
        item = {"request_id": request.get("request_id", f"query_{i}"), "target": task,
                "unit": TASK_UNITS.get(task, "training_dataset_unit"), "prediction": None,
                "status": "unsupported", "chemical_validation": "not_performed",
                "experimental_success": None}
        state = features.state["targets"].get(task)
        if state is None or not state["supported"]:
            item["reason"] = "unknown_task" if state is None else "no_training_labels_for_task"
            outputs.append(item)
            continue
        arrays = features.transform([row])
        # 不将占位监督字段传给模型，因此它们不能成为正向预测的输入。
        keys = ("component_features", "component_roles", "component_mask", "fractions", "fraction_mask",
                "conditions", "condition_mask", "task_id")
        batch = {key: torch.from_numpy(arrays[key]).to(device) for key in keys}
        value = float(features.decode_values(forward(batch).cpu().numpy()[0], task))
        if not math.isfinite(value):
            raise FloatingPointError("forward prediction is nonfinite after inverse label transform")
        item.update(prediction=value, status="predicted", training_catalog_complete=bool(arrays["catalog_complete"][0]))
        outputs.append(item)
    return {"rows": outputs, "model_identity": _identity(model_dir, features),
            "observed_labels_required": False, "fitting_performed": False,
            "input_ratios_renormalized": False,
            "scope": "model predictions; property reliability must be assessed from heldout evaluation"}


def predict_classical_forward(model_dir, recipe_rows, *, trusted_generated_artifact=False):
    """只读取已确认来源的本项目经典模型部署产物，预测原单位性质。

    joblib 使用 pickle；即使 SHA 正确，未知来源文件仍可能执行任意代码。因此
    调用方必须确认目录来自自己运行的 refit_full_baseline，才能显式设为 True。
    本函数先核验完整清单与元数据，再反序列化，不接受未知用户上传的 pickle。
    """
    if trusted_generated_artifact is not True:
        raise ValueError("joblib loading requires explicit trusted_generated_artifact=True for your own generated artifacts")
    if not isinstance(recipe_rows, Sequence) or isinstance(recipe_rows, (str, bytes)) or not recipe_rows:
        raise ValueError("recipe_rows must be a nonempty sequence")
    prepared = [_recipe(row) for row in recipe_rows]
    directory = Path(model_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    required = {"selection_lock.json", "features.json", "models.joblib", "audit.json"}
    if (manifest.get("schema_version") != "pu-baseline-deployment-files-v1"
            or len(manifest.get("files", [])) != len(required)
            or {item.get("path") for item in manifest["files"]} != required):
        raise ValueError("invalid or incomplete classical deployment manifest")
    digests = {}
    for item in manifest["files"]:
        path = directory / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"classical artifact size mismatch: {item['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"classical artifact SHA mismatch: {item['path']}")
        digests[item["path"]] = digest
    wrapper = json.loads((directory / "features.json").read_text())
    lock = json.loads((directory / "selection_lock.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    features = FoldFeatures.from_state(wrapper["state"])
    ids = wrapper["fit_observation_ids"]
    identity = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if (audit.get("status") != "completed" or audit.get("independent_test_claim") is not False
            or len(ids) != len(set(ids)) or ids != features.state["fit_observation_ids"]
            or ids != lock["fit_observation_ids"]
            or any(item.get("fit_ids_sha256") != identity for item in (wrapper, lock, audit, features.state))):
        raise ValueError("classical artifact preprocessing/completion identity mismatch")
    import joblib
    models = joblib.load(directory / "models.joblib")
    expected_tasks = {task for task, state in features.state["targets"].items() if state["supported"]}
    if not isinstance(models, dict) or set(models) != expected_tasks:
        raise ValueError("classical model tasks disagree with the saved training state")
    for task, model in models.items():
        if model["train_observations"] != features.state["targets"][task]["n"] or model["train_observations"] != audit["per_task_train_observations"][task]:
            raise ValueError("classical task training denominator mismatch")
        if _number(model["scale"], "classical label scale") <= 0:
            raise ValueError("classical label scale must be positive")
        _number(model["center"], "classical label center")
    outputs = []
    for i, (request, row) in enumerate(zip(recipe_rows, prepared)):
        task = row["target"]
        item = {"request_id": request.get("request_id", f"query_{i}"), "target": task,
                "unit": TASK_UNITS.get(task, "training_dataset_unit"), "prediction": None,
                "status": "unsupported", "chemical_validation": "not_performed", "experimental_success": None}
        if task not in models:
            item["reason"] = "unknown_task" if task not in features.tasks else "no_training_labels_for_task"
            outputs.append(item)
            continue
        matrix = features.tabular([row])
        model = models[task]
        prediction = np.asarray(model["estimator"].predict(matrix))
        if prediction.shape != (1,):
            raise ValueError("classical estimator returned an incompatible prediction shape")
        value = _number(float(prediction[0]) * model["scale"] + model["center"], "classical prediction")
        item.update(prediction=value, status="predicted",
                    training_catalog_complete=bool(features.transform([row])["catalog_complete"][0]))
        outputs.append(item)
    return {"rows": outputs, "model_identity": {"files_sha256": digests, "fit_ids_sha256": identity,
                "selected_candidate": lock["selected_candidate"], "seed": lock["seed"],
                "dataset_sha256": lock["dataset_sha256"]},
            "observed_labels_required": False, "fitting_performed": False, "input_ratios_renormalized": False,
            "scope": "full-data deployment predictions; independent performance belongs to the prior OOF evaluation"}


def _role_limits(request):
    defaults = {"D": (1, 3), "P": (1, 3), "E": (0, 3), "OTHER": (0, 0)}
    specified = request.get("role_limits", {})
    if not isinstance(specified, Mapping) or set(specified) - set(defaults):
        raise ValueError("role_limits must use D/P/E/OTHER keys")
    for role, limits in specified.items():
        if (not isinstance(limits, Sequence) or isinstance(limits, (str, bytes)) or len(limits) != 2
                or any(type(value) is not int for value in limits) or not 0 <= limits[0] <= limits[1]):
            raise ValueError("each role limit must be two ordered nonnegative integers")
        defaults[role] = tuple(limits)
    if defaults["D"][0] < 1 or defaults["P"][0] < 1:
        raise ValueError("PU proposal requires at least one D and one P")
    return tuple(defaults[role] for role in ROLES)


@torch.no_grad()
def propose_inverse(model_dir, request, count, seed, device="cpu"):
    """按指定预算产生目录内组分及连续摩尔份额，不执行隐藏的筛选或补采样。

    request={'targets': {'tg': -20.0}, 'conditions': {...}}。多个目标共享同一套
    条件；本训练版本没有共同状态监督，因而多目标始终标为未验证的组合查询。
    """
    if not isinstance(request, Mapping):
        raise ValueError("inverse request must be a mapping")
    allowed = {"targets", "conditions", "role_limits", "request_id"} | SHARED_METADATA
    context = _shared_context(request)
    if set(request) - allowed:
        raise ValueError(f"unsupported inverse request fields: {sorted(set(request) - allowed)}")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    targets = request.get("targets")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must map trained task names to finite scalar values")
    values = {task: _number(value, f"target {task}") for task, value in targets.items()}
    _, inverse, features = load_pair(model_dir, device=device)
    for task in values:
        if task not in features.tasks or not features.state["targets"][task]["supported"]:
            raise ValueError(f"unsupported inverse task: {task}")
    catalog = features.state["catalog"]
    if not catalog:
        raise ValueError("the checkpoint has no generation catalog")
    # 这里只借用冻结 transform 编码共同条件；目录原料是接口占位，不输入逆模型。
    first_task = next(iter(values))
    arrays = features.transform([{"target": first_task, "value": values[first_task],
                                  "components": [catalog[0]["component"]], **context}])
    target_values = np.zeros((1, len(features.tasks)), dtype=np.float32)
    target_mask = np.zeros_like(target_values, dtype=bool)
    for task, value in values.items():
        index = features.tasks.index(task)
        target_values[0, index] = features.encode_value(value, task)
        target_mask[0, index] = True
    if not np.isfinite(target_values).all():
        raise ValueError("target values overflow the frozen feature precision")
    batch = {"targets": torch.from_numpy(target_values).to(device),
             "target_mask": torch.from_numpy(target_mask).to(device),
             "conditions": torch.from_numpy(arrays["conditions"]).to(device),
             "condition_mask": torch.from_numpy(arrays["condition_mask"]).to(device)}
    generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
    role_limits = _role_limits(request)
    proposals = inverse.propose(batch, count, generator=generator, role_limits=role_limits)
    candidates = []
    for i, proposal in enumerate(proposals):
        fractions = proposal["fractions"]
        if any(not math.isfinite(value) or value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1., abs_tol=1e-6):
            raise FloatingPointError("proposal produced invalid mole fractions; no hidden retry was performed")
        components = []
        for index, fraction in zip(proposal["component_indices"], fractions):
            source = catalog[index]
            component = {key: copy.deepcopy(source["component"][key]) for key in COMPONENT_IDENTITY if key in source["component"]}
            component.update(catalog_key=source["key"], molecule_ratio=fraction,
                             ratio_mask=True, ratio_origin="model_sample")
            components.append(component)
        candidates.append({"proposal_index": i, "components": components,
                           "extender_state": "PRESENT" if any(c["role"] == "E" for c in components) else "NONE",
                           "mixture_component": proposal["mixture_component"],
                           "chemical_validation": "not_performed", "experimental_success": None})
    return {"request_id": request.get("request_id"), "targets": values, "conditions": context["conditions"],
            "target_units": {task: TASK_UNITS.get(task, "training_dataset_unit") for task in values},
            "shared_context": context, "role_limits": dict(zip(ROLES, role_limits)),
            "scope": "single_target_query" if len(values) == 1 else "compositional_query_unvalidated",
            "joint_target_training_verified": False, "candidate_count": len(candidates), "requested_count": count,
            "seed": seed, "proposal_budget_consumed": len(proposals), "hidden_retries": 0,
            "ratio_basis": "molecular_mole_fraction", "candidates": candidates,
            "model_identity": _identity(model_dir, features), "fitting_performed": False,
            "chemical_validation": "not_performed", "experimental_success": None,
            "forward_ranking_performed": False}
