"""仅在当前训练折拟合的特征、目标变换与逆向原料目录。"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

MAIN_TARGETS = ("tg", "tm", "tensile_strength_ultimate", "elongation_at_break")
ROLES = {"D": 0, "P": 1, "E": 2, "OTHER": 3}
NUMERIC_CONDITIONS = (
    "temperature_c", "relative_humidity_percent", "strain_rate_s_inv",
    "crosshead_speed_mm_min", "heating_rate_c_min", "cooling_rate_c_min",
    "frequency_hz", "specimen_thickness_mm", "scan_index", "transition_event_index",
)
CATEGORICAL_CONDITIONS = (
    "method", "standard", "load_mode", "atmosphere", "specimen_geometry",
    "specimen_preparation", "conditioning", "transition_criterion", "transition_scope",
)


def component_key(component: dict) -> str:
    """原料牌号/规格参与身份，投料量和文献/材料 ID 不参与原料身份。"""
    identity = {k: component.get(k) for k in (
        "component_id", "role", "smiles", "psmiles", "mn_g_mol",
        "molecular_weight_g_mol", "functionality", "mixture_fraction")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _hash_text(text: str, size: int) -> np.ndarray:
    """固定有符号哈希，不拟合全数据词表；不接收自由文本中的实验结果。"""
    out = np.zeros(size, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for token in tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]:
        value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
        out[value % size] += 1 if value & (1 << 63) else -1
    norm = np.linalg.norm(out)
    return out / max(norm, 1.0)


@lru_cache(maxsize=8192)
def _molecular_features(smiles: str, fallback: str, bits: int) -> np.ndarray:
    """Morgan 指纹是无标签确定性变换；无结构时保留名称哈希及结构缺失标记。"""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    fingerprint = np.zeros(bits, dtype=np.float32)
    if molecule is not None:
        vector = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits).GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(vector, fingerprint)
    name = _hash_text(fallback, 32) if molecule is None else np.zeros(32, dtype=np.float32)
    return np.concatenate([fingerprint, name, [float(molecule is not None)]]).astype(np.float32)


def _component_numbers(component):
    values = [_finite(component.get(k)) for k in ("mn_g_mol", "molecular_weight_g_mol", "functionality")]
    # 未报告或非正规格不参与拟合；不以填充值参与任何物理换算。
    return np.array([math.log1p(v) if v is not None and v > 0 else np.nan for v in values])


def _condition_numbers(row):
    fields = row.get("conditions", {})
    values = []
    for key in NUMERIC_CONDITIONS:
        field = fields.get(key, {})
        value = _finite(field.get("value")) if field.get("mask") else None
        values.append(value if value is not None else np.nan)
    return np.asarray(values, dtype=np.float64)


def _scale_fit(values):
    """按列拟合中位数和四分位距；全缺失列显式采用零中心、单位尺度。"""
    center, scale = [], []
    for column in np.asarray(values).T:
        observed = column[np.isfinite(column)]
        center.append(float(np.median(observed)) if len(observed) else 0.0)
        spread = float(np.subtract(*np.percentile(observed, [75, 25]))) if len(observed) else 0.0
        if spread <= 1e-8:
            spread = float(np.std(observed)) if len(observed) else 0.0
        scale.append(spread if spread > 1e-8 else 1.0)
    return {"center": center, "scale": scale}


def _scale_apply(values, state):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    normalized = (np.where(mask, values, state["center"]) - state["center"]) / state["scale"]
    return normalized.astype(np.float32), mask


class FoldFeatures:
    """全量母表共享，拟合状态/目录严格局限于给定 train IDs。"""

    def __init__(self, tasks=None, fingerprint_bits=128, condition_hash_dim=64, mechanical_log=False):
        self.tasks = list(tasks or MAIN_TARGETS)
        self.fingerprint_bits = int(fingerprint_bits)
        self.condition_hash_dim = int(condition_hash_dim)
        self.mechanical_log = bool(mechanical_log)
        self.state = None

    @property
    def component_dim(self):
        return self.fingerprint_bits + 32 + 1 + 6

    @property
    def condition_dim(self):
        return len(NUMERIC_CONDITIONS) + self.condition_hash_dim

    def fit(self, rows):
        if self.state is not None:
            raise RuntimeError("FoldFeatures cannot be refit; construct a new fold object")
        if not rows or any(not row.get("shared_trainable") for row in rows):
            raise ValueError("fit requires nonempty shared-trainable observations")
        ids = [row["observation_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicated training observation IDs")
        components = {}
        for row in rows:
            for component in row["components"]:
                zero_amount = bool(component.get("ratio_mask")) and _finite(component.get("molecule_ratio")) == 0
                if component.get("identity_mask", 1) and not zero_amount:
                    components[component_key(component)] = copy.deepcopy(component)
        if not components:
            raise ValueError("no identifiable training components")
        catalog = [{"key": key, "component": components[key]} for key in sorted(components)]
        targets = {}
        for task in self.tasks:
            values = np.asarray([row["value"] for row in rows if row["target"] == task], dtype=np.float64)
            if not len(values):
                targets[task] = {"supported": False, "center": 0.0, "scale": 1.0, "physical_scale": 1.0,
                                 "transform": "linear", "n": 0}
                continue
            transform = "log1p" if self.mechanical_log and ("strength" in task or "elongation" in task) else "linear"
            if transform == "log1p" and (values < 0).any():
                raise ValueError(f"negative observed mechanical value in {task}; resolve source semantics")
            working = np.log1p(values) if transform == "log1p" else values
            scale = _scale_fit(working[:, None])
            targets[task] = {"supported": True, "center": scale["center"][0], "scale": scale["scale"][0],
                             "physical_scale": _scale_fit(values[:, None])["scale"][0],
                             "transform": transform, "n": int(len(values))}
        self.state = {
            "schema_version": "fold_features_v2", "fit_observation_ids": sorted(ids),
            "fit_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
            "tasks": self.tasks, "fingerprint_bits": self.fingerprint_bits,
            "condition_hash_dim": self.condition_hash_dim, "mechanical_log": self.mechanical_log,
            "component_scale": _scale_fit([_component_numbers(item["component"]) for item in catalog]),
            "condition_scale": _scale_fit([_condition_numbers(row) for row in rows]),
            "targets": targets, "catalog": catalog,
            "condition_policy": "allowlisted structured fields only; no raw outcome text or source/material IDs",
        }
        self._index = {item["key"]: i for i, item in enumerate(catalog)}
        return self

    @classmethod
    def from_state(cls, state):
        result = cls(state["tasks"], state["fingerprint_bits"], state["condition_hash_dim"], state["mechanical_log"])
        result.state = copy.deepcopy(state)
        result._index = {item["key"]: i for i, item in enumerate(state["catalog"])}
        return result

    def encode_value(self, value, task):
        state = self.state["targets"][task]
        if state["transform"] == "log1p":
            if value < 0:
                raise ValueError(f"log1p does not support negative mechanical target: {task}")
            value = math.log1p(value)
        return (value - state["center"]) / state["scale"]

    def decode_values(self, values, task):
        state = self.state["targets"][task]
        restored = np.asarray(values, dtype=np.float64) * state["scale"] + state["center"]
        return np.expm1(restored) if state["transform"] == "log1p" else restored

    def transform(self, rows):
        if self.state is None:
            raise RuntimeError("fit must precede transform")
        batch, width = len(rows), max(len(row["components"]) for row in rows)
        catalog_size, task_count = len(self._index), len(self.tasks)
        result = {
            "component_features": np.zeros((batch, width, self.component_dim), np.float32),
            "component_roles": np.zeros((batch, width), np.int64),
            "component_mask": np.zeros((batch, width), bool),
            "fractions": np.zeros((batch, width), np.float32), "fraction_mask": np.zeros((batch, width), bool),
            "conditions": np.zeros((batch, self.condition_dim), np.float32),
            "condition_mask": np.zeros((batch, self.condition_dim), bool),
            "task_id": np.zeros(batch, np.int64), "y": np.zeros(batch, np.float32),
            "targets": np.zeros((batch, task_count), np.float32), "target_mask": np.zeros((batch, task_count), bool),
            "presence": np.zeros((batch, catalog_size), np.float32),
            "presence_mask": np.zeros((batch, catalog_size), bool),
            "composition": np.zeros((batch, catalog_size), np.float32),
            "composition_mask": np.zeros((batch, catalog_size), bool),
            "catalog_complete": np.ones(batch, bool), "forward_supported": np.zeros(batch, bool),
        }
        for i, row in enumerate(rows):
            task = self.tasks.index(row["target"])
            result["task_id"][i] = task
            result["y"][i] = self.encode_value(float(row["value"]), row["target"])
            result["targets"][i, task] = result["y"][i]
            result["target_mask"][i, task] = True
            result["forward_supported"][i] = self.state["targets"][row["target"]]["supported"]
            components = row["components"]
            # 完整原料身份集合可以监督目录内的缺席；缺身份项时只监督已知阳性。
            identities_complete = all(c.get("identity_mask", 1) or (c.get("ratio_mask") and _finite(c.get("molecule_ratio")) == 0)
                                      for c in components)
            result["presence_mask"][i] = identities_complete
            for j, component in enumerate(components):
                structure = component.get("smiles") or component.get("psmiles") or component.get("structure_psmiles") or ""
                fallback = component.get("component_parent_code") or component.get("name") or "unknown"
                molecular = _molecular_features(structure, str(fallback), self.fingerprint_bits)
                numbers, number_mask = _scale_apply(_component_numbers(component), self.state["component_scale"])
                result["component_features"][i, j] = np.concatenate([molecular, numbers, number_mask])
                result["component_roles"][i, j] = ROLES.get(component.get("role"), 3)
                ratio = _finite(component.get("molecule_ratio"))
                known_ratio = bool(component.get("ratio_mask")) and ratio is not None and ratio >= 0
                result["component_mask"][i, j] = not (known_ratio and ratio == 0)
                result["fractions"][i, j] = ratio if known_ratio else 0.0
                result["fraction_mask"][i, j] = known_ratio
                key = component_key(component)
                if known_ratio and ratio == 0:
                    # 明确零投料代表未加入；原记录仍保留，不能把可选E的NONE改成存在。
                    if key in self._index:
                        result["presence_mask"][i, self._index[key]] = True
                    continue
                if not component.get("identity_mask", 1) or key not in self._index:
                    result["catalog_complete"][i] = False
                    continue
                column = self._index[key]
                result["presence"][i, column] = 1.0
                result["presence_mask"][i, column] = True
                if known_ratio and ratio > 0:
                    result["composition"][i, column] += ratio
                    result["composition_mask"][i, column] = True
            numeric, numeric_mask = _scale_apply(_condition_numbers(row), self.state["condition_scale"])
            fields = row.get("conditions", {})
            phrases = [f"{key}={fields[key]['value']}" for key in CATEGORICAL_CONDITIONS
                       if key in fields and fields[key].get("mask") and fields[key].get("value") is not None]
            # 任务子类型与相域本身属于测量定义，不把自由文本定义中的数值带进来。
            for key in ("transition_criterion", "transition_scope", "architecture"):
                if row.get(key):
                    phrases.append(f"{key}={row[key]}")
            categorical = _hash_text(" ".join(phrases), self.condition_hash_dim)
            result["conditions"][i] = np.concatenate([numeric, categorical])
            result["condition_mask"][i] = np.concatenate([numeric_mask, np.full(self.condition_hash_dim, bool(phrases))])
        return result

    def tabular(self, rows):
        """角色分组的平均/计量加权特征，用于同数据同划分的表格基线。"""
        batch = self.transform(rows)
        parts = []
        for role in range(4):
            mask = batch["component_mask"] & (batch["component_roles"] == role)
            features = batch["component_features"]
            count = mask.sum(axis=1, keepdims=True)
            parts.append((features * mask[..., None]).sum(axis=1) / np.maximum(count, 1))
            parts.append((features * (mask * batch["fractions"])[..., None]).sum(axis=1))
            parts.append(count)
        parts.extend([batch["conditions"], batch["condition_mask"]])
        return np.concatenate(parts, axis=1).astype(np.float32)

    def support(self, rows):
        arrays = self.transform(rows)
        return {"observations": len(rows), "complete_catalog_observations": int(arrays["catalog_complete"].sum()),
                "partial_or_unseen_catalog_ids": [row["observation_id"] for row, flag in zip(rows, arrays["catalog_complete"]) if not flag],
                "fit_ids_sha256": self.state["fit_ids_sha256"]}
