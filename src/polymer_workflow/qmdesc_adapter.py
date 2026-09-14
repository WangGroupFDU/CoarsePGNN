"""qmdesc 1.0.6 的显式设备、批量推理和磁盘缓存适配器。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Iterable

import torch
from rdkit import Chem


SUPPORTED_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}


@dataclass(frozen=True)
class ChargePrediction:
    """保存一个规范单体的 qmdesc Hirshfeld 电荷和缓存来源。"""

    canonical_smiles: str
    partial_charges: tuple[float, ...]
    atom_records: tuple[dict[str, object], ...]
    cache_hit: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    流式计算文件 SHA-256。

    参数:
        path: 待计算文件。
        chunk_size: 每次读取字节数，默认 1 MiB。

    返回值:
        小写十六进制 SHA-256。

    可能报错:
        文件不存在或无读取权限时保留底层 ``OSError``，不伪造哈希。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_qmdesc_smiles(smiles: str) -> tuple[str, Chem.Mol]:
    """
    规范化并验证 qmdesc 第一版适用域。

    参数:
        smiles: 用户原始单体 SMILES，不能含 PSMILES dummy atom。

    返回值:
        ``(canonical_smiles, canonical_mol)``；后者的原子顺序定义稳定
        ``orig_atom_index``，并与传给 qmdesc 的字符串一致。

    关键流程:
        拒绝多片段、dummy atom、形式电荷和未支持元素；去除显式氢后生成一次规范 SMILES，
        再从该字符串重建分子以固定索引。

    可能报错:
        SMILES 无法解析、含盐/离子/金属/Si 或其他超域元素时抛出 ``ValueError``。
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit 无法解析 SMILES: {smiles}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("qmdesc 第一版只接受单一连通片段，不能包含盐、溶剂或点分隔组分。")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("qmdesc 只接收原始单体 SMILES，不能含 * dummy atom。")
    charged = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0]
    if charged:
        raise ValueError(f"qmdesc 第一版拒绝形式带电原子，索引: {charged}")
    unsupported = sorted(
        {(atom.GetSymbol(), atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetAtomicNum() not in SUPPORTED_ATOMIC_NUMBERS},
        key=lambda item: item[1],
    )
    if unsupported:
        details = ", ".join(f"{symbol}@{index}" for symbol, index in unsupported)
        raise ValueError(f"qmdesc 第一版存在未支持元素: {details}")

    mol = Chem.RemoveHs(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    canonical_mol = Chem.MolFromSmiles(canonical)
    if canonical_mol is None:
        raise RuntimeError("规范 SMILES 无法由 RDKit 重建。")
    return canonical, canonical_mol


class QmDescPredictor:
    """
    qmdesc 1.0.6 的可复用批量预测器。

    参数:
        device: ``cpu``、``cuda`` 或 PyTorch 可识别设备；不会自动抢占 GPU。
        cache_dir: 按规范 SMILES、模型版本和权重哈希保存 JSON 缓存的目录。

    返回值:
        ``predict_many`` 返回规范 SMILES 到 ``ChargePrediction`` 的映射。

    关键流程:
        权重只加载一次；未缓存的唯一单体合并成一个 qmdesc 分子图进行推理；结果按
        RDKit AddHs 原子数切分，并检查长度、元素顺序和有限值。

    可能报错:
        指定 CUDA 但不可用、权重损坏、模型输出长度不一致或出现 NaN/Inf 时立即停止。
        本适配器不修改环境，也不提供替代电荷。
    """

    def __init__(self, device: str = "cpu", cache_dir: str | Path = ".cache/qmdesc") -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("请求了 CUDA，但当前 PyTorch 未检测到可用 CUDA 设备。")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        import pkg_resources

        checkpoint = pkg_resources.resource_filename("qmdesc.handler", "QM_137k.pt")
        self.checkpoint_path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.model_version = version("qmdesc")
        self._model: torch.nn.Module | None = None

    def _load_model(self) -> torch.nn.Module:
        """在首次未命中缓存时加载官方 qmdesc 权重，后续调用复用同一模型。"""

        if self._model is not None:
            return self._model

        from qmdesc.featurization import get_atom_fdim, get_bond_fdim
        from qmdesc.model import MoleculeModel

        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(state, dict) or "args" not in state or "state_dict" not in state:
            raise RuntimeError("qmdesc 权重文件缺少 args 或 state_dict。")
        atom_fdim = get_atom_fdim()
        bond_fdim = get_bond_fdim() + atom_fdim
        model = MoleculeModel(state["args"], atom_fdim, bond_fdim)
        model.load_state_dict(state["state_dict"])
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def _cache_path(self, canonical_smiles: str) -> Path:
        """根据结构和模型身份返回无碰撞概率可忽略的缓存文件路径。"""

        key_text = f"{canonical_smiles}\nqmdesc={self.model_version}\nweights={self.checkpoint_sha256}"
        key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _atom_records(canonical_smiles: str, charges: list[float]) -> tuple[dict[str, object], ...]:
        """把 AddHs 原子顺序、电荷和氢原子的母重原子索引组合成明确记录。"""

        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise RuntimeError("缓存结构的规范 SMILES 无法解析。")
        heavy_count = mol.GetNumAtoms()
        mol_h = Chem.AddHs(mol)
        if len(charges) != mol_h.GetNumAtoms():
            raise RuntimeError(
                f"qmdesc 输出长度 {len(charges)} 与 AddHs 原子数 {mol_h.GetNumAtoms()} 不一致。"
            )
        records: list[dict[str, object]] = []
        for atom, charge in zip(mol_h.GetAtoms(), charges):
            if not torch.isfinite(torch.tensor(charge)):
                raise RuntimeError("qmdesc 输出包含 NaN 或 Inf。")
            parent = None
            if atom.GetAtomicNum() == 1:
                neighbors = atom.GetNeighbors()
                if len(neighbors) != 1:
                    raise RuntimeError("显式氢原子没有唯一母原子。")
                parent = int(neighbors[0].GetIdx())
            records.append(
                {
                    "qmdesc_atom_index": int(atom.GetIdx()),
                    "orig_atom_index": int(atom.GetIdx()) if atom.GetIdx() < heavy_count else None,
                    "parent_orig_atom_index": parent,
                    "symbol": atom.GetSymbol(),
                    "atomic_number": int(atom.GetAtomicNum()),
                    "is_hydrogen": atom.GetAtomicNum() == 1,
                    "partial_charge": float(charge),
                }
            )
        return tuple(records)

    def _read_cache(self, canonical_smiles: str) -> ChargePrediction | None:
        """读取并完整校验单个缓存；损坏缓存会报错而不是静默重算。"""

        path = self._cache_path(canonical_smiles)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("canonical_smiles") != canonical_smiles
            or payload.get("qmdesc_version") != self.model_version
            or payload.get("checkpoint_sha256") != self.checkpoint_sha256
        ):
            raise RuntimeError(f"qmdesc 缓存身份不一致: {path}")
        charges = [float(value) for value in payload["partial_charge"]]
        records = self._atom_records(canonical_smiles, charges)
        return ChargePrediction(canonical_smiles, tuple(charges), records, True)

    def _write_cache(self, canonical_smiles: str, charges: list[float]) -> None:
        """用同目录临时文件和原子替换写缓存，避免中断留下半个 JSON。"""

        path = self._cache_path(canonical_smiles)
        payload = {
            "canonical_smiles": canonical_smiles,
            "qmdesc_version": self.model_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "partial_charge": [float(value) for value in charges],
        }
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def predict_many(self, canonical_smiles: Iterable[str]) -> dict[str, ChargePrediction]:
        """
        对唯一规范单体批量预测 Hirshfeld 电荷并复用已有缓存。

        参数:
            canonical_smiles: 已通过适用域校验的规范 SMILES 迭代器。

        返回值:
            以规范 SMILES 为键的预测结果；重复输入只推理一次。

        可能报错:
            输入不是规范可解析结构、批量输出长度异常或缓存损坏时停止。
        """

        unique = list(dict.fromkeys(canonical_smiles))
        predictions: dict[str, ChargePrediction] = {}
        missing: list[str] = []
        for smiles in unique:
            cached = self._read_cache(smiles)
            if cached is None:
                missing.append(smiles)
            else:
                predictions[smiles] = cached

        if missing:
            from qmdesc.featurization import mol2graph

            model = self._load_model()
            graph = mol2graph(missing)
            components = graph.get_components()
            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2br, bond_types = components
            data = (
                f_atoms.to(self.device),
                f_bonds.to(self.device),
                a2b.to(self.device),
                b2a.to(self.device),
                b2revb.to(self.device),
                a_scope,
                b_scope,
                b2br.to(self.device),
                bond_types.to(self.device),
            )
            with torch.inference_mode():
                outputs = model(data)
            charges_all = outputs[0].detach().cpu().reshape(-1).tolist()
            expected_counts = [Chem.AddHs(Chem.MolFromSmiles(smi)).GetNumAtoms() for smi in missing]
            if sum(expected_counts) != len(charges_all):
                raise RuntimeError(
                    f"qmdesc 批量输出长度 {len(charges_all)} 与预期 {sum(expected_counts)} 不一致。"
                )
            offset = 0
            for smiles, count in zip(missing, expected_counts):
                charges = [float(value) for value in charges_all[offset : offset + count]]
                offset += count
                records = self._atom_records(smiles, charges)
                self._write_cache(smiles, charges)
                predictions[smiles] = ChargePrediction(smiles, tuple(charges), records, False)
        return predictions
