"""工作流的数据模型与输入校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import floor, isfinite
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


RunStatus = Literal["generated", "classified_not_generated", "failed_validation"]


@dataclass(frozen=True)
class MonomerInput:
    """
    表示用户提供的一种单体。

    参数:
        name: 单体名称；若用户省略，由输入顺序生成 ``M1``、``M2`` 等名称。
        smiles: 不带聚合位点的原始单体 SMILES。
        ratio: 非负摩尔配比；至少一种单体必须为正。

    返回值:
        该类是不可变输入记录，不直接修改用户数据。

    关键流程:
        ``parse_monomers`` 会完成类型、名称唯一性和配比合法性检查。

    边界情况:
        零配比允许保留在分类报告中，但不会进入库存；负数、NaN 和空 SMILES 会报错。
    """

    name: str
    smiles: str
    ratio: float


@dataclass
class SiteRecord:
    """
    保存单体模板上的一个可消耗反应位点。

    参数:
        site_id: 单体模板内稳定的位点编号，例如 ``S1``。
        orig_atom_index: 规范 SMILES 重建后用于电荷评分的零基原子索引。
        attachment_atom_index: 主成键原子的零基索引；复杂规则的其余原子见
            ``reaction_atom_indices``。
        role: ``electrophile`` 或 ``nucleophile``。
        functional_group: 位点官能团标签。
        hirshfeld_charge: qmdesc 在原始中性单体上预测的源单体电荷。
        smipoly_rule_id: SMiPoly 官方反应规则编号。
        atom_map: 编号 PSMILES 中 dummy atom 的 atom-map 编号。
        capacity: 本工作流固定为 1，确保每个位点至多使用一次。

    返回值:
        可通过 ``as_dict`` 输出 JSON 兼容字典。

    关键流程:
        复杂规则将多原子反应中心和离去基保存在附加字段中，避免用浮点电荷反查原子。

    边界情况:
        ``attachment_atom_index`` 只是复杂多键反应的主索引，不能代替完整反应中心。
    """

    site_id: str
    orig_atom_index: int
    attachment_atom_index: int
    role: str
    functional_group: str
    hirshfeld_charge: float
    smipoly_rule_id: int
    atom_map: int
    capacity: int = 1
    reaction_atom_indices: dict[str, int] = field(default_factory=dict)
    leaving_atom_indices: list[int] = field(default_factory=list)
    group_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """返回 JSON 可序列化字典；不改变位点记录。"""

        return asdict(self)


@dataclass
class MonomerTemplate:
    """保存规范结构、qmdesc 电荷、SMiPoly 分类和规则位点的单体模板。"""

    input_index: int
    name: str
    raw_smiles: str
    canonical_smiles: str
    ratio: float
    inventory: int
    atom_charges: list[dict[str, Any]]
    sites: list[SiteRecord]
    psmiles: str
    smipoly_classes: dict[str, bool] = field(default_factory=dict)
    qmdesc_cache_hit: bool = False

    def as_dict(self) -> dict[str, Any]:
        """返回用于 ``monomer_sites.json`` 的完整单体记录。"""

        data = asdict(self)
        return data


@dataclass
class MonomerInstance:
    """粗粒化图中的一个单体节点及其位点使用状态。"""

    instance_id: str
    template_index: int
    growth_index: int
    used_sites: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class PolymerEdge:
    """粗粒化树中的一条有向生长边。"""

    source_instance_id: str
    source_site_id: str
    target_instance_id: str
    target_site_id: str
    smipoly_rule_id: int
    source_hirshfeld_charge: float
    target_hirshfeld_charge: float
    electrophile_hirshfeld_charge: float
    nucleophile_hirshfeld_charge: float
    reaction_score: float
    new_bond_type: str

    def as_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容的边记录。"""

        return asdict(self)


@dataclass
class GenerationResult:
    """
    工作流公开返回对象。

    参数:
        status: 固定为 ``generated``、``classified_not_generated`` 或
            ``failed_validation``。
        polymer_id: 本次任务标识。
        output_dir: 本次不可覆盖运行目录。
        message: 面向用户的简短状态说明。
        diagnostics: 机器可读的补充诊断。

    边界情况:
        ``classified_not_generated`` 和 ``failed_validation`` 不会伪造正式聚合物图文件。
    """

    status: RunStatus
    polymer_id: str
    output_dir: Path
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """返回适合 CLI 打印的 JSON 兼容结果。"""

        return {
            "status": self.status,
            "polymer_id": self.polymer_id,
            "output_dir": str(self.output_dir),
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


def parse_monomers(monomers: Sequence[MonomerInput | Mapping[str, Any]]) -> list[MonomerInput]:
    """
    解析并验证 Python API 的单体列表。

    参数:
        monomers: ``MonomerInput`` 或含 ``smiles``、``ratio``、可选 ``name`` 的字典序列。

    返回值:
        按用户输入顺序排列、名称补全后的 ``MonomerInput`` 列表。

    关键流程:
        名称按输入顺序补全；随后检查空值、有限非负配比和名称唯一性。

    可能报错:
        输入为空、字段缺失、名称重复、所有配比为零或配比非法时抛出 ``ValueError``。
    """

    if not isinstance(monomers, Sequence) or isinstance(monomers, (str, bytes)) or not monomers:
        raise ValueError("monomers 必须是非空列表。")

    parsed: list[MonomerInput] = []
    for index, item in enumerate(monomers, start=1):
        if isinstance(item, MonomerInput):
            name = item.name.strip() or f"M{index}"
            smiles = item.smiles.strip()
            ratio = float(item.ratio)
        elif isinstance(item, Mapping):
            if "smiles" not in item or "ratio" not in item:
                raise ValueError(f"第 {index} 个单体缺少 smiles 或 ratio。")
            name = str(item.get("name") or f"M{index}").strip()
            smiles = str(item["smiles"]).strip()
            ratio = float(item["ratio"])
        else:
            raise ValueError(f"第 {index} 个单体不是字典或 MonomerInput。")

        if not name:
            raise ValueError(f"第 {index} 个单体名称为空。")
        if not smiles:
            raise ValueError(f"单体 {name} 的 SMILES 为空。")
        if not isfinite(ratio) or ratio < 0:
            raise ValueError(f"单体 {name} 的 ratio 必须是有限非负数。")
        parsed.append(MonomerInput(name=name, smiles=smiles, ratio=ratio))

    names = [item.name for item in parsed]
    if len(names) != len(set(names)):
        raise ValueError("单体 name 必须唯一，避免实例和位点映射歧义。")
    if not any(item.ratio > 0 for item in parsed):
        raise ValueError("至少一种单体的 ratio 必须大于 0。")
    return parsed


def allocate_inventory(monomers: Iterable[MonomerInput], target_nodes: int) -> dict[str, int]:
    """
    用最大余数法把摩尔配比转换成总和严格等于目标节点数的整数库存。

    参数:
        monomers: 已通过 ``parse_monomers`` 校验的单体。
        target_nodes: 目标单体节点总数，必须为正整数。

    返回值:
        ``{monomer_name: integer_count}`` 字典，总和等于 ``target_nodes``。

    关键流程:
        先向下取整配额，再按小数余数降序分配剩余节点；余数相同时保持输入顺序。

    可能报错:
        正配比单体最终被分到 0 个节点时直接报错，防止静默丢失组分。
    """

    if isinstance(target_nodes, bool) or int(target_nodes) != target_nodes or int(target_nodes) <= 0:
        raise ValueError("target_nodes 必须是正整数。")
    target_nodes = int(target_nodes)
    items = list(monomers)
    ratio_sum = sum(item.ratio for item in items)
    quotas = [item.ratio / ratio_sum * target_nodes for item in items]
    counts = [floor(quota) for quota in quotas]
    remaining = target_nodes - sum(counts)
    order = sorted(range(len(items)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:remaining]:
        counts[index] += 1

    inventory = {item.name: int(count) for item, count in zip(items, counts)}
    zeroed = [item.name for item, count in zip(items, counts) if item.ratio > 0 and count == 0]
    if zeroed:
        joined = ", ".join(zeroed)
        raise ValueError(f"正配比单体被分配为 0 个节点: {joined}；请增大 target_nodes。")
    if sum(inventory.values()) != target_nodes:
        raise RuntimeError("最大余数法库存总和异常。")
    return inventory
