"""显式聚合物拓扑后端的数据模型与输入校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from polymer_workflow.models import MonomerTemplate, SiteRecord


OUTSIDE_CONTRACT_REASON = "topology_showcase_outside_current_100_node_tree_contract"


@dataclass(frozen=True)
class ReactionSchema:
    """描述一条受模板约束的亲电–亲核反应模式。"""

    schema_id: str
    display_name: str
    smipoly_rule_id: int | None
    new_bond_type: str
    implementation: str


REACTION_SCHEMAS: dict[str, ReactionSchema] = {
    "urethane_rule_111": ReactionSchema(
        "urethane_rule_111",
        "NCO–OH urethane formation",
        111,
        "C-O + N=C->N-C",
        "reuse_rule_111",
    ),
    "amide_rule_108": ReactionSchema(
        "amide_rule_108",
        "acyl chloride–amine amidation",
        108,
        "C-N",
        "reuse_rule_108",
    ),
    "polyimide_rule_110": ReactionSchema(
        "polyimide_rule_110",
        "dianhydride–diamine polyimidization",
        110,
        "2x(C-N)",
        "reuse_rule_110",
    ),
    "esterification_cooh_oh": ReactionSchema(
        "esterification_cooh_oh",
        "carboxylic acid–alcohol esterification",
        104,
        "C-O",
        "template_constrained_rule_104_edit",
    ),
    "lactone_rop": ReactionSchema(
        "lactone_rop",
        "hydroxyl-initiated lactone ring opening",
        None,
        "break acyl-C/ring-O; form previous-O/acyl-C",
        "fixed_lactone_ring_opening_edit",
    ),
    "epoxide_amine_ring_opening": ReactionSchema(
        "epoxide_amine_ring_opening",
        "epoxide–primary amine ring opening",
        None,
        "break attacked-C/ring-O; form attacked-C/N",
        "template_constrained_epoxide_amine_edit",
    ),
    "bbl_perinone_cyclocondensation": ReactionSchema(
        "bbl_perinone_cyclocondensation",
        "NTCDA–ortho-diamine perinone cyclocondensation",
        None,
        "C=N + 2x(C-N) per grouped event",
        "template_constrained_grouped_perinone_edit",
    ),
}


@dataclass(frozen=True)
class MonomerSpec:
    """保存显式拓扑配置中的一种原始中性单体。"""

    monomer_id: str
    name: str
    smiles: str
    reaction_schema_ids: tuple[str, ...]
    color_key: str
    semantic_role: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MonomerSpec":
        """从 JSON 字典构造单体，并拒绝空字段与未知反应模式。"""

        schemas = tuple(str(value) for value in payload.get("reaction_schema_ids", ()))
        unknown = sorted(set(schemas) - set(REACTION_SCHEMAS))
        if unknown:
            raise ValueError(f"单体含未知 reaction_schema_id: {unknown}")
        values = {
            "monomer_id": str(payload.get("monomer_id", "")).strip(),
            "name": str(payload.get("name", "")).strip(),
            "smiles": str(payload.get("smiles", "")).strip(),
            "color_key": str(payload.get("color_key", "")).strip(),
            "semantic_role": str(payload.get("semantic_role", "")).strip(),
        }
        empty = [key for key, value in values.items() if not value]
        if empty or not schemas:
            raise ValueError(f"单体字段不完整: empty={empty}, schemas={schemas}")
        return cls(reaction_schema_ids=schemas, **values)


@dataclass
class TopologyNode:
    """表示一个单体实例节点及其代数、臂和主链/侧链元数据。"""

    instance_id: str
    monomer_id: str
    generation: int = -1
    arm: int = -1
    backbone: bool = False
    sidechain: bool = False
    branch_state: str = ""
    used_sites: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TopologyNode":
        """从配置字典构造节点；布尔字段不会由非空字符串隐式转真。"""

        instance_id = str(payload.get("instance_id", "")).strip()
        monomer_id = str(payload.get("monomer_id", "")).strip()
        if not instance_id or not monomer_id:
            raise ValueError("TopologyNode 必须含非空 instance_id 和 monomer_id。")
        return cls(
            instance_id=instance_id,
            monomer_id=monomer_id,
            generation=int(payload.get("generation", -1)),
            arm=int(payload.get("arm", -1)),
            backbone=payload.get("backbone", False) is True,
            sidechain=payload.get("sidechain", False) is True,
            branch_state=str(payload.get("branch_state", "")).strip(),
        )

    def as_dict(self) -> dict[str, Any]:
        """返回稳定、可直接写入 JSON 的节点字典。"""

        return asdict(self)


@dataclass
class TopologyEdge:
    """保存一条显式端口边及 qmdesc 评分和全原子编辑审计记录。"""

    edge_id: str
    source_instance_id: str
    source_site_id: str
    target_instance_id: str
    target_site_id: str
    reaction_schema_id: str
    reaction_event_id: str | None = None
    event_bond_role: str | None = None
    smipoly_rule_id: int | None = None
    source_hirshfeld_charge: float | None = None
    target_hirshfeld_charge: float | None = None
    electrophile_hirshfeld_charge: float | None = None
    nucleophile_hirshfeld_charge: float | None = None
    reaction_score: float | None = None
    new_bond_type: str | None = None
    atom_edit_record: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TopologyEdge":
        """兼容嵌套 endpoint 或扁平字段，并校验反应模式。"""

        source = dict(payload.get("source", {}))
        target = dict(payload.get("target", {}))
        schema_id = str(payload.get("reaction_schema_id", "")).strip()
        if schema_id not in REACTION_SCHEMAS:
            raise ValueError(f"未知 reaction_schema_id: {schema_id}")
        edge = cls(
            edge_id=str(payload.get("edge_id", "")).strip(),
            source_instance_id=str(source.get("monomer_instance_id", payload.get("source_instance_id", ""))).strip(),
            source_site_id=str(source.get("site_id", payload.get("source_site_id", ""))).strip(),
            target_instance_id=str(target.get("monomer_instance_id", payload.get("target_instance_id", ""))).strip(),
            target_site_id=str(target.get("site_id", payload.get("target_site_id", ""))).strip(),
            reaction_schema_id=schema_id,
            reaction_event_id=(
                str(payload.get("reaction_event_id", "")).strip() or None
            ),
            event_bond_role=(
                str(payload.get("event_bond_role", "")).strip() or None
            ),
        )
        required = (
            edge.edge_id,
            edge.source_instance_id,
            edge.source_site_id,
            edge.target_instance_id,
            edge.target_site_id,
        )
        if not all(required):
            raise ValueError(f"边 {payload!r} 缺少 ID 或显式端口。")
        if edge.source_instance_id == edge.target_instance_id:
            raise ValueError(f"边 {edge.edge_id} 不能连接同一单体实例。")
        return edge

    def as_dict(self) -> dict[str, Any]:
        """返回包含嵌套端点和化学审计字段的 JSON 字典。"""

        return {
            "edge_id": self.edge_id,
            "source": {
                "monomer_instance_id": self.source_instance_id,
                "site_id": self.source_site_id,
            },
            "target": {
                "monomer_instance_id": self.target_instance_id,
                "site_id": self.target_site_id,
            },
            "reaction_schema_id": self.reaction_schema_id,
            "reaction_event_id": self.resolved_reaction_event_id,
            "event_bond_role": self.resolved_event_bond_role,
            "smipoly_rule_id": self.smipoly_rule_id,
            "source_hirshfeld_charge": self.source_hirshfeld_charge,
            "target_hirshfeld_charge": self.target_hirshfeld_charge,
            "electrophile_hirshfeld_charge": self.electrophile_hirshfeld_charge,
            "nucleophile_hirshfeld_charge": self.nucleophile_hirshfeld_charge,
            "reaction_score": self.reaction_score,
            "new_bond_type": self.new_bond_type,
            "atom_edit_record": self.atom_edit_record,
        }

    @property
    def resolved_reaction_event_id(self) -> str:
        """返回稳定反应事件 ID；旧配置中的普通边按一边一事件处理。"""

        return self.reaction_event_id or self.edge_id

    @property
    def resolved_event_bond_role(self) -> str:
        """返回事件内成键角色；普通反应使用统一占位语义。"""

        return self.event_bond_role or "single_event_link"


@dataclass
class TopologyModel:
    """一个可变节点数、可含多反应类型的有限聚合物展示模型。"""

    polymer_id: str
    display_name: str
    topology_type: str
    parameters: dict[str, Any]
    monomers: list[MonomerSpec]
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]
    references: list[dict[str, str]] = field(default_factory=list)
    scientific_scope_note: str = "literature-inspired idealized finite topology"
    training_eligible: bool = False
    training_exclusion_reason: str = OUTSIDE_CONTRACT_REASON

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TopologyModel":
        """解析统一配置并执行不依赖化学预测的结构完整性检查。"""

        model = cls(
            polymer_id=str(payload.get("polymer_id", "")).strip(),
            display_name=str(payload.get("display_name", "")).strip(),
            topology_type=str(payload.get("topology_type", "")).strip(),
            parameters=dict(payload.get("parameters", {})),
            monomers=[MonomerSpec.from_dict(item) for item in payload.get("monomers", ())],
            nodes=[TopologyNode.from_dict(item) for item in payload.get("nodes", ())],
            edges=[TopologyEdge.from_dict(item) for item in payload.get("edges", ())],
            references=[dict(item) for item in payload.get("references", ())],
            scientific_scope_note=str(
                payload.get("scientific_scope_note", "literature-inspired idealized finite topology")
            ),
            training_eligible=payload.get("training_eligible", False) is True,
            training_exclusion_reason=str(payload.get("training_exclusion_reason", OUTSIDE_CONTRACT_REASON)),
        )
        model.validate_basic()
        return model

    def validate_basic(self) -> None:
        """校验唯一 ID、端点存在性和固定训练边界，不执行 qmdesc。"""

        if not self.polymer_id or not self.display_name or not self.topology_type:
            raise ValueError("polymer_id、display_name 和 topology_type 均不能为空。")
        if not self.monomers or not self.nodes or not self.edges:
            raise ValueError(f"{self.polymer_id} 的单体、节点和边均必须非空。")
        monomer_ids = [item.monomer_id for item in self.monomers]
        node_ids = [item.instance_id for item in self.nodes]
        edge_ids = [item.edge_id for item in self.edges]
        for label, values in (("monomer_id", monomer_ids), ("instance_id", node_ids), ("edge_id", edge_ids)):
            if len(values) != len(set(values)):
                raise ValueError(f"{self.polymer_id} 含重复 {label}。")
        unknown_monomers = sorted({node.monomer_id for node in self.nodes} - set(monomer_ids))
        if unknown_monomers:
            raise ValueError(f"节点引用未知单体: {unknown_monomers}")
        unknown_nodes = sorted(
            {
                endpoint
                for edge in self.edges
                for endpoint in (edge.source_instance_id, edge.target_instance_id)
                if endpoint not in set(node_ids)
            }
        )
        if unknown_nodes:
            raise ValueError(f"边引用未知节点: {unknown_nodes}")
        if self.training_eligible or self.training_exclusion_reason != OUTSIDE_CONTRACT_REASON:
            raise ValueError("展示拓扑必须固定为 training_eligible=false 并保留冻结契约外标记。")

    def as_dict(self) -> dict[str, Any]:
        """序列化统一显式拓扑配置，供 CLI 输入和发布审计。"""

        return {
            "polymer_id": self.polymer_id,
            "display_name": self.display_name,
            "topology_type": self.topology_type,
            "parameters": self.parameters,
            "monomers": [asdict(item) for item in self.monomers],
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
            "references": self.references,
            "scientific_scope_note": self.scientific_scope_note,
            "training_eligible": self.training_eligible,
            "training_exclusion_reason": self.training_exclusion_reason,
        }


@dataclass
class PreparedMonomer:
    """把配置单体、旧工作流模板和逐模式 PSMILES 组合为内部记录。"""

    spec: MonomerSpec
    template: MonomerTemplate
    psmiles_by_schema: dict[str, str]
    classification: dict[str, Any]

    def site(self, site_id: str) -> SiteRecord:
        """按显式端口 ID 返回位点；缺失时给出可操作错误。"""

        for record in self.template.sites:
            if record.site_id == site_id:
                return record
        available = [record.site_id for record in self.template.sites]
        raise KeyError(f"单体 {self.spec.monomer_id} 不含端口 {site_id}；可用端口: {available}")


@dataclass(frozen=True)
class PreparationBundle:
    """保存一个模型的单体模板、分类结果和 qmdesc 身份。"""

    prepared: tuple[PreparedMonomer, ...]
    smipoly_report: dict[str, Any]
    qmdesc_report: dict[str, Any]

    @property
    def by_monomer_id(self) -> dict[str, PreparedMonomer]:
        """返回 monomer_id 到已准备单体的映射。"""

        return {item.spec.monomer_id: item for item in self.prepared}

    @property
    def templates(self) -> list[MonomerTemplate]:
        """按配置顺序返回兼容旧原子编辑辅助函数的模板列表。"""

        return [item.template for item in self.prepared]


def load_models(path: str | Path) -> list[TopologyModel]:
    """从单模型或 ``models`` 列表 JSON 文件读取统一配置。"""

    import json

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records: Sequence[Mapping[str, Any]]
    if isinstance(payload, Mapping) and "models" in payload:
        records = payload["models"]
    elif isinstance(payload, Mapping):
        records = [payload]
    else:
        raise ValueError("配置根节点必须是单模型字典或含 models 的字典。")
    models = [TopologyModel.from_dict(record) for record in records]
    ids = [model.polymer_id for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("配置集合含重复 polymer_id。")
    return models
