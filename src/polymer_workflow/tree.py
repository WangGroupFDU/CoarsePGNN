"""模板限定的 dynamic_softmax_greedy 连通无环树生成器。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .models import MonomerInstance, MonomerTemplate, PolymerEdge, SiteRecord
from .rules import RULES, pair_score, sites_are_compatible


@dataclass(frozen=True)
class RuleSelection:
    """保存最高有效 Hirshfeld 分数的规则选择及全部候选诊断。"""

    selected_rule_id: int
    candidate_diagnostics: tuple[dict[str, object], ...]


@dataclass
class TreeGeneration:
    """保存生成的粗粒化树、剩余库存和停止原因。"""

    instances: list[MonomerInstance] = field(default_factory=list)
    edges: list[PolymerEdge] = field(default_factory=list)
    remaining_inventory: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""


def select_reaction_rule(
    candidate_rule_ids: Sequence[int],
    sites_by_rule: dict[int, list[list[SiteRecord]]],
    templates: Sequence[MonomerTemplate],
    reaction_rule_id: int | None = None,
) -> RuleSelection:
    """
    按最高有效 Hirshfeld 位点对分数选择一条亲电–亲核规则。

    参数:
        candidate_rule_ids: SMiPoly 类别匹配得到的白名单候选。
        sites_by_rule: ``rule_id -> 每个输入单体的位点列表``。
        templates: 按输入顺序的单体模板，库存用于排除无实例的位点对。
        reaction_rule_id: 用户可选的显式规则；仍必须是实际候选且分数有效。

    返回值:
        ``RuleSelection``，含被选规则和每个候选的最高位点对证据。

    关键流程:
        只在模板合法位点间计算 ``max(q_e,0)*max(-q_n,0)``；排序依次使用分数、
        规则 ID、输入顺序和原子索引，确保确定性决胜。

    可能报错:
        无候选、指定规则不匹配或所有候选分数为零时抛出 ``ValueError``。
    """

    candidates = sorted(set(int(rule_id) for rule_id in candidate_rule_ids))
    if not candidates:
        raise ValueError("没有 SMiPoly 亲电–亲核白名单候选。")
    if reaction_rule_id is not None and reaction_rule_id not in candidates:
        raise ValueError(f"指定规则 {reaction_rule_id} 不是当前输入的 SMiPoly 白名单候选。")

    ranked: list[tuple[tuple[float, int, int, int, int, int], dict[str, object]]] = []
    for rule_id in candidates:
        site_lists = sites_by_rule.get(rule_id, [])
        best: tuple[float, int, int, SiteRecord, SiteRecord] | None = None
        for left_index, left_sites in enumerate(site_lists):
            if templates[left_index].inventory <= 0:
                continue
            for right_index, right_sites in enumerate(site_lists):
                if templates[right_index].inventory <= 0:
                    continue
                # 同一模板需要至少两个实例才能建立实例间边；禁止把同一分子内部闭环当成聚合。
                if left_index == right_index and templates[left_index].inventory < 2:
                    continue
                for left in left_sites:
                    for right in right_sites:
                        if not sites_are_compatible(rule_id, left, right, left_index, right_index):
                            continue
                        score = pair_score(left, right)
                        key = (
                            -score,
                            left_index,
                            right_index,
                            left.orig_atom_index,
                            right.orig_atom_index,
                        )
                        if best is None:
                            best = (score, left_index, right_index, left, right)
                        else:
                            current_key = (
                                -best[0],
                                best[1],
                                best[2],
                                best[3].orig_atom_index,
                                best[4].orig_atom_index,
                            )
                            if key < current_key:
                                best = (score, left_index, right_index, left, right)

        if best is None:
            diagnostic = {
                "rule_id": rule_id,
                "rule_name": RULES[rule_id].name,
                "valid_site_pair": False,
                "max_reaction_score": 0.0,
            }
            rank_key = (math.inf, rule_id, math.inf, math.inf, math.inf, math.inf)
        else:
            score, left_index, right_index, left, right = best
            diagnostic = {
                "rule_id": rule_id,
                "rule_name": RULES[rule_id].name,
                "valid_site_pair": score > 0.0,
                "max_reaction_score": score,
                "left": {
                    "monomer": templates[left_index].name,
                    "input_index": left_index,
                    "site_id": left.site_id,
                    "orig_atom_index": left.orig_atom_index,
                    "role": left.role,
                    "hirshfeld_charge": left.hirshfeld_charge,
                },
                "right": {
                    "monomer": templates[right_index].name,
                    "input_index": right_index,
                    "site_id": right.site_id,
                    "orig_atom_index": right.orig_atom_index,
                    "role": right.role,
                    "hirshfeld_charge": right.hirshfeld_charge,
                },
            }
            rank_key = (-score, rule_id, left_index, right_index, left.orig_atom_index, right.orig_atom_index)
        ranked.append((rank_key, diagnostic))

    diagnostics = tuple(item[1] for item in sorted(ranked, key=lambda item: item[1]["rule_id"]))
    if reaction_rule_id is not None:
        chosen = next(item for item in ranked if int(item[1]["rule_id"]) == reaction_rule_id)
        if not bool(chosen[1]["valid_site_pair"]):
            raise ValueError(f"指定规则 {reaction_rule_id} 没有正的有效 Hirshfeld 位点对分数。")
        return RuleSelection(reaction_rule_id, diagnostics)

    valid = [item for item in ranked if bool(item[1]["valid_site_pair"])]
    if not valid:
        raise ValueError("所有 SMiPoly 白名单候选的有效 Hirshfeld 位点对分数均为 0。")
    selected = min(valid, key=lambda item: item[0])
    return RuleSelection(int(selected[1]["rule_id"]), diagnostics)


@dataclass(frozen=True)
class _Event:
    """内部生长事件：已有树位点与一种库存单体模板位点反应。"""

    host_instance_index: int
    host_site: SiteRecord
    pool_template_index: int
    pool_site: SiteRecord
    score: float


def simulate_tree(
    templates: Sequence[MonomerTemplate],
    rule_id: int,
    random_seed: int,
) -> TreeGeneration:
    """
    从一个种子节点逐步加入库存单体，生成连通无环树。

    参数:
        templates: 已换成选定规则位点/PSMILES 的单体模板。
        rule_id: 唯一选定的白名单规则。
        random_seed: 通道内等价位点选择的固定随机种子。

    返回值:
        ``TreeGeneration``，包含实际实例、N-1 条有向生长边、剩余库存和停止原因。

    关键流程:
        通道权重严格沿用现有语义：``log(weight)=score+ln(n_alpha)+ln(n_beta)``；
        贪心选择最大通道，再用固定种子在该通道的等价已有位点间选择。每步只增加一个
        新节点和一条粗边，因此不会产生分子内闭环或周期网络。

    可能报错:
        无可参与模板、位点被重复使用或库存状态异常时抛出异常。
    """

    if rule_id not in RULES:
        raise ValueError(f"未知规则 {rule_id}。")
    rng = random.Random(int(random_seed))
    remaining = {template.name: int(template.inventory) for template in templates}

    def template_participates(index: int) -> bool:
        if remaining[templates[index].name] <= 0 or not templates[index].sites:
            return False
        for other_index, other in enumerate(templates):
            if remaining[other.name] <= 0:
                continue
            if index == other_index and remaining[other.name] < 2:
                continue
            if any(
                sites_are_compatible(rule_id, left, right, index, other_index)
                and pair_score(left, right) > 0
                for left in templates[index].sites
                for right in other.sites
            ):
                return True
        return False

    seed_candidates = [index for index in range(len(templates)) if template_participates(index)]
    if not seed_candidates:
        raise ValueError("没有可作为树种子的正分数反应单体。")
    seed_template_index = seed_candidates[0]

    instances: list[MonomerInstance] = [
        MonomerInstance(instance_id="M0001", template_index=seed_template_index, growth_index=0)
    ]
    remaining[templates[seed_template_index].name] -= 1
    edges: list[PolymerEdge] = []
    target_nodes = sum(template.inventory for template in templates)
    stop_reason = "target_inventory_exhausted"

    while len(instances) < target_nodes:
        events: list[_Event] = []
        for host_index, instance in enumerate(instances):
            host_template = templates[instance.template_index]
            for host_site in host_template.sites:
                if host_site.site_id in instance.used_sites:
                    continue
                for pool_index, pool_template in enumerate(templates):
                    if remaining.get(pool_template.name, 0) <= 0:
                        continue
                    for pool_site in pool_template.sites:
                        if not sites_are_compatible(
                            rule_id,
                            host_site,
                            pool_site,
                            instance.template_index,
                            pool_index,
                        ):
                            continue
                        score = pair_score(host_site, pool_site)
                        if score > 0.0:
                            events.append(_Event(host_index, host_site, pool_index, pool_site, score))
        if not events:
            stop_reason = "no_legal_positive_score_event"
            break

        # 中文注释：精确位点类型组成有向通道；同一通道中事件数就是 n_alpha。
        channels: dict[tuple[int, str, int, str], list[_Event]] = {}
        for event in events:
            host_template_index = instances[event.host_instance_index].template_index
            channel = (
                host_template_index,
                event.host_site.site_id,
                event.pool_template_index,
                event.pool_site.site_id,
            )
            channels.setdefault(channel, []).append(event)

        ranked_channels: list[tuple[tuple[float, int, int, int, int], tuple[int, str, int, str]]] = []
        for channel, channel_events in channels.items():
            representative = channel_events[0]
            n_alpha = len({(event.host_instance_index, event.host_site.site_id) for event in channel_events})
            n_beta = remaining[templates[representative.pool_template_index].name]
            log_weight = representative.score + math.log(n_alpha) + math.log(n_beta)
            host_atom = representative.host_site.orig_atom_index
            pool_atom = representative.pool_site.orig_atom_index
            ranked_channels.append(
                ((-log_weight, channel[0], channel[2], host_atom, pool_atom), channel)
            )
        _, chosen_channel = min(ranked_channels, key=lambda item: item[0])
        chosen = rng.choice(channels[chosen_channel])

        host = instances[chosen.host_instance_index]
        pool_template = templates[chosen.pool_template_index]
        new_id = f"M{len(instances) + 1:04d}"
        new_instance = MonomerInstance(
            instance_id=new_id,
            template_index=chosen.pool_template_index,
            growth_index=len(instances),
        )
        if chosen.host_site.site_id in host.used_sites:
            raise RuntimeError("dynamic_softmax_greedy 选择了已使用位点。")
        host.used_sites[chosen.host_site.site_id] = (new_id, chosen.pool_site.site_id)
        new_instance.used_sites[chosen.pool_site.site_id] = (host.instance_id, chosen.host_site.site_id)
        instances.append(new_instance)
        remaining[pool_template.name] -= 1

        electrophile = chosen.host_site if chosen.host_site.role == "electrophile" else chosen.pool_site
        nucleophile = chosen.host_site if chosen.host_site.role == "nucleophile" else chosen.pool_site
        edges.append(
            PolymerEdge(
                source_instance_id=host.instance_id,
                source_site_id=chosen.host_site.site_id,
                target_instance_id=new_id,
                target_site_id=chosen.pool_site.site_id,
                smipoly_rule_id=rule_id,
                source_hirshfeld_charge=chosen.host_site.hirshfeld_charge,
                target_hirshfeld_charge=chosen.pool_site.hirshfeld_charge,
                electrophile_hirshfeld_charge=electrophile.hirshfeld_charge,
                nucleophile_hirshfeld_charge=nucleophile.hirshfeld_charge,
                reaction_score=chosen.score,
                new_bond_type=RULES[rule_id].new_bond_type,
            )
        )

    return TreeGeneration(instances, edges, remaining, stop_reason)


def validate_tree(
    generation: TreeGeneration,
    templates: Sequence[MonomerTemplate],
    target_nodes: int,
    min_completion_ratio: float,
) -> dict[str, object]:
    """
    验证粗图连通、无环、无自环、位点唯一及完成率。

    参数:
        generation: ``simulate_tree`` 输出。
        templates: 单体模板，用于核对位点和实际组成。
        target_nodes: 目标节点数。
        min_completion_ratio: 最低完成率，取上整得到最小节点数。

    返回值:
        逐项布尔检查、实际组成、配比偏差和剩余库存组成的验证报告。

    边界情况:
        单节点树视为连通无环；正式工作流仍由完成率门槛决定能否发布。
    """

    node_ids = [instance.instance_id for instance in generation.instances]
    node_set = set(node_ids)
    adjacency = {node_id: set() for node_id in node_ids}
    used_endpoints: list[tuple[str, str]] = []
    no_self_loops = True
    endpoints_exist = True
    for edge in generation.edges:
        if edge.source_instance_id == edge.target_instance_id:
            no_self_loops = False
        if edge.source_instance_id not in node_set or edge.target_instance_id not in node_set:
            endpoints_exist = False
            continue
        adjacency[edge.source_instance_id].add(edge.target_instance_id)
        adjacency[edge.target_instance_id].add(edge.source_instance_id)
        used_endpoints.extend(
            [(edge.source_instance_id, edge.source_site_id), (edge.target_instance_id, edge.target_site_id)]
        )

    visited: set[str] = set()
    if node_ids:
        stack = [node_ids[0]]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current] - visited)
    connected = len(visited) == len(node_ids)
    edge_count_tree = len(generation.edges) == max(0, len(node_ids) - 1)
    endpoint_unique = len(used_endpoints) == len(set(used_endpoints))
    acyclic = connected and edge_count_tree

    actual_composition = {template.name: 0 for template in templates}
    for instance in generation.instances:
        actual_composition[templates[instance.template_index].name] += 1
    target_composition = {template.name: template.inventory for template in templates}
    actual_total = len(node_ids)
    required_nodes = math.ceil(float(min_completion_ratio) * int(target_nodes))
    ratio_deviation = {
        template.name: {
            "target_fraction": template.inventory / target_nodes,
            "actual_fraction": (actual_composition[template.name] / actual_total) if actual_total else 0.0,
            "absolute_deviation": abs(
                (actual_composition[template.name] / actual_total if actual_total else 0.0)
                - template.inventory / target_nodes
            ),
        }
        for template in templates
    }
    checks = {
        "node_ids_unique": len(node_ids) == len(node_set),
        "edge_endpoints_exist": endpoints_exist,
        "no_self_loops": no_self_loops,
        "site_used_at_most_once": endpoint_unique,
        "connected": connected,
        "acyclic": acyclic,
        "edge_count_is_n_minus_1": edge_count_tree,
        "completion_threshold_met": actual_total >= required_nodes,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "target_nodes": int(target_nodes),
        "required_nodes": required_nodes,
        "actual_nodes": actual_total,
        "completion_ratio": actual_total / target_nodes,
        "target_composition": target_composition,
        "actual_composition": actual_composition,
        "ratio_deviation": ratio_deviation,
        "remaining_inventory": generation.remaining_inventory,
        "stop_reason": generation.stop_reason,
    }
