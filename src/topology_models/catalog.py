"""十类文献原型启发的理想化有限拓扑配置。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .models import MonomerSpec, TopologyEdge, TopologyModel, TopologyNode


def _edge(
    number: int,
    source_instance_id: str,
    source_site_id: str,
    target_instance_id: str,
    target_site_id: str,
    reaction_schema_id: str,
    *,
    reaction_event_id: str | None = None,
    event_bond_role: str | None = None,
) -> TopologyEdge:
    """用连续边号构造一条显式端口边。"""

    return TopologyEdge(
        edge_id=f"E{number:03d}",
        source_instance_id=source_instance_id,
        source_site_id=source_site_id,
        target_instance_id=target_instance_id,
        target_site_id=target_site_id,
        reaction_schema_id=reaction_schema_id,
        reaction_event_id=reaction_event_id,
        event_bond_role=event_bond_role,
    )


def linear_polyurethane() -> TopologyModel:
    """返回羟基封端 ``BDO-(MDI-BDO)10`` 的 21 节点路径。"""

    monomers = [
        MonomerSpec(
            "MDI",
            "4,4'-methylenediphenyl diisocyanate (MDI)",
            "O=C=Nc1ccc(Cc2ccc(N=C=O)cc2)cc1",
            ("urethane_rule_111",),
            "electrophile_orange",
            "electrophile",
        ),
        MonomerSpec(
            "BDO",
            "1,4-butanediol (BDO)",
            "OCCCCO",
            ("urethane_rule_111",),
            "nucleophile_green",
            "nucleophile",
        ),
    ]
    nodes = [
        TopologyNode(
            instance_id=f"L{index + 1:02d}",
            monomer_id="BDO" if index % 2 == 0 else "MDI",
            generation=index,
            backbone=True,
        )
        for index in range(21)
    ]
    counters: dict[str, Counter[str]] = {node.instance_id: Counter() for node in nodes}
    edges: list[TopologyEdge] = []
    for index, (left, right) in enumerate(zip(nodes, nodes[1:]), start=1):
        left_role = "N" if left.monomer_id == "BDO" else "E"
        right_role = "N" if right.monomer_id == "BDO" else "E"
        counters[left.instance_id][left_role] += 1
        counters[right.instance_id][right_role] += 1
        edges.append(
            _edge(
                index,
                left.instance_id,
                f"urethane_rule_111:{left_role}{counters[left.instance_id][left_role]}",
                right.instance_id,
                f"urethane_rule_111:{right_role}{counters[right.instance_id][right_role]}",
                "urethane_rule_111",
            )
        )
    model = TopologyModel(
        polymer_id="linear_polyurethane",
        display_name="Hydroxyl-terminated linear polyurethane",
        topology_type="linear_path",
        parameters={"formula": "BDO-(MDI-BDO)10", "mdi_units": 10, "bdo_units": 11},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[],
        scientific_scope_note="Idealized finite, hydroxyl-terminated polyurethane path; not an equilibrated sample.",
    )
    model.validate_basic()
    return model


def bis_mpa_g3_dendrimer() -> TopologyModel:
    """返回季戊四醇核和 28 个 bis-MPA 的完整 G3 树。"""

    schema = "esterification_cooh_oh"
    monomers = [
        MonomerSpec(
            "PENTA_CORE",
            "Pentaerythritol core",
            "C(CO)(CO)(CO)CO",
            (schema,),
            "core_blue",
            "core",
        ),
        MonomerSpec(
            "BIS_MPA",
            "2,2-bis(hydroxymethyl)propionic acid (bis-MPA)",
            "CC(CO)(CO)C(=O)O",
            (schema,),
            "ab2_violet",
            "AB2 branch monomer",
        ),
    ]
    core = TopologyNode("D000", "PENTA_CORE", generation=0)
    nodes = [core]
    edges: list[TopologyEdge] = []
    parents: list[tuple[TopologyNode, int]] = [(core, -1)]
    next_index = 1
    edge_index = 1
    for generation, children_per_parent in ((1, 4), (2, 2), (3, 2)):
        children: list[tuple[TopologyNode, int]] = []
        for parent_position, (parent, inherited_arm) in enumerate(parents):
            for child_position in range(children_per_parent):
                arm = child_position + 1 if generation == 1 else inherited_arm
                child = TopologyNode(
                    instance_id=f"D{next_index:03d}",
                    monomer_id="BIS_MPA",
                    generation=generation,
                    arm=arm,
                    sidechain=True,
                )
                next_index += 1
                nodes.append(child)
                children.append((child, arm))
                parent_port = child_position + 1
                edges.append(
                    _edge(
                        edge_index,
                        parent.instance_id,
                        f"{schema}:N{parent_port}",
                        child.instance_id,
                        f"{schema}:E1",
                        schema,
                    )
                )
                edge_index += 1
        parents = children
    model = TopologyModel(
        polymer_id="bis_mpa_g3_dendrimer",
        display_name="Pentaerythritol-core bis-MPA G3 dendrimer",
        topology_type="complete_generation_dendrimer",
        parameters={"generation": 3, "shell_counts": [1, 4, 8, 16], "bis_mpa_units": 28},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[
            {
                "label": "Polymer Chemistry architecture prototype",
                "url": "https://pubs.rsc.org/en/content/articlehtml/2021/py/d0py01070g",
            }
        ],
        scientific_scope_note="Complete G3 architecture consistent with a 28-bis-MPA pentaerythritol-core prototype.",
    )
    model.validate_basic()
    return model


def aromatic_oligoamide_macrocycle() -> TopologyModel:
    """返回 8 个 TPC 与 8 个 mPDA 交替连接的单环。"""

    schema = "amide_rule_108"
    monomers = [
        MonomerSpec(
            "TPC",
            "Terephthaloyl chloride (TPC)",
            "O=C(Cl)c1ccc(C(=O)Cl)cc1",
            (schema,),
            "electrophile_orange",
            "electrophile",
        ),
        MonomerSpec(
            "MPDA",
            "m-Phenylenediamine (mPDA)",
            "Nc1cccc(N)c1",
            (schema,),
            "nucleophile_green",
            "nucleophile",
        ),
    ]
    tpc_nodes = [TopologyNode(f"C_T{i + 1:02d}", "TPC", generation=0) for i in range(8)]
    pda_nodes = [TopologyNode(f"C_N{i + 1:02d}", "MPDA", generation=0) for i in range(8)]
    nodes: list[TopologyNode] = []
    for tpc, pda in zip(tpc_nodes, pda_nodes):
        nodes.extend((tpc, pda))
    edges: list[TopologyEdge] = []
    edge_index = 1
    for index in range(8):
        edges.append(
            _edge(
                edge_index,
                tpc_nodes[index].instance_id,
                f"{schema}:E1",
                pda_nodes[index].instance_id,
                f"{schema}:N1",
                schema,
            )
        )
        edge_index += 1
        edges.append(
            _edge(
                edge_index,
                pda_nodes[index].instance_id,
                f"{schema}:N2",
                tpc_nodes[(index + 1) % 8].instance_id,
                f"{schema}:E2",
                schema,
            )
        )
        edge_index += 1
    model = TopologyModel(
        polymer_id="aromatic_oligoamide_macrocycle_16",
        display_name="Alternating aromatic oligoamide 16-mer macrocycle",
        topology_type="single_cycle",
        parameters={"ring_units": 16, "tpc_units": 8, "mpda_units": 8},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[{"label": "JACS oligoamide macrocycle prototype", "url": "https://doi.org/10.1021/ja807935y"}],
        scientific_scope_note="Idealized 16-unit alternating aromatic oligoamide ring; paper substituents are not reproduced.",
    )
    model.validate_basic()
    return model


def four_arm_pcl_star() -> TopologyModel:
    """返回季戊四醇核和四条等长 ε-CL 四聚臂。"""

    schema = "lactone_rop"
    monomers = [
        MonomerSpec(
            "PENTA_CORE",
            "Pentaerythritol core",
            "C(CO)(CO)(CO)CO",
            (schema,),
            "core_blue",
            "core initiator",
        ),
        MonomerSpec(
            "ECL",
            "ε-Caprolactone (ε-CL)",
            "O=C1OCCCCC1",
            (schema,),
            "ab2_violet",
            "lactone propagation monomer",
        ),
    ]
    core = TopologyNode("S_CORE", "PENTA_CORE", generation=0)
    nodes = [core]
    edges: list[TopologyEdge] = []
    edge_index = 1
    for arm in range(1, 5):
        previous = core
        for position in range(1, 5):
            child = TopologyNode(
                instance_id=f"S_A{arm}_{position:02d}",
                monomer_id="ECL",
                generation=position,
                arm=arm,
                sidechain=True,
            )
            nodes.append(child)
            source_site = f"{schema}:N{arm}" if position == 1 else f"{schema}:N1"
            edges.append(
                _edge(
                    edge_index,
                    previous.instance_id,
                    source_site,
                    child.instance_id,
                    f"{schema}:E1",
                    schema,
                )
            )
            edge_index += 1
            previous = child
    model = TopologyModel(
        polymer_id="four_arm_pcl_star",
        display_name="Pentaerythritol-initiated four-arm PCL star",
        topology_type="star",
        parameters={"arm_count": 4, "arm_length": 4, "epsilon_cl_units": 16},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[{"label": "Four-arm PCL prototype", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9062625/"}],
        scientific_scope_note="Idealized four-arm finite PCL model; arm length is a visualization choice.",
    )
    model.validate_basic()
    return model


def pi_g_pcl_bottlebrush() -> TopologyModel:
    """返回 9 节点 PI 主链和五条等长 ε-CL 四聚侧链。"""

    backbone_schema = "polyimide_rule_110"
    side_schema = "lactone_rop"
    monomers = [
        MonomerSpec(
            "PMDA",
            "Pyromellitic dianhydride (PMDA)",
            "O=C1OC(=O)c2cc3C(=O)OC(=O)c3cc12",
            (backbone_schema,),
            "electrophile_orange",
            "backbone electrophile",
        ),
        MonomerSpec(
            "DAP",
            "2,4-Diaminophenol",
            "Nc1ccc(O)c(N)c1",
            (backbone_schema, side_schema),
            "nucleophile_green",
            "backbone nucleophile and graft initiator",
        ),
        MonomerSpec(
            "ECL",
            "ε-Caprolactone (ε-CL)",
            "O=C1OCCCCC1",
            (side_schema,),
            "ab2_violet",
            "sidechain lactone monomer",
        ),
    ]
    dap_nodes = [TopologyNode(f"B_D{i + 1:02d}", "DAP", generation=0, backbone=True) for i in range(5)]
    pmda_nodes = [TopologyNode(f"B_P{i + 1:02d}", "PMDA", generation=0, backbone=True) for i in range(4)]
    nodes: list[TopologyNode] = []
    for index in range(4):
        nodes.extend((dap_nodes[index], pmda_nodes[index]))
    nodes.append(dap_nodes[-1])
    edges: list[TopologyEdge] = []
    edge_index = 1
    for index, pmda in enumerate(pmda_nodes):
        edges.append(
            _edge(
                edge_index,
                dap_nodes[index].instance_id,
                f"{backbone_schema}:N1",
                pmda.instance_id,
                f"{backbone_schema}:E1",
                backbone_schema,
            )
        )
        edge_index += 1
        edges.append(
            _edge(
                edge_index,
                pmda.instance_id,
                f"{backbone_schema}:E2",
                dap_nodes[index + 1].instance_id,
                f"{backbone_schema}:N2",
                backbone_schema,
            )
        )
        edge_index += 1

    for graft, dap in enumerate(dap_nodes, start=1):
        previous = dap
        for position in range(1, 5):
            child = TopologyNode(
                instance_id=f"B_G{graft}_{position:02d}",
                monomer_id="ECL",
                generation=position,
                arm=graft,
                sidechain=True,
            )
            nodes.append(child)
            edges.append(
                _edge(
                    edge_index,
                    previous.instance_id,
                    f"{side_schema}:N1",
                    child.instance_id,
                    f"{side_schema}:E1",
                    side_schema,
                )
            )
            edge_index += 1
            previous = child
    model = TopologyModel(
        polymer_id="pi_g_pcl_bottlebrush",
        display_name="PI-g-PCL bottlebrush finite model",
        topology_type="bottlebrush",
        parameters={
            "backbone_nodes": 9,
            "graft_points": 5,
            "sidechain_length": 4,
            "epsilon_cl_units": 20,
        },
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[{"label": "PI-g-PCL molecular brush prototype", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8513028/"}],
        scientific_scope_note="Idealized finite PI-g-PCL brush; backbone chemistry and sidechain length are simplified.",
    )
    model.validate_basic()
    return model


def hyperbranched_bis_mpa_31() -> TopologyModel:
    """返回一个固定的 31 节点非满代 bis-MPA 超支化聚酯树。"""

    schema = "esterification_cooh_oh"
    monomers = [
        MonomerSpec(
            "BIS_MPA",
            "2,2-bis(hydroxymethyl)propionic acid (bis-MPA)",
            "CC(CO)(CO)C(=O)O",
            (schema,),
            "ab2_violet",
            "AB2 hyperbranched monomer",
        )
    ]
    children = {
        0: (1, 2), 1: (3, 4), 2: (5,), 3: (6, 7), 4: (8,),
        5: (9, 10), 6: (11,), 7: (12, 13), 8: (14,), 9: (15, 16),
        10: (17,), 11: (18,), 12: (19, 20), 13: (21,), 14: (22, 23),
        15: (24,), 16: (25,), 17: (26, 27), 18: (28,), 19: (29, 30),
    }
    parent_of = {child: parent for parent, values in children.items() for child in values}
    depths = {0: 0}
    for index in range(1, 31):
        depths[index] = depths[parent_of[index]] + 1
    branch_state = {
        index: ("D" if len(children.get(index, ())) == 2 else "L" if len(children.get(index, ())) == 1 else "T")
        for index in range(31)
    }
    nodes = [
        TopologyNode(
            f"H{index:02d}",
            "BIS_MPA",
            generation=depths[index],
            sidechain=True,
            branch_state=branch_state[index],
        )
        for index in range(31)
    ]
    edges: list[TopologyEdge] = []
    edge_number = 1
    for parent, child_indices in children.items():
        for child_position, child in enumerate(child_indices, start=1):
            edges.append(
                _edge(
                    edge_number,
                    f"H{parent:02d}",
                    f"{schema}:N{child_position}",
                    f"H{child:02d}",
                    f"{schema}:E1",
                    schema,
                )
            )
            edge_number += 1
    model = TopologyModel(
        polymer_id="hyperbranched_bis_mpa_31",
        display_name="Irregular 31-unit hyperbranched bis-MPA polyester",
        topology_type="irregular_hyperbranched",
        parameters={
            "unit_count": 31,
            "D_L_T_counts": {"D": 10, "L": 10, "T": 11},
            "degree_of_branching": 21.0 / 31.0,
            "shell_counts": [1, 2, 3, 5, 7, 10, 3],
            "maximum_depth": 6,
        },
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[
            {
                "label": "bis-MPA hyperbranched polyester prototype",
                "url": "https://doi.org/10.1021/ma951084c",
            }
        ],
        scientific_scope_note="One explicit irregular AB2 tree realization; not a statistical ensemble.",
    )
    model.validate_basic()
    return model


def amide_scnp_28() -> TopologyModel:
    """返回 24 节点酰胺主链和四个分子内长程交联剂组成的 SCNP。"""

    schema = "amide_rule_108"
    monomers = [
        MonomerSpec(
            "TPC", "Terephthaloyl chloride (TPC)",
            "O=C(Cl)c1ccc(C(=O)Cl)cc1", (schema,),
            "electrophile_orange", "backbone electrophile",
        ),
        MonomerSpec(
            "TAMB", "1,3,5-Triaminobenzene",
            "Nc1cc(N)cc(N)c1", (schema,),
            "nucleophile_green", "backbone nucleophile with pendant amine",
        ),
        MonomerSpec(
            "ADIPOYL_CL", "Adipoyl chloride",
            "O=C(Cl)CCCCC(=O)Cl", (schema,),
            "crosslinker_rose", "intramolecular crosslinker",
        ),
    ]
    backbone = [
        TopologyNode(f"B{index:02d}", "TPC" if index % 2 == 0 else "TAMB", backbone=True)
        for index in range(24)
    ]
    crosslinkers = [TopologyNode(f"X{index:02d}", "ADIPOYL_CL") for index in range(4)]
    nodes = [*backbone, *crosslinkers]
    counters = {node.instance_id: Counter() for node in nodes}
    edges: list[TopologyEdge] = []
    for left, right in zip(backbone, backbone[1:]):
        left_role = "E" if left.monomer_id == "TPC" else "N"
        right_role = "E" if right.monomer_id == "TPC" else "N"
        counters[left.instance_id][left_role] += 1
        counters[right.instance_id][right_role] += 1
        edges.append(
            _edge(
                len(edges) + 1,
                left.instance_id,
                f"{schema}:{left_role}{counters[left.instance_id][left_role]}",
                right.instance_id,
                f"{schema}:{right_role}{counters[right.instance_id][right_role]}",
                schema,
            )
        )
    pairs = ((1, 13), (3, 17), (5, 21), (7, 23))
    for crosslinker, (first, second) in zip(crosslinkers, pairs):
        for port, backbone_index in enumerate((first, second), start=1):
            edges.append(
                _edge(
                    len(edges) + 1,
                    crosslinker.instance_id,
                    f"{schema}:E{port}",
                    f"B{backbone_index:02d}",
                    f"{schema}:N3",
                    schema,
                )
            )
    model = TopologyModel(
        polymer_id="amide_scnp_28",
        display_name="Four-loop aromatic amide single-chain nanoparticle",
        topology_type="single_chain_nanoparticle",
        parameters={"backbone_nodes": 24, "crosslinker_nodes": 4, "independent_loops": 4, "crosslink_pairs": pairs},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[
            {"label": "SCNP long-range loop prototype", "url": "https://doi.org/10.1021/acs.macromol.1c02071"}
        ],
        scientific_scope_note="Idealized finite intramolecularly crosslinked chain; not a solution conformation.",
    )
    model.validate_basic()
    return model


def dgeba_tren_epoxy_network_25() -> TopologyModel:
    """返回 15 个 DGEBA 和 10 个 TREN 组成的五环部分固化网络。"""

    schema = "epoxide_amine_ring_opening"
    monomers = [
        MonomerSpec(
            "DGEBA", "Bisphenol A diglycidyl ether (DGEBA)",
            "CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4",
            (schema,), "electrophile_orange", "diepoxide electrophile",
        ),
        MonomerSpec(
            "TREN", "Tris(2-aminoethyl)amine (TREN)",
            "NCCN(CCN)CCN", (schema,),
            "nucleophile_green", "trifunctional primary-amine curing node",
        ),
    ]
    epoxides = [TopologyNode(f"E{index:02d}", "DGEBA") for index in range(15)]
    amines = [TopologyNode(f"A{index:02d}", "TREN") for index in range(10)]
    nodes = [*epoxides, *amines]
    edges: list[TopologyEdge] = []
    for index in range(10):
        edges.append(_edge(len(edges) + 1, f"E{index:02d}", f"{schema}:E1", f"A{index:02d}", f"{schema}:N1", schema))
        edges.append(_edge(len(edges) + 1, f"E{index:02d}", f"{schema}:E2", f"A{(index + 1) % 10:02d}", f"{schema}:N2", schema))
    extra_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    for offset, (first, second) in enumerate(extra_pairs, start=10):
        edges.append(_edge(len(edges) + 1, f"E{offset:02d}", f"{schema}:E1", f"A{first:02d}", f"{schema}:N3", schema))
        edges.append(_edge(len(edges) + 1, f"E{offset:02d}", f"{schema}:E2", f"A{second:02d}", f"{schema}:N3", schema))
    edges.append(_edge(len(edges) + 1, "E14", f"{schema}:E1", "A08", f"{schema}:N3", schema))
    model = TopologyModel(
        polymer_id="dgeba_tren_epoxy_network_25",
        display_name="Partially cured DGEBA–TREN epoxy network",
        topology_type="crosslinked_epoxy_network",
        parameters={"dgeba_units": 15, "tren_units": 10, "independent_cycles": 5, "unreacted_epoxide_sites": 1, "unreacted_primary_amine_sites": 1},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[
            {"label": "Amine-cured epoxide network prototype", "url": "https://doi.org/10.1021/ma991831w"}
        ],
        scientific_scope_note="Idealized partial cure; secondary-amine reaction and etherification are not simulated.",
    )
    model.validate_basic()
    return model


def pi_g_bis_mpa_g3_dendronized_44() -> TopologyModel:
    """返回 9 节点 PI 主链和五个完整 G3 bis-MPA dendron。"""

    backbone_schema = "polyimide_rule_110"
    dendron_schema = "esterification_cooh_oh"
    monomers = [
        MonomerSpec(
            "PMDA", "Pyromellitic dianhydride (PMDA)",
            "O=C1OC(=O)c2cc3C(=O)OC(=O)c3cc12", (backbone_schema,),
            "electrophile_orange", "backbone electrophile",
        ),
        MonomerSpec(
            "DAP", "2,4-Diaminophenol",
            "Nc1ccc(O)c(N)c1", (backbone_schema, dendron_schema),
            "nucleophile_green", "backbone nucleophile and dendron graft point",
        ),
        MonomerSpec(
            "BIS_MPA", "2,2-bis(hydroxymethyl)propionic acid (bis-MPA)",
            "CC(CO)(CO)C(=O)O", (dendron_schema,),
            "ab2_violet", "dendron AB2 monomer",
        ),
    ]
    dap = [TopologyNode(f"DP_D{index}", "DAP", backbone=True) for index in range(5)]
    pmda = [TopologyNode(f"DP_P{index}", "PMDA", backbone=True) for index in range(4)]
    nodes: list[TopologyNode] = []
    for index in range(4):
        nodes.extend((dap[index], pmda[index]))
    nodes.append(dap[-1])
    edges: list[TopologyEdge] = []
    for index, electrophile in enumerate(pmda):
        edges.append(_edge(len(edges) + 1, dap[index].instance_id, f"{backbone_schema}:N1", electrophile.instance_id, f"{backbone_schema}:E1", backbone_schema))
        edges.append(_edge(len(edges) + 1, electrophile.instance_id, f"{backbone_schema}:E2", dap[index + 1].instance_id, f"{backbone_schema}:N2", backbone_schema))
    for arm, graft in enumerate(dap, start=1):
        generation_one = TopologyNode(f"DP_A{arm}_G1_1", "BIS_MPA", generation=1, arm=arm, sidechain=True)
        generation_two = [TopologyNode(f"DP_A{arm}_G2_{position}", "BIS_MPA", generation=2, arm=arm, sidechain=True) for position in range(1, 3)]
        generation_three = [TopologyNode(f"DP_A{arm}_G3_{position}", "BIS_MPA", generation=3, arm=arm, sidechain=True) for position in range(1, 5)]
        nodes.extend((generation_one, *generation_two, *generation_three))
        edges.append(_edge(len(edges) + 1, graft.instance_id, f"{dendron_schema}:N1", generation_one.instance_id, f"{dendron_schema}:E1", dendron_schema))
        for position, child in enumerate(generation_two, start=1):
            edges.append(_edge(len(edges) + 1, generation_one.instance_id, f"{dendron_schema}:N{position}", child.instance_id, f"{dendron_schema}:E1", dendron_schema))
        for parent_position, parent in enumerate(generation_two):
            for child_position in range(2):
                child = generation_three[parent_position * 2 + child_position]
                edges.append(_edge(len(edges) + 1, parent.instance_id, f"{dendron_schema}:N{child_position + 1}", child.instance_id, f"{dendron_schema}:E1", dendron_schema))
    model = TopologyModel(
        polymer_id="pi_g_bis_mpa_g3_dendronized_44",
        display_name="PI backbone with five complete G3 bis-MPA dendrons",
        topology_type="dendronized_polymer",
        parameters={"backbone_nodes": 9, "dendron_count": 5, "dendron_generation": 3, "dendron_shell_counts": [1, 2, 4]},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[
            {"label": "bis-MPA dendronized polymer prototype", "url": "https://doi.org/10.1002/pola.20846"}
        ],
        scientific_scope_note="Idealized PI-g-bis-MPA finite hierarchy; not the exact reported ATRP backbone.",
    )
    model.validate_basic()
    return model


def bbl_ladder_8() -> TopologyModel:
    """返回 8 单体、7 复合事件和 21 条平行 C–N 子边的 trans-BBL。"""

    schema = "bbl_perinone_cyclocondensation"
    monomers = [
        MonomerSpec(
            "TAB4", "1,2,4,5-Tetraaminobenzene",
            "Nc1cc(N)c(N)cc1N", (schema,),
            "nucleophile_green", "two ortho-diamine pair nucleophile",
        ),
        MonomerSpec(
            "NTCDA", "1,4,5,8-Naphthalenetetracarboxylic dianhydride",
            "C1=CC2=C3C(=CC=C4C3=C1C(=O)OC4=O)C(=O)OC2=O", (schema,),
            "electrophile_orange", "two cyclic-anhydride electrophile",
        ),
    ]
    nodes = [TopologyNode(f"Q{index}", "TAB4" if index % 2 == 0 else "NTCDA", backbone=True) for index in range(8)]
    roles = ("lost_c_imine_n", "lost_c_bridge_n", "retained_c_bridge_n")
    edges: list[TopologyEdge] = []
    for adjacency in range(7):
        left = nodes[adjacency]
        right = nodes[adjacency + 1]
        left_role = "N" if left.monomer_id == "TAB4" else "E"
        right_role = "N" if right.monomer_id == "TAB4" else "E"
        for rank, role in enumerate(roles, start=1):
            edges.append(
                _edge(
                    len(edges) + 1,
                    left.instance_id,
                    f"{schema}:{left_role}{rank + 3}",
                    right.instance_id,
                    f"{schema}:{right_role}{rank}",
                    schema,
                    reaction_event_id=f"R{adjacency + 1:02d}",
                    event_bond_role=role,
                )
            )
    model = TopologyModel(
        polymer_id="bbl_ladder_8",
        display_name="Eight-monomer trans-BBL ladder oligomer",
        topology_type="bbl_ladder_multigraph",
        parameters={"monomer_nodes": 8, "reaction_events": 7, "parallel_edges_per_event": 3, "expected_formula": "C80H28N16O10", "allow_parallel_edges": True},
        monomers=monomers,
        nodes=nodes,
        edges=edges,
        references=[
            {"label": "BBL ladder polymer prototype", "url": "https://doi.org/10.1016/S0032-3861(98)00777-0"},
            {"label": "trans-perinone structure fixture", "url": "https://pubchem.ncbi.nlm.nih.gov/compound/78141"},
        ],
        scientific_scope_note="Idealized finite trans-BBL oligomer with grouped three-bond cyclocondensation events.",
    )
    model.validate_basic()
    return model


def builtin_models() -> list[TopologyModel]:
    """按固定发布顺序返回原五类和新增五类显式拓扑模型。"""

    return [
        linear_polyurethane(),
        bis_mpa_g3_dendrimer(),
        aromatic_oligoamide_macrocycle(),
        four_arm_pcl_star(),
        pi_g_pcl_bottlebrush(),
        hyperbranched_bis_mpa_31(),
        amide_scnp_28(),
        dgeba_tren_epoxy_network_25(),
        pi_g_bis_mpa_g3_dendronized_44(),
        bbl_ladder_8(),
    ]


def builtin_config_payload() -> dict[str, Any]:
    """返回可写入统一 JSON 输入文件的十模型 V2 配置。"""

    return {
        "schema_version": "topology_models_explicit_v2",
        "models": [model.as_dict() for model in builtin_models()],
    }
