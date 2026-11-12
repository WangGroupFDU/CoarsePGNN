"""候选配方适配层的化学计量与实际模型输入检查；夹具为合成示例。"""
import copy
import unittest

import torch

from coarsepgnn.data import GraphFeaturizer
from coarsepgnn.schema import package_schema
from coarsepgnn_inverse.graph import recipe_to_complete_graph


def component_fixture() -> list[dict]:
    """返回仅用于接口验证的 D/P/E 模板，不将夹具当作实测或合成证据。"""
    return [
        {"component_id": "D", "component_role": "diisocyanate", "functionality": 2,
         "canonical_smiles": "O=C=Nc1ccc(Cc2ccc(N=C=O)cc2)cc1",
         "structure_kind": "exact_smiles", "molecular_weight_g_mol": 250.25},
        {"component_id": "P", "component_role": "polyol", "functionality": 2,
         "canonical_repeat_unit_psmiles": "*CCO*", "mn_g_mol": 1000.0,
         "structure_kind": "repeat_unit"},
        {"component_id": "E", "component_role": "hydroxyl_chain_extender", "functionality": 2,
         "canonical_smiles": "OCCO", "molecular_weight_g_mol": 62.068,
         "structure_kind": "exact_smiles"},
    ]


class RecipeGraphTest(unittest.TestCase):
    """验证计量、显式反应边及连续比例经过真实 featurizer 的传递。"""

    def test_none_and_source_immutability(self) -> None:
        """NONE 保持零扩链剂，输入模板不被改写，实际图能进入预测特征器。"""
        components = component_fixture()[:2]
        before = copy.deepcopy(components)
        graph = recipe_to_complete_graph(components, 1.1, 0.0)
        self.assertEqual(components, before)
        self.assertEqual(len(graph["nodes"]), 100)
        self.assertEqual(len(graph["edges"]), 99)
        self.assertEqual(graph["stoichiometry"]["molecule_ratios"], [1.1, 1.0])
        self.assertTrue(all(edge["smipoly_rule_id"] == 111 for edge in graph["edges"]))
        data = GraphFeaturizer.from_sample_schema(package_schema()).prepare_graph(graph).data
        self.assertEqual(data.node_num.shape[0], 100)
        self.assertTrue(bool((data.site_token_id != 0).all()), "NCO/OH 位点不能落入 UNK")

    def test_continuous_ratio_reaches_model(self) -> None:
        """保持离散节点和拓扑时，小幅配比变化仍改变真实模型数值特征。"""
        first = recipe_to_complete_graph(component_fixture(), 1.05, 0.30)
        second = recipe_to_complete_graph(component_fixture(), 1.05, 0.3001)
        self.assertEqual(first["edges"], second["edges"])
        featurizer = GraphFeaturizer.from_sample_schema(package_schema())
        a = featurizer.prepare_graph(first).data
        b = featurizer.prepare_graph(second).data
        self.assertFalse(torch.equal(a.node_num, b.node_num))
        self.assertFalse(torch.equal(a.node_weight, b.node_weight))

    def test_rejects_outside_template(self) -> None:
        """不允许静默移除额外反应组分或把非法官能度降级为双官能。"""
        components = component_fixture()
        with self.assertRaises(ValueError):
            recipe_to_complete_graph(components + [components[2]], 1.0, 0.3)
        components[0]["functionality"] = 3
        with self.assertRaises(ValueError):
            recipe_to_complete_graph(components, 1.0, 0.3)
        with self.assertRaises(ValueError):
            recipe_to_complete_graph(component_fixture()[:2], 1.0, 0.3)


if __name__ == "__main__":
    unittest.main()
