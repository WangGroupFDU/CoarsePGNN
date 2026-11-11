"""逆向准备层的数学、隔离和输入边界测试；合成样例不代表材料实验。"""
from __future__ import annotations

import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path

import torch

from coarsepgnn_inverse.audit import assign_splits, identity_groups
from coarsepgnn_inverse.chemistry import decode_ratios
from coarsepgnn_inverse.io import write_csv
from coarsepgnn_inverse.model import InverseDesigner
from coarsepgnn_inverse.schema import constraint_results, validate_request


class ChemistryAndBoundaryTests(unittest.TestCase):
    """验证计量守恒、缺失语义、硬约束以及跨平台 CSV 格式。"""

    def test_stoichiometry_and_mass(self):
        """已知 2:1:1 配方必须复现质量分数；非化学计量 R 仍保持原值。"""
        out = decode_ratios(1, .5, [250.25, 1000, 90.12], True)
        self.assertEqual(out["molecule_ratios"], [1, .5, .5])
        self.assertAlmostEqual(out["mass_fractions"][0], 250.25 / 795.31)
        self.assertAlmostEqual(sum(out["mass_fractions"]), 1)
        off = decode_ratios(1.05, .3, [168.19, 1000, 90.12], True)
        self.assertAlmostEqual(off["molecule_ratios"][0] / sum(off["molecule_ratios"][1:]), 1.05)
        self.assertEqual(decode_ratios(1, 0, [168.19, 1000], False)["molecule_ratios"], [1, 1])

    def test_reject_bad_ratio(self):
        """非有限、越界、NONE 带正投料及质量缺失均必须显式报错。"""
        for r, v, m, e in [(float("nan"), .2, [1, 1, 1], True),
                            (1, 0, [1, 1, 1], True), (1, .1, [1, 1], False),
                            (1, 1, [1, 1, 1], True), (1, .5, [1, 0, 1], True)]:
            with self.assertRaises(ValueError):
                decode_ratios(r, v, m, e)

    def test_zero_target_and_hard_constraints(self):
        """真实 Tg=0 不当成缺失；另一性质失败不能由达标性质抵消。"""
        req = {"schema_version": 1, "constraints": [
            {"target": "tg", "unit": "degC", "lower": 0, "upper": 5,
             "protocol_id": "test_only", "condition": {"method": "DSC"},
             "assay": {"transition_criterion": "midpoint"}},
            {"target": "elongation_at_break", "unit": "%", "lower": 400,
             "protocol_id": "test_only", "condition": {"temperature_c": 23},
             "assay": {"property_subtype": "elongation_at_break"}}]}
        validate_request(req)
        checks = constraint_results(req["constraints"], [0, 399])
        self.assertTrue(checks[0]["model_constraint_passed"])
        self.assertFalse(checks[1]["model_constraint_passed"])
        bad = copy.deepcopy(req)
        bad["constraints"][0]["upper"] = -1
        with self.assertRaises(ValueError):
            validate_request(bad)
        bad = copy.deepcopy(req)
        bad["constraints"][0]["condition"] = {}
        with self.assertRaises(ValueError):
            validate_request(bad)

    def test_csv_roundtrip(self):
        """含逗号、引号和换行的证据必须可无损往返，并具有 BOM/CRLF。"""
        value = '中文, "quoted"\nline'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            write_csv(path, [{"evidence": value}], ["evidence"])
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            self.assertIn(b"\r\n", raw)
            self.assertEqual(list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))[0]["evidence"], value)
            with self.assertRaises(FileExistsError):
                write_csv(path, [], ["evidence"])


class SplitTests(unittest.TestCase):
    """验证跨任务和来源的传递闭包，不以记录随机划分替代材料隔离。"""

    def test_transitive_closure(self):
        """A 同材料 B、B 同图 C 时，A/C 即使无直接共有字段也必须同组。"""
        rows = [{"observation_id": "A", "material_id": "m1", "graph_input_sha256": "g1", "source_id": "s1"},
                {"observation_id": "B", "material_id": "m1", "graph_input_sha256": "g2", "source_id": "s2"},
                {"observation_id": "C", "material_id": "m2", "graph_input_sha256": "g2", "source_id": "s3"},
                {"observation_id": "D", "material_id": "m3", "graph_input_sha256": "g3", "source_id": "s4"}]
        groups = identity_groups(rows)
        self.assertEqual(groups["A"], groups["C"])
        self.assertNotEqual(groups["A"], groups["D"])
        self.assertEqual(groups, identity_groups(list(reversed(rows))))

    def test_split_ignores_values_and_row_order(self):
        """固定划分不能随性质值、预测表现或输入排列改变。"""
        tasks = ["tg", "tm", "tensile_strength_ultimate", "elongation_at_break"]
        rows = [{"observation_id": f"{i}_{t}", "material_id": f"m{i}", "source_id": f"s{i}",
                 "formulation_key": f"f{i}", "graph_input_sha256": f"g{i}", "target": t, "value": i}
                for i in range(50) for t in tasks]
        groups = identity_groups(rows)
        original = assign_splits(rows, groups, 20261111)
        changed = [{**r, "value": 999999} for r in reversed(rows)]
        self.assertEqual(original, assign_splits(changed, identity_groups(changed), 20261111))
        self.assertEqual(set(original.values()), {"train", "validation", "test"})


class NeuralTests(unittest.TestCase):
    """真实 PyTorch 张量核查集合编码、MDN 梯度、角色合法性和确定性提案。"""

    def setUp(self):
        """构造有限角色小库；第六项为 NONE，全部为合成测试特征。"""
        torch.manual_seed(7)
        features = torch.randn(7, 12)
        masks = torch.zeros(3, 7, dtype=torch.bool)
        masks[0, :2] = True
        masks[1, 2:4] = True
        masks[2, 4:] = True
        self.model = InverseDesigner(features, masks, none_index=6)
        self.inputs = {"property_ids": torch.tensor([[0, 2, 1], [3, 1, 0]]),
                       "bounds": torch.randn(2, 3, 2), "bound_mask": torch.ones(2, 3, 2, dtype=torch.bool),
                       "conditions": torch.randn(2, 3, 48), "assays": torch.randn(2, 3, 16),
                       "target_mask": torch.tensor([[True, True, False], [True, False, False]]),
                       "nco_oh_ratio": torch.tensor([1.0, 1.05])}

    def test_set_permutation_and_missingness(self):
        """请求顺序和未启用字段不影响上下文，但真实已启用目标可以影响它。"""
        expected = self.model.encode(**self.inputs)
        changed = {k: v[:, [2, 0, 1]] if v.ndim >= 2 else v for k, v in self.inputs.items()}
        torch.testing.assert_close(expected, self.model.encode(**changed))
        changed = copy.deepcopy(self.inputs)
        off = ~changed["target_mask"]
        for key in ["bounds", "conditions", "assays"]:
            changed[key][off] = float("nan")
        torch.testing.assert_close(expected, self.model.encode(**changed))
        changed = copy.deepcopy(self.inputs)
        changed["bounds"][0, 0, 0] += 1
        self.assertFalse(torch.equal(expected, self.model.encode(**changed)))

    def test_mdn_finite_backward_including_none(self):
        """NONE 和极端但有限配比混合训练时仍有有限梯度，原料特征保持冻结。"""
        context = self.model.encode(**self.inputs)
        loss = self.model.supervised_loss(context, torch.tensor([[0, 2, 4], [1, 3, 6]]), torch.tensor([1e-5, 0.0]))
        loss["loss"].mean().backward()
        self.assertEqual(float(loss["ratio_nll_z"][1]), 0.0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.assertIsNone(self.model.grade_features.grad)
        self.assertIsNotNone(self.model.ratio_head[-1].weight.grad)

    def test_ratio_reads_final_extender_and_proposals_are_legal(self):
        """只改变 E 必须改变配比参数；重复束搜索稳定且不产生角色非法原料。"""
        context = self.model.encode(**self.inputs)
        embeddings = self.model.grade_encoder(self.model.grade_features)
        a = self.model.ratio_parameters(context[:1], torch.tensor([[0, 2, 4]]), embeddings)
        b = self.model.ratio_parameters(context[:1], torch.tensor([[0, 2, 5]]), embeddings)
        self.assertFalse(torch.equal(a[1], b[1]))
        first = self.model.propose(context[:1], beam_width=8)
        self.assertEqual(first, self.model.propose(context[:1], beam_width=8))
        self.assertGreater(len(first), 0)
        for candidate in first:
            for role, index in enumerate(candidate["grade_indices"]):
                self.assertTrue(self.model.role_masks[role, index])
            if candidate["grade_indices"][-1] == 6:
                self.assertEqual(candidate["extender_oh_fraction"], 0)
            else:
                self.assertTrue(0 < candidate["extender_oh_fraction"] < 1)


if __name__ == "__main__":
    unittest.main()
