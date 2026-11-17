"""微型真实运行检查 CLI 接线、无标签输入与不可覆盖输出。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from coarsepgnn_retrain.champion import FORWARD_FAMILY_POLICY
from coarsepgnn_retrain.cli import _rows, main


@pytest.fixture(scope="module")
def cli_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("retrain_cli")
    rows = []
    for index in range(4):
        # 每个来源使用真实不同的组成；仅修改 ID 不能代表独立配方闭包。
        components = [{"component_id": "D1", "role": "D", "name": "D1", "smiles": "O=C=NCCN=C=O",
                       "functionality": 2, "identity_mask": True, "molecule_ratio": .5 + index / 100, "ratio_mask": True},
                      {"component_id": "P1", "role": "P", "name": "P1", "smiles": "OCCO",
                       "functionality": 2, "identity_mask": True, "molecule_ratio": .5 - index / 100, "ratio_mask": True}]
        rows.append({"observation_id": f"o{index}", "material_id": f"m{index}", "source_id": f"s{index}",
                     "formulation_key": f"f{index}", "target": "tg", "value": float(index), "unit": "degC",
                     "shared_trainable": True, "components": components, "conditions": {}})
    # JSON 数组母表和 JSONL 请求同时覆盖；原 JSONL 母表入口保持兼容。
    (root / "dataset.json").write_text(json.dumps(rows))
    configuration = {"seeds": [73], "bootstrap_replicates": 10,
                     "candidates": [{"hidden": 8, "fingerprint_bits": 32, "mixture_components": 2,
                                     "batch_size": 8, "max_epochs": 1, "patience": 2, "dropout": 0.}],
                     "baseline_candidates": [{"name": "ridge", "family": "ridge", "alpha": 1.}],
                     "forward_family_selection": deepcopy(FORWARD_FAMILY_POLICY)}
    (root / "configuration.json").write_text(json.dumps(configuration))
    common = ["--dataset", str(root / "dataset.json"), "--splits", str(root / "splits.json")]
    previous = torch.get_num_threads()
    try:
        main(["split", "--dataset", str(root / "dataset.json"), "--output", str(root / "splits.json"),
              "--outer-folds", "2", "--inner-folds", "2", "--seed", "73"])
        for command in ("baselines", "train"):
            main([command, *common, "--config", str(root / "configuration.json"),
                  "--output", str(root / command), "--threads", "1", "--max-seconds", "90"])
        main(["baseline-refit", *common, "--baseline-run", str(root / "baselines"),
              "--output", str(root / "classical"), "--seed", "73", "--max-seconds", "90"])
    finally:
        torch.set_num_threads(previous)
    recipe = {key: deepcopy(value) for key, value in rows[0].items() if key != "value"}
    recipe["request_id"] = "cli_recipe"
    unsupported = deepcopy(recipe)
    unsupported["target"] = "unknown_task"
    (root / "forward.jsonl").write_text("\n".join(json.dumps(row) for row in (recipe, unsupported)) + "\n")
    (root / "inverse.json").write_text(json.dumps({"targets": {"tg": 1.}, "conditions": {}}))
    return root, root / "train/outer_0/refit_73_trial_0"


def test_predict_actual_module_entrypoint_and_no_overwrite(cli_inputs, tmp_path):
    root, model = cli_inputs
    output = tmp_path / "prediction.json"
    repository = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "coarsepgnn_retrain", "predict", "--model-dir", str(model),
               "--input", str(root / "forward.jsonl"), "--output", str(output), "--threads", "1"]
    process = subprocess.run(command, cwd=repository, env={**os.environ, "PYTHONPATH": str(repository / "src")},
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["status"] == "completed"
    payload = json.loads(output.read_text())
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["status"] == "predicted"
    assert payload["rows"][1]["status"] == "unsupported"
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["predict", "--model-dir", "does_not_exist", "--input", str(root / "forward.jsonl"), "--output", str(output)])
    assert output.read_bytes() == original


def test_propose_cli_keeps_one_request_and_exact_budget(cli_inputs, tmp_path, capsys):
    root, model = cli_inputs
    output = tmp_path / "proposals.json"
    main(["propose", "--model-dir", str(model), "--input", str(root / "inverse.json"),
          "--output", str(output), "--count", "3", "--seed", "17", "--threads", "1"])
    assert json.loads(capsys.readouterr().out)["command"] == "propose"
    result = json.loads(output.read_text())
    assert result["requested_count"] == result["candidate_count"] == result["proposal_budget_consumed"] == 3
    assert result["hidden_retries"] == 0
    assert result["experimental_success"] is None
    multiple = tmp_path / "multiple.jsonl"
    multiple.write_text(root.joinpath("inverse.json").read_text() + "\n" + root.joinpath("inverse.json").read_text())
    with pytest.raises(ValueError, match="exactly one request"):
        main(["propose", "--model-dir", "does_not_exist", "--input", str(multiple),
              "--output", str(tmp_path / "not_created.json"), "--count", "3", "--seed", "17"])
    assert not (tmp_path / "not_created.json").exists()


def test_classical_cli_uses_real_full_refit_and_requires_explicit_trust(cli_inputs, tmp_path):
    root, _ = cli_inputs
    audit = json.loads((root / "classical/audit.json").read_text())
    assert audit["status"] == "completed"
    assert audit["all_supervised_observations_used"] is True
    assert audit["supervised_observations"] == 4
    output = tmp_path / "classical_prediction.json"
    command = ["predict", "--family", "classical", "--model-dir", str(root / "classical"),
               "--input", str(root / "forward.jsonl"), "--output", str(output)]
    with pytest.raises(ValueError, match="trusted_generated_artifact"):
        main(command)
    assert not output.exists()
    main([*command, "--trusted-generated-artifact"])
    result = json.loads(output.read_text())
    assert [row["status"] for row in result["rows"]] == ["predicted", "unsupported"]
    assert result["fitting_performed"] is False
    with pytest.raises(FileExistsError):
        main(["baseline-refit", "--dataset", str(root / "dataset.json"), "--splits", str(root / "splits.json"),
              "--baseline-run", str(root / "baselines"), "--output", str(root / "classical")])


def test_champion_cli_requires_completed_bound_runs_and_writes_outputs(cli_inputs, tmp_path):
    root, _ = cli_inputs
    output = tmp_path / "champion"
    command = ["champion", "--dataset", str(root / "dataset.json"), "--splits", str(root / "splits.json"),
               "--baseline-run", str(root / "baselines"), "--neural-run", str(root / "train"),
               "--bootstrap-replicates", "10", "--output", str(output)]
    main(command)
    assert all((output / name).is_file() for name in ("selection_lock.json", "selected_oof.csv", "metrics.json"))
    assert json.loads((output / "selection_lock.json").read_text())["outer_predictions_used_for_selection"] is False
    with pytest.raises(FileExistsError):
        main(command)


def test_json_and_jsonl_reader_rejects_empty_or_nonobject_records(tmp_path):
    path = tmp_path / "input.json"
    for content in ("", "[]", "[1, 2]", "null"):
        path.write_text(content)
        with pytest.raises(ValueError):
            _rows(path)
    for content in ('{"a": 1}', '[{"a": 1}]', '{"a": 1}\n'):
        path.write_text(content)
        assert _rows(path) == [{"a": 1}]


def test_cli_help_preserves_old_commands_and_lists_new_commands(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("fetch", "prepare", "split", "baselines", "smoke", "train", "refit",
                    "predict", "propose", "champion", "baseline-refit"):
        assert command in help_text
