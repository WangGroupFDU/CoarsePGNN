# CoarsePGNN

**Coarse-grained polymer graphs for reproducible property prediction.**

This repository contains the electrophile–nucleophile topology generator, the frozen Hybrid condition encoder, and the code underlying the archived five-fold CoarsePGNN experiments. The release author is **Jifeng Wang, Fudan University**.

The reproducibility assets contain **85 property checkpoints**: five OpenPoly Tg models and 80 PU models. PU includes structure-only and Hybrid models for Tg, Tm, elongation at break, and ultimate tensile strength, under both record-random and formulation/graph-grouped partitions. The early, compatible Hybrid numerical head is supplied separately.

## Start here

1. Follow [environment setup](docs/02_environment.md).
2. Follow [asset download and integrity checks](docs/03_assets.md).
3. Run a pretrained example from [five-fold prediction](docs/07_prediction.md).
4. Recompute the archived results with [evaluation](docs/08_evaluation.md).

All commands assume that the current directory is this repository. The public interface is `python -m coarsepgnn_release`; an optional package installation also provides `coarsepgnn`.

```bash
export PYTHONPATH="$PWD/src"
python -m coarsepgnn_release --help
python -m coarsepgnn_release download --source github --output assets
python -m coarsepgnn_release verify-assets --assets assets
python -m coarsepgnn_release predict --assets assets \
  --dataset pu --model hybrid --target tg \
  --input examples/pu_tg.jsonl --output outputs/pu_tg.csv --device cpu
python -m coarsepgnn_release evaluate --results results \
  --output outputs/recomputed
```

The example uses five grouped PU models. Each fold is inverted to the original target unit before arithmetic averaging. Its standard deviation measures disagreement among the five models, not a calibrated confidence interval. Example observations belong to the training archive and demonstrate loading; their ensemble predictions are not independent validation.

## Documentation

| Chapter | Purpose |
|---|---|
| [01 — Overview](docs/01_overview.md) | Scope, architecture, and scientific limitations |
| [02 — Environment](docs/02_environment.md) | Fixed dependencies and platform-specific setup |
| [03 — Assets](docs/03_assets.md) | Four archives, extraction layout, and SHA-256 checks |
| [04 — Topology](docs/04_topology.md) | Automatic tree growth and explicit topology constraints |
| [05 — Hybrid encoder](docs/05_hybrid.md) | Text, numeric fields, PLE, training, and cached inference |
| [06 — Property training](docs/06_training.md) | OpenPoly and PU five-fold training |
| [07 — Prediction](docs/07_prediction.md) | New inputs and mean-of-five inference |
| [08 — Evaluation](docs/08_evaluation.md) | Fold metrics, OOF results, histories, and PNG figures |
| [09 — Input reference](docs/09_inputs.md) | Schemas, units, missing values, troubleshooting, citation |
| [10 — Supplementary datasets](docs/10_supplementary_data.md) | COF and dendrimer literature data, completeness flags, and candidate records |

Additional references: [data card](docs/DATA_CARD.md), [model card](docs/MODEL_CARD.md), [release provenance](docs/PROVENANCE.md), [validation record](docs/VALIDATION.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

The same unpublished Zenodo draft also contains [supplementary COF and dendrimer literature datasets](docs/10_supplementary_data.md), with separate scientific scope and eligibility annotations. They do not change the frozen OpenPoly/PU benchmarks.

## Availability and license

Code and compact results are hosted at [WangGroupFDU/CoarsePGNN](https://github.com/WangGroupFDU/CoarsePGNN). The complete assets are attached to [release v1.0.1](https://github.com/WangGroupFDU/CoarsePGNN/releases/tag/v1.0.1). The [Zenodo asset record](https://zenodo.org/uploads/22741277) is an unpublished draft for owner review. Its DOI has not been published; use GitHub for public downloads. See [asset status and download instructions](docs/03_assets.md).

Original code is licensed under [MIT](LICENSE). Original data curation and model weights are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party sources retain their own rights and attribution. Use [CITATION.cff](CITATION.cff) to cite this software; cite the underlying literature when reusing individual experimental measurements.

