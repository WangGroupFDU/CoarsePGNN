# CoarsePGNN

**Coarse-grained polymer graphs for reproducible property prediction.**

This repository contains the electrophile–nucleophile topology generator, the frozen PU Hybrid condition encoder, and the v1.0.1 software for the archived OpenPoly/PU experiments. The release author is **Jifeng Wang, Fudan University**.

## Published reproducibility archive: v1.1.0

The datasets, fixed cross-validation splits, and model assets are published on [Zenodo as v1.1.0](https://zenodo.org/records/22741277), dated 7 November 2026. Cite the version-specific DOI [10.5281/zenodo.22741277](https://doi.org/10.5281/zenodo.22741277).

The archive covers **six benchmark tasks**: homopolymer Tg, four polyurethane properties, and block-copolymer phase classification. It contains 15,239 task records, 90 full-row train/validation/test tables, and **95 property checkpoints**: 55 manuscript-task checkpoints plus 40 retained grouped-validation PU checkpoints. The task-record total is not a count of independent materials. COF and dendrimer literature collections are supplementary, outside these six benchmarks.

**The Zenodo archive version and this repository's software version are different.** Zenodo v1.1.0 uses four archives sharing a `CoarsePGNN/` directory and includes its own archived-ID prediction interface and matching historical source packages. The repository CLI remains v1.0.1 and requires the older GitHub asset layout. It does not load the new layout or provide phase classification; `download --source zenodo --record-id 22741277` is not compatible with its pinned v1.0.1 filenames and hashes.

### Use the Zenodo v1.1.0 archive

1. Follow [v1.1.0 download, SHA-256 checks, and extraction](docs/03_assets.md#zenodo-v110-download-and-extraction).
2. Run `documentation/predict.py` from the extracted `CoarsePGNN/` directory using the archive's environment and [archived-ID examples](docs/03_assets.md#v110-archived-id-prediction).
3. Read the archive's `documentation/model_index.json`, `documentation/validation/`, and `documentation/MANUSCRIPT_AUDIT.md` for exact model pairings and metric conventions.

Phase Hybrid inference requires the structure checkpoint and condition encoder from the same fold. Archive prediction reproduces selected existing IDs; it does not construct new polymer graphs or retrain models. Five-fold mean metrics and pooled out-of-fold statistics are distinct, and random record splits do not establish unseen-material generalization.

### Use the existing repository CLI: v1.0.1

For the existing topology, training, new-input prediction, and compact-result workflows, follow [environment setup](docs/02_environment.md) and use the matching [GitHub v1.0.1 assets](https://github.com/WangGroupFDU/CoarsePGNN/releases/tag/v1.0.1). These assets contain 85 regression checkpoints: five OpenPoly Tg models and 80 PU models, including both record-random and formulation/graph-grouped splits. The compatible early PU Hybrid numerical head is supplied separately.

Run from this repository, not the extracted Zenodo directory:

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

The example uses five grouped PU models. Each fold is inverted to the original target unit before arithmetic averaging. Its standard deviation measures disagreement among the five models, not a calibrated confidence interval. Example observations belong to the training archive and demonstrate loading; their ensemble predictions are not independent validation. An optional package installation also provides the `coarsepgnn` command.

## Documentation

Chapters 02 and 04–09 describe the existing v1.0.1 repository workflows. Chapter 03 separates the two asset layouts; Chapter 10 describes the published v1.1.0 supplements.

| Chapter | Purpose |
|---|---|
| [01 — Overview](docs/01_overview.md) | Archive scope, repository architecture, and scientific limitations |
| [02 — Environment](docs/02_environment.md) | Repository dependencies and platform-specific setup |
| [03 — Assets](docs/03_assets.md) | Zenodo v1.1.0 download and integrity checks; legacy CLI compatibility |
| [04 — Topology](docs/04_topology.md) | Automatic tree growth and explicit topology constraints |
| [05 — Hybrid encoder](docs/05_hybrid.md) | PU text/numeric fields, PLE, training, and cached inference |
| [06 — Property training](docs/06_training.md) | Existing OpenPoly and PU five-fold training |
| [07 — Prediction](docs/07_prediction.md) | Repository new-input and mean-of-five inference |
| [08 — Evaluation](docs/08_evaluation.md) | Repository regression results, OOF metrics, histories, and figures |
| [09 — Input reference](docs/09_inputs.md) | Schemas, units, missing values, troubleshooting, and citation |
| [10 — Supplementary datasets](docs/10_supplementary_data.md) | Published COF/dendrimer collections and grouped-validation assets |

Additional references: [data card](docs/DATA_CARD.md), [model card](docs/MODEL_CARD.md), [release provenance](docs/PROVENANCE.md), [validation record](docs/VALIDATION.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

## Availability and license

Code and compact v1.0.1 regression results are hosted at [WangGroupFDU/CoarsePGNN](https://github.com/WangGroupFDU/CoarsePGNN). [GitHub release v1.0.1](https://github.com/WangGroupFDU/CoarsePGNN/releases/tag/v1.0.1) remains the asset source for the repository CLI. The separately packaged [Zenodo v1.1.0 record](https://zenodo.org/records/22741277) is public and requires no personal Zenodo token. Neither archive replaces the other layout in place.

Original code is licensed under [MIT](LICENSE). Original data curation and model weights are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party sources retain their own rights and attribution. [CITATION.cff](CITATION.cff) distinguishes the v1.0.1 software from the v1.1.0 dataset citation. Cite the underlying literature when reusing individual measurements.
