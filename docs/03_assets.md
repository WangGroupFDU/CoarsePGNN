# 3. Downloading and verifying the reproducibility assets

## Choose the matching interface

[Zenodo record 22741277](https://zenodo.org/records/22741277) is **published**, with version **1.1.0**, publication date **7 November 2026**, and DOI [10.5281/zenodo.22741277](https://doi.org/10.5281/zenodo.22741277). No personal token is needed for its public files.

| Use | Asset source and layout | Interface |
|---|---|---|
| Six-task manuscript archive, including phase classification | Zenodo v1.1.0; one `CoarsePGNN/` root | Bundled `documentation/predict.py` for archived IDs |
| Existing repository topology/training/new-input prediction | GitHub v1.0.1; `assets/openpoly`, `pu`, `hybrid`, `models` | `python -m coarsepgnn_release` |

**Do not mix these layouts.** The repository software is still v1.0.1. Its `src/coarsepgnn_release/assets.json` pins the four GitHub v1.0.1 files, sizes, and SHA-256 values. `download --source zenodo` cannot retrieve v1.1.0: it expects old filenames absent from the published record, then checks the old hashes and ZIP roots. Setting `--record-id 22741277` does not change that. `verify-assets` likewise expects the old per-part `RELEASE_MANIFEST.json` files.

## Zenodo v1.1.0 download and extraction

The official [archive manifest](https://zenodo.org/records/22741277/files/archive_manifest_v1.1.0.json?download=1) and [SHA-256 list](https://zenodo.org/records/22741277/files/SHA256SUMS_v1.1.0.csv?download=1) identify these four files:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `CoarsePGNN_datasets_v1.1.0.zip` | 210,473,479 | `330910b95d542d82f2cfcd4c05d580096d791d05fd031c37baebaf63c2aa409c` |
| `CoarsePGNN_models_v1.1.0.zip` | 927,367,742 | `097a49d0e873d793666474d6f3d6bf4d12ec420eda29afdbe1913421160cde96` |
| `CoarsePGNN_condition_encoders_v1.1.0.zip` | 6,143,198 | `9d86088a5c7b7b2606c1409b145532485856febbf11dfade0ab8e1c8c8aba5fc` |
| `CoarsePGNN_supplementary_v1.1.0.zip` | 822,156,436 | `0f18e57c4881ba39247b6a9278717df66809a0e4cf205814572b4b469eef117d` |

Download all four from the record, or use this POSIX-shell example from the repository root. The ZIP files total approximately 1.97 GB; allow additional space for extraction.

```bash
mkdir -p downloads/v1.1.0
for part in datasets models condition_encoders supplementary; do
  curl --fail --location \
    "https://zenodo.org/records/22741277/files/CoarsePGNN_${part}_v1.1.0.zip?download=1" \
    --output "downloads/v1.1.0/CoarsePGNN_${part}_v1.1.0.zip"
done
curl --fail --location \
  "https://zenodo.org/records/22741277/files/archive_manifest_v1.1.0.json?download=1" \
  --output downloads/v1.1.0/archive_manifest_v1.1.0.json
```

Verify before extracting. This Python 3.11 standard-library check pins the manifest itself, then checks every ZIP size and SHA-256 without loading large archives into memory:

```bash
python - <<'PYVERIFY'
import hashlib
import json
from pathlib import Path

root = Path("downloads/v1.1.0")
manifest_bytes = (root / "archive_manifest_v1.1.0.json").read_bytes()
expected_manifest = "ad0334e83941942aeb3e8e5143892a578e34bfb7445e7f33169daa8f44572064"
if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest:
    raise SystemExit("Archive manifest SHA-256 mismatch")
manifest = json.loads(manifest_bytes)
for item in manifest["archives"]:
    path = root / item["filename"]
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if path.stat().st_size != item["size_bytes"] or actual != item["sha256"]:
        raise SystemExit(f"Archive size/SHA-256 mismatch: {path}")
    print(f"Verified: {path.name}")
PYVERIFY
```

After all four checks pass, extract into a **new** directory, separate from legacy `assets/`. All four ZIPs share the same `CoarsePGNN/` root:

```bash
mkdir archive_v1.1.0
python -m zipfile -e downloads/v1.1.0/CoarsePGNN_datasets_v1.1.0.zip archive_v1.1.0
python -m zipfile -e downloads/v1.1.0/CoarsePGNN_models_v1.1.0.zip archive_v1.1.0
python -m zipfile -e downloads/v1.1.0/CoarsePGNN_condition_encoders_v1.1.0.zip archive_v1.1.0
python -m zipfile -e downloads/v1.1.0/CoarsePGNN_supplementary_v1.1.0.zip archive_v1.1.0
```

| Extracted path under `archive_v1.1.0/CoarsePGNN/` | Contents |
|---|---|
| `datasets/homopolymer_tg/` | 7,682 homopolymer Tg task records |
| `datasets/polyurethane/` | Tg, Tm, tensile-strength and elongation tasks, plus shared graphs/provenance |
| `datasets/block_copolymer_phase/` | 3,900 BCDB phase task records and 312 approximate graphs |
| `models/` | 55 manuscript-task checkpoints: 5 homopolymer, 40 PU, 10 phase |
| `condition_encoders/` | Original PU encoder and five separate phase fold encoders/caches |
| `supplementary/` | COF and dendrimer literature data, grouped PU splits and 40 additional checkpoints |
| `documentation/` | Historical source packages, model index, provenance, checksums and validation evidence |

The datasets ZIP contains the shared documentation. Inference also needs the models and condition-encoder ZIPs; grouped PU inference additionally needs the supplementary ZIP. The archive's `documentation/SHA256SUMS.csv` lists extracted-file checksums. Qwen foundation-model weights and article full-text PDFs are not redistributed.

## v1.1.0 archived-ID prediction

Use compatible dependencies from the archive's recorded environments and run from its `CoarsePGNN/` directory. These commands belong to the Zenodo archive, not the repository CLI:

```bash
cd archive_v1.1.0/CoarsePGNN
python -B documentation/predict.py --root . \
  --task homopolymer_tg --mode structure_only --fold 1 \
  --ids openpoly_00001 --output outputs/homopolymer_example.csv
python -B documentation/predict.py --root . \
  --task block_copolymer_phase --mode hybrid --fold 1 \
  --ids BCDB-diblock-00008 --output outputs/phase_example.csv
```

The interface selects existing IDs from the archived `data.csv` tables. It does not build new graphs, fit encoders, download foundation-model weights or retrain predictors. Display folds are 1–5; `source_fold` preserves regression folds 0–4 and phase folds 1–5. Each invocation reproduces one fold, not the legacy five-fold ensemble.

Phase Hybrid requires the matching same-fold structure model and encoder; use the pairings in `documentation/model_index.json`. The main PU strategy is `record_random_interpolation`; grouped PU results are retained under `supplementary/grouped_validation`. Homopolymer Tg is in K, PU thermal targets in °C, strength in MPa, and elongation in %. Phase classes are lamellar, cylinder, gyroid, disordered and sphere. Read `documentation/MANUSCRIPT_AUDIT.md` before equating fold means with pooled OOF statistics.

## Legacy GitHub v1.0.1 assets for the repository CLI

Configure Python as in [Chapter 2](02_environment.md), return to the repository root, and use [GitHub release v1.0.1](https://github.com/WangGroupFDU/CoarsePGNN/releases/tag/v1.0.1):

```bash
export PYTHONPATH="$PWD/src"
python -m coarsepgnn_release download --source github --download-dir downloads --output assets
python -m coarsepgnn_release verify-assets --assets assets
```

| Archive part | Extracted directory | Contents |
|---|---|---|
| `openpoly` | `assets/openpoly` | Canonical structures, raw observations, exclusions, fixed folds, graph index, 7,682 graph tensors |
| `pu` | `assets/pu` | 3,657 observations, fixed folds, conditions, 1,162 graph files, cached vectors, relational provenance |
| `hybrid` | `assets/hybrid` | Early PU encoder head, PLE knots, 16-field registry, prompts, training records, tuning logs and prompt cache |
| `models` | `assets/models` | 80 PU checkpoints and 5 OpenPoly checkpoints |

To download one component, add `--part pu`, `--part openpoly`, `--part hybrid`, or `--part models`. The command verifies size and SHA-256 before extraction, rejects unsafe archive members, and refuses to overwrite an existing part. A matching completed download can be reused. Inspect an interrupted `.partial` file before removing it and retrying.

For manual extraction, download the matching **v1.0.1** GitHub ZIPs and verify their sizes and hashes against `src/coarsepgnn_release/assets.json` before running:

```bash
python -m zipfile -e downloads/CoarsePGNN_openpoly_v1.0.1.zip assets
python -m zipfile -e downloads/CoarsePGNN_pu_v1.0.1.zip assets
python -m zipfile -e downloads/CoarsePGNN_hybrid_v1.0.1.zip assets
python -m zipfile -e downloads/CoarsePGNN_models_v1.0.1.zip assets
python -m coarsepgnn_release verify-assets --assets assets
```

Expected verification output is JSON with status `passed` and the checked-file count. Each part's `RELEASE_MANIFEST.json` enumerates its files and hashes; the manifest excludes itself and is protected by the ZIP hash.

Original PU dataset/fold manifests retain their checkpoint-pinned bytes. Historical `graph_relpath` values are provenance; repository training resolves graphs inside `assets/pu/graphs/MATERIAL_ID/GRAPH_ID.json`. Prediction JSONL paths are relative to the input file. The supplied examples assume `examples/` and `assets/` are siblings; update `graph_file` paths if assets are elsewhere.

See [Chapter 10](10_supplementary_data.md) for the public v1.1.0 literature supplements. The legacy automatic downloader does not download them.
