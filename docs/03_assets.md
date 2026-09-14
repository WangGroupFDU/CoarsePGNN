# 3. Downloading and verifying the reproducibility assets

## Prerequisites and deposit status

Configure Python as in Chapter 2. The release-specific Zenodo status, record identifier, archive sizes, and SHA-256 values are recorded in `src/coarsepgnn_release/assets.json`. **Until the deposit is published, its files require owner access and the public downloader cannot retrieve them.** An upload draft or reserved DOI is not a published data record.

The four complete archives are available as assets of [GitHub release v1.0.1](https://github.com/WangGroupFDU/CoarsePGNN/releases/tag/v1.0.1). The [Zenodo draft 22741277](https://zenodo.org/uploads/22741277) remains unpublished and requires owner access. Final publication and DOI registration await specific-draft approval. The GitHub copies and the Zenodo deposit use the same pinned archive hashes.

The code release is version `1.0.1`. Use the matching code and asset manifests together.

## Archive layout

| Archive part | Extracted directory | Contents |
|---|---|---|
| `openpoly` | `assets/openpoly` | Canonical structures, raw member observations, exclusions, fixed folds, graph index, 7,682 graph tensors |
| `pu` | `assets/pu` | 3,657 observations, fixed folds, conditions, 1,162 graph files, cached vectors, relational provenance |
| `hybrid` | `assets/hybrid` | Early encoder head, PLE knots, 16-field registry, prompts, head-training records, development folds, tuning logs, prompt cache |
| `models` | `assets/models` | 80 PU checkpoints and 5 OpenPoly checkpoints |

Each archive contains a top-level directory named after its part. Shared data are stored once. No Qwen base weights or article full-text PDFs are included.

## Download the available GitHub assets

```bash
python -m coarsepgnn_release download --source github --download-dir downloads --output assets
python -m coarsepgnn_release verify-assets --assets assets
```

To download a single component, add `--part pu`, `--part openpoly`, `--part hybrid`, or `--part models`. After a Zenodo record is published, select `--source zenodo`. A record can be supplied explicitly with `--record-id RECORD_ID`, but its files must still match the pinned archive hashes. No personal Zenodo token is needed for a published public record.

The command verifies file size and SHA-256 before extraction, rejects unsafe archive members, and refuses to overwrite an existing asset part. A complete, matching download can be reused. Inspect an interrupted `.partial` file before removing it and retrying.

## Manual extraction

Download the four ZIP files from the matching GitHub release, or from the matching Zenodo record once published. Run from the repository root:

```bash
python -m zipfile -e downloads/CoarsePGNN_openpoly_v1.0.1.zip assets
python -m zipfile -e downloads/CoarsePGNN_pu_v1.0.1.zip assets
python -m zipfile -e downloads/CoarsePGNN_hybrid_v1.0.1.zip assets
python -m zipfile -e downloads/CoarsePGNN_models_v1.0.1.zip assets
python -m coarsepgnn_release verify-assets --assets assets
```

Expected output is a JSON status of `passed` and the number of checked files. `RELEASE_MANIFEST.json` inside each part enumerates its data files and hashes. The manifest excludes itself; the ZIP hash protects it.

## Identity and portability

The original PU dataset and fold manifests are retained byte-for-byte because their hashes are stored in checkpoints. Their historical `graph_relpath` values serve as provenance. Training resolves them inside `assets/pu/graphs/MATERIAL_ID/GRAPH_ID.json`; no connection to the original server is made.

Prediction JSONL paths are resolved relative to the input file. The supplied examples assume `examples/` and `assets/` are siblings under the repository. If assets are stored elsewhere, set `--assets` and update each example's `graph_file` accordingly.

Read [DATA_CARD.md](DATA_CARD.md) and the dictionaries inside the archives before interpreting raw, standardized, or masked values.


## Supplementary literature datasets

COF and dendrimer data are separate additions to the same unpublished Zenodo record. See [Chapter 10](10_supplementary_data.md) for their manifests, counts, scientific boundaries, and extraction commands. The automatic downloader above covers the four core OpenPoly/PU/Hybrid/model archives.
