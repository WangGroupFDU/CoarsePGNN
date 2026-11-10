# 9. Input reference, errors, and citation

**Scope:** This chapter describes the repository software and GitHub assets at v1.0.1. For the separately packaged Zenodo v1.1.0 archive and its archived-ID interface, see [Chapter 3](03_assets.md).

## Prerequisites

Use the archive-specific dictionaries in `assets/*/DATA_DICTIONARY.md`, the JSON registry for numeric conditions, and the model graph schema. This chapter describes the public prediction interface.

## Prediction JSONL fields

| Field | Type | Requirement and interpretation |
|---|---|---|
| `sample_id` | string | Required, unique within the input file |
| `graph_file` | path string | PU coarse-graph JSON; OpenPoly compatible PyG PT; relative to this JSONL file |
| `canonical_psmiles` | string | OpenPoly alternative to a stored graph |
| `h_capped_smiles` | string | Required with new OpenPoly PSMILES; matching hydrogen-capped repeat unit |
| `condition_id` | string | Hybrid vector identifier in the archived or explicitly supplied NPZ |
| `condition` | object | Hybrid alternative to `condition_id`; contains raw text and numeric tags |
| `assay` | object | Hybrid categorical assay descriptors; omitted values map to unknown |

The seven assay keys are `property_subtype`, `property_task`, `transition_kind`, `transition_criterion`, `transition_scope`, `statistic_type`, and `definition_status`. Their allowed learned categories are checkpoint-specific; the provided examples use real archived values. Do not infer unreported assay definitions.

The graph featurizer uses 2,048-bit Morgan fingerprints for raw and reaction-site representations, numerical values with explicit masks, site ownership/state, directed reaction edges, and graph-level attributes. See `graph_contract.required_graph_tensors` and `feature_contract` in the schema for exact shapes and ordering. Site indices belong to the concatenated site list, not the atom list. Unknown categorical IDs are zero.

## Units and missing values

For PU, `value_celsius` and `value_kelvin` are retained separately for temperature targets. The public PU predictions are °C; OpenPoly predictions are K. Elongation is percent strain, not a fraction; ultimate tensile strength is MPa. Numeric condition tags include their units and are normalized through the frozen registry.

A missing numerical condition is absent from `numeric_tags`; an observed zero is a real value. Raw text and normalized condition JSON coexist. Masked values must not be treated as observations. Original SMILES, structural PSMILES, and reaction-site PSMILES are separate fields. Component Mn is not necessarily final-polymer Mn.

## Common errors

| Error | Resolution |
|---|---|
| Zenodo filename lookup failure | The repository downloader expects v1.0.1 files; use `--source github` or follow the separate v1.1.0 instructions in Chapter 3 |
| Missing archive or checkpoint | Extract the matching asset part; all five folds are required |
| SHA-256 mismatch | Use the original release file; investigate accidental editing or an incomplete transfer |
| Existing output | Choose a new output path; workflows do not replace earlier results |
| Unknown condition ID | Verify NPZ IDs and the explicit `--condition-vectors` path |
| Uncached prompts | Provide the fixed Qwen snapshot with `--qwen-model` |
| Qwen revision mismatch | Use the exact revision directory named in Chapter 5 |
| More than 1,024 tokens | Supply a scientifically faithful shorter condition description; do not silently truncate |
| Missing or invalid graph fields | Use the correct generator/schema and inspect its validation report |
| No supported reaction template | Treat classification-only output as such; do not force an unrelated reaction |
| CUDA unavailable | Select `--device cpu` or configure a compatible GPU runtime |
| Arial missing | Supply a licensed Arial font or run numerical evaluation without `--plots` |
| Hybrid structural identity mismatch | Use the same target, strategy, fold, seed, and frozen data manifests |

For example, verify a downloaded subset before diagnosing model errors:

```bash
python -m coarsepgnn_release verify-assets --assets assets --part models
python -m coarsepgnn_release predict --help
```

Expected output is a passed model-file check and the argument reference.

## Citation and reuse

Cite the software entry in [`CITATION.cff`](../CITATION.cff) for the repository software v1.0.1. For the published dataset/model archive v1.1.0, cite Jifeng Wang (Fudan University), *CoarsePGNN datasets, cross-validation splits, and reproducibility assets*, Zenodo, 7 November 2026, [doi:10.5281/zenodo.22741277](https://doi.org/10.5281/zenodo.22741277). These versions describe different artifacts; the dataset DOI does not identify a new v1.1.0 repository CLI release.

For OpenPoly context, see [Wang et al., Chinese Journal of Polymer Science (2025)](https://doi.org/10.1007/s10118-025-3402-y) and the [OpenPoly repository](https://github.com/WangGroupFDU/Openpoly_benchmark). The larger dated input snapshot used here is identified by its source hash and member references; it is not asserted to be identical to an earlier public benchmark. Cite source articles listed in the data provenance when reusing individual measurements.

Report the code version, asset hashes, dataset, target, split strategy, model family, fold handling, device, and whether you retrained or used archived checkpoints.

