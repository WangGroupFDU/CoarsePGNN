# 5. Constructing, training, and applying the Hybrid encoder

**Scope:** This chapter describes the repository software and GitHub assets at v1.0.1. For the separately packaged Zenodo v1.1.0 archive and its archived-ID interface, see [Chapter 3](03_assets.md).

## Prerequisites and frozen version

Extract the `hybrid` archive. Extract `pu` as well when reproducing PU cached features. This release uses `PU_CONDITION_ENCODERS_V2_EXT16_20260824T104653Z_0fb2aa90`, the early encoder associated with the property checkpoints.

The Hybrid representation concatenates a normalized 32-dimensional Qwen text vector and a normalized 16-dimensional numerical DeepSets vector. Registered numerical values are unit-normalized and represented using piecewise linear encoding (PLE), field identity, and masks. The frozen PLE configuration defines its knots; it must not be refitted on test data.

## Text and numerical inputs

```json
{"sample_id":"new_condition_01","condition":{"raw_condition_text":"Uniaxial tensile test at 23 °C with a crosshead speed of 100 mm/min.","numeric_tags":[{"key":"test_temperature_c","value":23,"unit":"°C"},{"key":"crosshead_speed_mm_min","value":100,"unit":"mm/min"}]}}
```

Save each record on one line in `new_conditions.jsonl`. Numerical tags must be explicit, unique, and registered. The interface does not extract every number from free text and does not include property labels or sample identifiers in prompts. Missing values are omitted, not set to zero. Raw text is preserved verbatim.

`assets/hybrid/field_registry_v2.json` specifies all 16 fields, canonical units, accepted aliases, bounds, and transformations. `prompt_templates_v2.json` preserves the template version. Eight canonical fields were populated through the frozen PU field mapping; a larger registry does not imply that every field was observed during property training.

## Reuse the archived prompt cache

```bash
python -m coarsepgnn_release encode --assets assets \
  --input examples/conditions_cached.jsonl \
  --output outputs/conditions_cached.npz --device cpu
```

This example requires no Qwen base-model download. Expected output is an NPZ with `observation_ids` and a finite `hybrid48` matrix of shape `[N,48]`, plus a JSON receipt. The vectors are aligned by identifier, not by an assumed row order.

## Encode previously unseen text

Download the official fixed revision into a directory whose final name is exactly the revision string:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-Embedding-4B', revision='959bed3521cc1e8714392c77e15ae213b00aa921', local_dir='models/Qwen3-Embedding-4B/959bed3521cc1e8714392c77e15ae213b00aa921')"
python -m coarsepgnn_release encode --assets assets \
  --input new_conditions.jsonl --output outputs/new_conditions.npz \
  --qwen-model models/Qwen3-Embedding-4B/959bed3521cc1e8714392c77e15ae213b00aa921 \
  --device cpu --batch-size 4
```

The base model is provided by [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-4B). Inference loads local files only, uses left padding and the last non-padding token, retains the first 32 embedding dimensions, and normalizes them. Inputs exceeding the frozen 1,024-token limit are rejected instead of truncated silently.

## Reproduce numerical-head training

The archive includes 524 source records, 1,800 benchmark records, development-family folds, prompt caches, the PLE configuration, and the selected head. Use cached Qwen outputs to retrain the small numerical head:

```bash
python -m coarsepgnn_release train-encoder --assets assets \
  --output outputs/hybrid_selected
```

The selected configuration is `cfg03`: learning rate 0.001, semantic margin 0.1, and 17 final training epochs. The final head is trained using the designated source train/validation records and benchmark development records. The final benchmark test partition is excluded.

To repeat the archived 20-configuration, five-fold development search:

```bash
python -m coarsepgnn_release train-encoder --assets assets \
  --output outputs/hybrid_search --tune --workers 1
```

The source code preserves sampling, loss weighting, seeds, early stopping, and selection criteria. Output includes the new head, PLE configuration, configuration selection, and training report. Output directories must be new.

## The two alpha values

The early head was selected with `alpha_raw=0.25`; this remains in its original checkpoint metadata. **The released PU property models use an explicit override of `alpha_raw=0.5`.** The two normalized branches are weighted by `sqrt(alpha)` and `sqrt(1-alpha)`; 0.5 therefore gives equal squared-norm contributions.

The public `encode` and `predict` interfaces use 0.5. Do not alter the head metadata to disguise the distinction. A newly trained head is an experimental output, not automatically a byte-identical replacement for the archived head or cached PU vectors.

