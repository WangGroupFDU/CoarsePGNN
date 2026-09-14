# 7. Prediction with the five trained folds

## Prerequisites

Extract the required data and model parts and run `verify-assets`. PU Hybrid also requires its condition cache, or the early encoder and a fixed Qwen snapshot for uncached text. Input is UTF-8 JSON Lines with unique, nonempty `sample_id` strings.

## Reproduce a loading example

```bash
python -m coarsepgnn_release predict --assets assets --dataset openpoly \
  --model structure --target tg --input examples/openpoly_tg.jsonl \
  --output outputs/openpoly_example.csv --device cpu
python -m coarsepgnn_release predict --assets assets --dataset pu \
  --model structure --target tensile_strength_ultimate \
  --input examples/pu_tensile_strength_ultimate.jsonl \
  --output outputs/pu_strength_structure.csv --device cpu
python -m coarsepgnn_release predict --assets assets --dataset pu \
  --model hybrid --target tg --input examples/pu_tg.jsonl \
  --output outputs/pu_tg_hybrid.csv --device cpu
```

PU defaults to `formulation_graph_grouped`. Specify `--strategy record_random_interpolation` explicitly to use the random-partition checkpoint family. The four PU example files and the OpenPoly example each contain two archived inputs. They demonstrate execution, not prospective validation. Corresponding CPU outputs, including all five predictions, mean, and standard deviation, are supplied in `results/prediction_examples/`.

## New PU samples

A structure-only input references a raw coarse-graph JSON matching `src/coarsepgnn/sample_schema.json`:

```json
{"sample_id":"candidate_01","graph_file":"candidate_01/polymer_graph.json"}
```

Paths are relative to the JSONL file, unless absolute. Use the topology generator and inspect its validation report before inference. Do not pass a visualization-only PyG graph or a graph from an unrelated feature schema.

A Hybrid input additionally contains either a `condition_id` that identifies a vector in an NPZ, or an inline `condition` object as defined in Chapter 5. These alternatives are mutually exclusive. Its optional `assay` object uses the categorical keys listed in Chapter 9.

```json
{"sample_id":"candidate_01","graph_file":"candidate_01/polymer_graph.json","condition_id":"new_condition_01","assay":{"property_subtype":"tensile_strength_ultimate","property_task":"tensile_strength_ultimate","statistic_type":"unknown","definition_status":"specified"}}
```

For vectors produced by `encode`:

```bash
python -m coarsepgnn_release predict --assets assets --dataset pu \
  --model hybrid --target tensile_strength_ultimate --input candidates.jsonl \
  --condition-vectors outputs/new_conditions.npz \
  --output outputs/candidate_strength.csv --device cpu
```

If inline text is uncached, supply `--qwen-model` with the fixed snapshot directory. Never reuse another specimen's condition vector merely because its graph is similar. Supply assay fields according to the experimental definition, not a desired prediction.

## New OpenPoly repeat units

An OpenPoly input can reference an archived compatible `.pt` graph or explicitly supply:

```json
{"sample_id":"polyethylene_example","canonical_psmiles":"[*]CC[*]","h_capped_smiles":"CC"}
```

The canonical PSMILES must be a connected valid repeat unit with two monovalent connection sites. The hydrogen-capped structure must describe that same repeat unit. The adapter creates its fixed Chain100 graph; it does not infer a synthesis route or measured degree of polymerization.

## Output and averaging

The CSV reports identifiers and model selection, `prediction_fold_0` through `prediction_fold_4`, `prediction_mean`, `prediction_std`, and `unit`. For fold \(f\), first compute \(y_f=T_f^{-1}(\hat z_f)\). Then report:

\[
\bar y=\frac{1}{5}\sum_{f=0}^{4}y_f,\qquad
s=\sqrt{\frac{1}{4}\sum_{f=0}^{4}(y_f-\bar y)^2}.
\]

PU temperatures are emitted as `degC`, OpenPoly Tg as K, elongation as %, and strength as MPa. A JSON receipt records the input hash, five checkpoint hashes, device, and aggregation convention. Model disagreement is neither a calibrated prediction interval nor an applicability-domain score.

Missing or altered weights, mismatched identities, nonfinite features, duplicate sample IDs, and existing output files are reported as errors. A five-model mean on the training archive must not replace held-out OOF evaluation.

