# 6. Training the five-fold property models

## Prerequisites

Install the recorded dependencies, extract and verify the assets, and provide a new output directory. CUDA is the default for training; set `--device cpu` explicitly for CPU execution. Training is performed sequentially across folds. No distributed training configuration is assumed.

The commands below reproduce the archived learning procedures from fixed splits. They do not perform a new random partition. Fold indices are **0–4**. Omit `--fold` to run all five, or specify one fold when distributing independent jobs into separate output directories.

## OpenPoly Tg

```bash
python -m coarsepgnn_release train --assets assets --dataset openpoly \
  --model structure --target tg --output outputs/openpoly_training --device cuda
```

The four-layer model uses 256 hidden features, AdamW, learning rate 0.0003, weight decay 0.0001, maximum 200 epochs, and patience 20. Dynamic batches contain at most 3,200 nodes and 6,400 directed message edges. The fold seed is `20260817 + fold`. Target mean and population standard deviation are fitted on the training split only, in kelvin.

The archived five runs completed 59, 80, 73, 87, and 30 epochs; the selected epochs were 39, 60, 53, 67, and 10. Fresh runs can vary numerically across hardware. Compare the output metrics and split identities before drawing scientific conclusions.

To reconstruct the graph tensors from the archived canonical structure and hydrogen-capped repeat-unit index:

```bash
python -m coarsepgnn_release prepare-openpoly --assets assets \
  --output outputs/openpoly_graph_rebuild
```

This is a full 7,682-graph operation. It preserves the 100-node representation and fixed folds. Reusing verified archived graphs is sufficient for ordinary training.

## PU structure model

```bash
python -m coarsepgnn_release train --assets assets --dataset pu \
  --model structure --target tg --strategy formulation_graph_grouped \
  --output outputs/pu_structure_training --device cuda
```

The six-layer backbone uses learning rate 0.0002, weight decay 0.0001, batch size 64, dropout 0.15, up to 400 epochs, and patience 60. Training uses a 5% warm-up, cosine decay, gradient clipping at 5, and seed `20260901 + fold`. Checkpoint selection uses validation MAE in original units, with the archived tie-breaking rule.

PU Tg and Tm are standardized in degrees Celsius using training statistics. Mechanical targets use `log1p`; their inverse is the frozen `expm1(max(z,0))` rule. No target transform is fitted on validation or test data.

## PU Hybrid model and structural initialization

```bash
python -m coarsepgnn_release train --assets assets --dataset pu \
  --model hybrid --target tg --strategy formulation_graph_grouped \
  --output outputs/pu_hybrid_training --device cuda
```

By default, each Hybrid fold initializes from the corresponding archived structure checkpoint in `assets/models/pu`. It must match the target, strategy, fold, seed, and frozen dataset/fold hashes. A missing or incompatible structure checkpoint is an error.

To use structure models that you just trained, pass the directory containing the strategy subdirectories:

```bash
python -m coarsepgnn_release train --assets assets --dataset pu \
  --model hybrid --target tg --strategy formulation_graph_grouped \
  --structure-models outputs/pu_structure_training/formal \
  --output outputs/pu_hybrid_from_retrained_structure --device cuda
```

Inspect your actual generated directory hierarchy before supplying this path. The Hybrid residual is initialized at zero; the structural branch remains frozen. Hybrid training uses learning rate 0.0005, up to 250 epochs, and patience 40, with the same batch size, dropout, weight decay, warm-up, and clipping conventions. Its 48-dimensional condition cache uses `alpha_raw=0.5`. The seven assay-field vocabularies are fitted on training observations; unseen values map to index zero.

## All PU targets and both strategies

```bash
for strategy in record_random_interpolation formulation_graph_grouped; do
  for target in tg tm elongation_at_break tensile_strength_ultimate; do
    python -m coarsepgnn_release train --assets assets --dataset pu \
      --model structure --target "$target" --strategy "$strategy" \
      --output outputs/pu_all_structure --device cuda
    python -m coarsepgnn_release train --assets assets --dataset pu \
      --model hybrid --target "$target" --strategy "$strategy" \
      --structure-models outputs/pu_all_structure/formal \
      --output outputs/pu_all_hybrid --device cuda
  done
done
```

This schedules 80 property-model runs and can require substantial time. Each run writes the selected checkpoint, configuration, history, predictions, and metrics beneath its target/strategy/fold hierarchy. Use the existing 85 checkpoints if the objective is inference or result reconstruction. Release preparation did not repeat full training.

