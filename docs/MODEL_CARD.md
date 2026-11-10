# Model card

## Models and interfaces

The repository-compatible v1.0.1 release contains 85 regression checkpoints and one frozen PU Hybrid numerical head. Checkpoints are grouped by dataset, split strategy, model family, target, and fold. Public prediction refuses an incomplete five-fold set.

| Model | Architecture and inputs | Weights |
|---|---|---:|
| OpenPoly CoarsePGNN-v2-Chain100 | Four-layer GINE, hidden dimension 256, fixed Chain100 graph, zero 512-vector condition input | 5 |
| PU CoarsePGNN-structure-only | Six-layer residual GINE, virtual node, multilevel pooling, structural regression head | 40 |
| PU CoarsePGNN-FULL | Frozen same-fold structural model plus gated condition-dependent residual; 48-vector and seven assay categories | 40 |
| Early Hybrid numerical head | PLE plus numerical DeepSets head; frozen Qwen text branch remains external | 1 |

The published source preserves model state keys, tensor dimensions, reaction semantics, feature definitions, and target transforms. The early encoder checkpoint metadata records alpha 0.25; PU feature construction explicitly uses alpha 0.5.

## Published v1.1.0 archive models

[Zenodo v1.1.0](https://zenodo.org/records/22741277) retains those 85 regression checkpoints unchanged and adds 10 phase checkpoints. The 95 property checkpoints comprise 55 main manuscript-task models (5 homopolymer, 40 PU, 10 phase) and 40 grouped-validation PU models. Encoder heads are separate from this property-checkpoint count.

The five phase Hybrid checkpoints are conditional residual models requiring their same-fold structure checkpoint and condition encoder. The phase encoders must not be exchanged between folds or replaced with the PU encoder. Exact paths, pairings and hashes are in the archive's `documentation/model_index.json`.

Use the bundled `documentation/predict.py` to replay archived IDs one fold at a time. The repository's v1.0.1 CLI supports OpenPoly/PU and a different asset layout, not phase classification. See [Chapter 3](03_assets.md) for compatibility and [VALIDATION.md](VALIDATION.md) for the distinct validation records.

## Training and evaluation

Each target is trained independently. Train/validation/test partitions are fixed. Structural initialization for PU Hybrid must match the fold's data hashes and seed. Label normalization and assay vocabularies are fitted using the designated training split.

Checkpoint selection uses the archived validation criterion. Test labels are not used for checkpoint selection. The evaluation tutorial reports both test-fold metrics and pooled OOF metrics. A five-fold ensemble is provided for prediction of new inputs, not for replacing held-out cross-validation estimates.

## Outputs

PU temperature predictions are in degrees Celsius; OpenPoly Tg is in kelvin. Mechanical predictions are in percent and MPa. Inverse target transformations are applied independently before averaging. Reported fold standard deviation is model disagreement, not calibrated uncertainty.

## Reproduction and limitations

All 85 checkpoints were loaded and used for two-sample CPU predictions in the existing Linux environment. Outputs were compared with archived predictions at absolute tolerance 0.001 and relative tolerance 1e-5. Cached Hybrid reconstruction was also compared numerically. These checks do not constitute validation on another computer or a fresh training reproduction.

The release does not include a text-only PU property model or competing property models. Support for a graph schema does not establish predictive validity on every representable topology, chemistry, condition, or molecular-weight regime. Consult the data card before interpreting grouped versus random performance.

Original model weights use CC BY 4.0. Qwen base weights are downloaded separately under their upstream license.

