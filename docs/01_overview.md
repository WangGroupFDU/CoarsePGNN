# 1. Project overview and method boundaries

## Prerequisites

Read this chapter before selecting a checkpoint. The two property tasks use different graph constructions and model architectures; their weights are not interchangeable.

## Included experiments

| Dataset | Target | Independent inputs | Model families | Split strategies | Checkpoints |
|---|---|---:|---:|---:|---:|
| OpenPoly | Tg | 7,682 canonical homopolymer structures | 1 | 1 | 5 |
| PU | Tg | 803 observations | 2 | 2 | 20 |
| PU | Tm | 954 observations | 2 | 2 | 20 |
| PU | Elongation at break | 1,032 observations | 2 | 2 | 20 |
| PU | Ultimate tensile strength | 868 observations | 2 | 2 | 20 |

OpenPoly uses a fixed 100-repeat-unit chain proxy with 99 coarse edges. PU uses archived formulation-derived graphs and, for Hybrid models, explicit test-condition information. A node describes a monomer or repeat-unit instance through fingerprints, numerical attributes, and reaction-site features. The released graph schema specifies tensor shapes and masking.

The PU structure model uses a six-layer residual GINE backbone with a virtual node and multilevel pooling. PU Hybrid freezes the corresponding trained structural predictor and learns a gated residual using the 48-dimensional Hybrid vector and seven categorical assay descriptors. OpenPoly uses the four-layer CoarsePGNN v2 implementation, with a zero 512-dimensional condition input. That placeholder is not the PU Hybrid representation.

## Reproducible experimental protocol

The preserved outer test folds partition each target dataset once. Validation is drawn from the remaining training pool, giving approximately 72% training, 8% validation, and 20% test data per fold. Exact counts are in `results/split_counts.csv`. Nothing is re-randomized during release preparation.

PU `record_random_interpolation` measures performance on randomly held-out records; related formulations can occur across splits. `formulation_graph_grouped` keeps the recorded formulation/graph groups separate. Group separation is not automatically paper separation or an independent prospective experiment.

## Input and output example

```bash
python -m coarsepgnn_release --help
python -m coarsepgnn_release predict --help
```

These commands describe the supported interfaces without loading model weights. A prediction request selects one dataset, target, model, and strategy and requires all five matching checkpoints.

## Scientific limits

A finite graph is a representation, not an atomistic simulation of polymerization kinetics. The automatic generator grows trees; the explicit backend can impose finite cyclic or branched structures. Explicit topology feasibility does not establish property-model applicability. Periodic COFs, equilibrium conformations, and molecular dynamics are outside this release.

Molar-mass descriptors may characterize a soft segment or starting component. They do not establish the final polymer Mn, Mw, or dispersity. The fixed OpenPoly chain length is a representation choice, not measured molecular weight.

Only the early encoder matching the trained PU models is included. No text-only PU property model, later encoder replacement, competing property benchmark model, or new training result is introduced.

