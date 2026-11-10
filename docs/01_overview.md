# 1. Project overview and method boundaries

## Prerequisites

Read this chapter before selecting a checkpoint. The repository's OpenPoly and PU workflows use different graph constructions and model architectures; their weights are not interchangeable. The separately packaged Zenodo archive also includes block-copolymer phase classification.

## Published Zenodo v1.1.0 scope

[Zenodo v1.1.0](https://zenodo.org/records/22741277) contains six fixed benchmark tasks, 15,239 task records and 90 full-row fold tables. The new phase task has 3,900 BCDB observations, 312 approximate graphs and five classes: lamellar, cylinder, gyroid, disordered and sphere. It uses 2,496/624/780 train/validation/test records per fold. These are task records, not independent-material counts.

The archive retains the original 85 regression checkpoints and adds 10 phase checkpoints. Its main `models/` contains 55 manuscript-task checkpoints; 40 grouped PU checkpoints are under `supplementary/grouped_validation`. Phase Hybrid requires the structure model and condition encoder from the same fold. The archive provides separate historical regression/classification source packages and an archived-ID prediction interface. See [Chapter 3](03_assets.md) for its layout and commands.

The sections below describe the **v1.0.1 repository workflows**, whose CLI and `results/` remain limited to OpenPoly/PU regression. They do not implement the archive's phase interface.

## Repository v1.0.1 experiments

| Dataset | Target | Task records | Model families | Split strategies | Checkpoints |
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

