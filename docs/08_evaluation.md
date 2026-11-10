# 8. Reconstructing prediction results and figures

**Scope:** This chapter describes the repository software and GitHub assets at v1.0.1. For the separately packaged Zenodo v1.1.0 archive and its archived-ID interface, see [Chapter 3](03_assets.md).

## Prerequisites

The GitHub `results/` directory contains the original per-fold predictions and training histories for all 85 included runs. Metric reconstruction requires NumPy and SciPy but does not require model weights, Qwen, CUDA, or retraining.

## Recompute numerical results

```bash
python -m coarsepgnn_release evaluate --results results \
  --output outputs/recomputed_metrics
```

The program checks prediction coverage and recomputes MAE, RMSE, R², and Spearman correlation in original target units. MAE, RMSE, and R² are compared with saved run metrics before the release summaries are written.

| Output | Interpretation |
|---|---|
| `metrics_by_split.csv` | Training, validation, and test metrics for every run |
| `metrics_by_fold.csv` | The 85 held-out test-fold results |
| `metrics_fold_mean_std.csv` | Unweighted mean and sample standard deviation across five fold metrics |
| `metrics_oof.csv` | Metrics computed on pooled held-out predictions for 17 model/strategy/target groups |
| `oof_predictions.csv` | Exactly one held-out test prediction per observation within each group |
| `verification_report.json` | Run count, OOF coverage, comparison tolerance, and completion status |

Pooled OOF R² generally differs from the average of five R² values. The two quantities are retained separately. Temperature error magnitudes in K and °C are numerically equal, while absolute predictions differ by 273.15.

## Generate publication-style figures

```bash
python -m coarsepgnn_release evaluate --results results \
  --output outputs/recomputed_with_figures --plots \
  --arial-font /path/to/Arial.ttf
```

If Arial is already registered, omit `--arial-font`. The command exports 17 OOF parity plots and 17 five-fold loss plots as PNG only, with Arial 7 pt, 0.75 pt axes, 600 DPI, and fixed canvases. Parity figures are 2.3 × 2.3 inches; history figures are 4.6 × 2.3 inches. Solid histories denote training loss and dashed histories denote validation loss in transformed target space.

## Expected results

The recomputed pooled OOF OpenPoly Tg result is **MAE 24.9131 K, RMSE 38.1995 K, R² 0.885948**, over 7,682 structures. For grouped PU Hybrid models:

| Target | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg | 803 | 11.2634 K | 17.6416 K | 0.841652 |
| Tm | 954 | 23.0565 K | 32.5827 K | 0.728572 |
| Elongation at break | 1,032 | 203.6660 percentage points | 343.2153 percentage points | 0.332726 |
| Ultimate tensile strength | 868 | 8.67935 MPa | 12.2653 MPa | 0.399963 |

All structure-only results, random-split results, and individual fold metrics remain available in `results/recomputed/`. These are reconstructions of the archived experiments, not results from a new training campaign.

## Interpretation

OOF predictions use only the checkpoint for the sample's held-out outer fold. Do not calculate cross-validation metrics from the five-checkpoint ensemble on these same samples: four checkpoints typically saw a given outer-test sample during training or validation.

The scripts retain weak and negative results rather than selecting only favorable folds. Grouped and random results answer different generalization questions. Neither split establishes independent experimental validation, ionic-polymer extrapolation, or calibrated uncertainty.

