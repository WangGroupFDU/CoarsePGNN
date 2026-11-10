# Validation record

## Published Zenodo v1.1.0 validation

The published [CoarsePGNN_VALIDATION_v1.1.0.json](https://zenodo.org/records/22741277/files/CoarsePGNN_VALIDATION_v1.1.0.json?download=1) reports checks of all six tasks, 15,239 task records, 90 exported fold tables and 95 property checkpoints (85 regression plus 10 phase). Archive preparation loaded all checkpoints and replayed two held-out samples per checkpoint on CPU. Full-data metrics were recomputed from saved predictions, not from a new full inference or retraining run.

The archive distinguishes five-fold means from pooled OOF metrics. Its [manuscript audit](https://zenodo.org/records/22741277/files/CoarsePGNN_MANUSCRIPT_AUDIT_v1.1.0.md?download=1) records PU R² reporting-convention discrepancies instead of changing data or predictions to force agreement. No new packages, full training run or cross-platform environment validation are claimed by that report.

These are **archived validation results**, not model checks rerun for this GitHub documentation update. The public record is published as v1.1.0; the report's frozen `remote_upload_verified: false` field describes its preparation-time scope, not the record's current publication status.

## Historical repository v1.0.1 checks

- Copied assets were checked by size and SHA-256 while assembling the independent release.
- All 85 property checkpoints were located, loaded, and used for forward predictions on two archived examples per model/target/strategy group.
- Those predictions agreed with the saved per-fold prediction tables under absolute tolerance 0.001 and relative tolerance 1e-5. Exact errors are in `results/validation/prediction_readback.csv`.
- Two cached conditions were reconstructed with the early encoder at alpha 0.5; maximum absolute vector difference was approximately 1.34e-7.
- The PU and OpenPoly fold loaders cover all five folds. Split coverage and formulation/graph-group isolation passed.
- Saved predictions were used to recompute 85 test-fold metric rows and 17 pooled OOF groups. The original MAE, RMSE, and R² values agreed within 1e-8.
- Source compilation passed in the existing Python 3.11.8 environment.
- Recomputed figures were generated as 600 DPI PNGs with Arial 7 pt and fixed-size canvases.
- Final asset manifests and ZIP hashes support independent download verification.

Machine-readable reports in `results/validation/` specify the checks actually completed. The 100-node PU topology tutorial passed. All ten explicit configurations were parsed and validated. Reproducing the selected Hybrid numerical head for 17 epochs produced tensors identical to the archived head; the full tuning search and explicit 3D showcases were not repeated.

## Historical v1.0.1 checks not performed

No new environment was installed, no Windows/macOS/CentOS 7 installation trial was performed, and the complete set of 85 property models was not retrained. No independent experimental test set was generated. Recomputed metrics are based on the archived predictions, and small-sample checkpoint checks are not a full new evaluation campaign.

The 20-configuration encoder tuning search, all 7,682 graph rebuilds, and all ten explicit 3D showcases were not rerun during release preparation. Commands for these operations preserve the source procedures and are documented for readers.

The v1.0.1 checks below remain historical evidence. They do not imply that the repository CLI supports the reorganized v1.1.0 Zenodo files; see [Chapter 3](03_assets.md).


## Packaging correction in 1.0.1

A real public download test on the release workstation exposed a case-insensitive filename collision between the original manifest.json and the new MANIFEST.json in version 1.0.0. Version 1.0.1 names the integrity file RELEASE_MANIFEST.json. Every ZIP member path is checked for case-folded uniqueness. Original data, cached vectors, and checkpoint bytes are unchanged.

The corrected archives passed full SHA-256 checks. Hybrid archive extraction on the macOS release workstation preserved both the original source manifest and the release integrity manifest; all 127 extracted files matched. In the existing Linux environment, all 9,109 asset files passed integrity checks and the PU Hybrid five-fold loading example passed again. This was an archive portability check, not a cross-platform environment installation trial.
