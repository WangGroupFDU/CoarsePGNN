# PU inverse design: non-fitting audit, formal round 1

Source C: `b309fedd1e004bbe33b5041a53b9cec3037922f3`. Compute: cn2, existing base Python, 27.592120880959556 seconds. Formal training remains 0. This round is an audit, not research PASS or frozen success contract.

## Verified execution

- Eight boundary tests passed. All three execution checks passed and source/input hashes were rechecked.
- Recovered all 561 atomic Tg members of 246 cross-measurement means from the immutable pre-aggregation SQLite backup. Original and derived records remain traceable; means are not used in primary supervision. Some atoms remain excluded for scope or unresolved endpoints.
- Conservative rules currently retain 1998 primary observations and 6 auxiliary digitized observations. 250 grades are provisionally admissible; 8 are quarantined, including the pure-TDI / isomer-mixture conflict. 902 formulations satisfy the template plus current grade gate. These are provisional semantic decisions, not final approved training counts.
- Original split assignments were retained. All 12 material/formulation/source/actual-tensor intersections are zero. All 1162 original graph tensors were recomputed; an independent candidate graph reconstruction exactly reproduced every model-input tensor for one actual training material.
- Four actual training batches loaded through the new train/validation-only entry and produced finite random-weight forward outputs. No optimizer update occurred. Normal development test reads: 0; deliberate negative control was refused.
- Old supervised weights are not reused. Development wiring is implemented; the complete training CLI/loop and final evaluation execution remain to be built after the contract is frozen.

## Independent review found an unresolved precision/endpoint problem

The file `training_measurement_precision.json` currently calls within-campaign pairs comparable repeats. This classification is NOT valid: first/second heating and multiple thermal peaks can share a campaign/property-task key. Its Tg median difference 5 K and Tm 46.7 K MUST NOT be used as experimental uncertainty or as a success-threshold basis. Preserve this original output as evidence of the problem.

Examples from the training-only evidence:
- MEA_00622204543a8c0a6be3_ATOMIC and MEA_6ed731656de35e06c403_ATOMIC explicitly refer to first and second heating scans; they are different protocols, not repeats.
- MEA_01e11935e556232ee06b and MEA_faa74e0bd9bb4b6eb398 have 111 and 5 degrees C but share generic hard-segment/DSC-peak metadata. Their peak identities require source-level resolution or quarantine. Do not average them.

The next revision must give explicit scan/peak identities separate protocol/assay fields, quarantine unresolved same-key thermal labels, and only call measurements repeats when independent replicate identity is actually supported. A report precision is not an experimental standard deviation.

Cross-campaign common-state links remain unresolved. Current approved states are limited to the same verified measurement campaign; no material-ID-only merging is performed. The original 48 four-target specimen candidates are not approved as true four-target joint measurements. Joint-state coverage and physical precision remain insufficient for a frozen scientific contract.

## Required reviewer decisions

Please return REVISE, not PASS, if the above repair is needed. Authorize a narrow second non-fitting audit to resolve scan/peak collisions and produce the corrected denominators, then complete the numeric success contract before any fitting. The scope remains four-property computational inverse design with independent single-property measured anchors; new combinations are not experimental success. Broad four-property population generalization cannot be claimed from the small independent-source denominator. Do not lower thresholds based on test performance; no test performance was computed.

## Evidence and access

`measurement_adjudication.csv` preserves the disposition of every original record plus recovered atoms. `tg_mean_membership.csv` preserves original measurement IDs and protocols. Validation/test values are withheld from the exported projections; the split files remain remote, with hashes in `split_labels_manifest.json`. `training_evidence.json` gives training-side definitions, evidence text and locators for actual review. `evaluation_denominators.json` gives counts only for held-out splits. The test Tm set currently has 13 materials in 4 closure groups; these counts may change only through pre-training semantic rules, never performance-based exclusions.

Twenty-one result files totaling 11848608 bytes were downloaded and independently SHA-256 checked. All required files are below 100 MiB, so this audit uses GitHub only. Server UTC timestamps are preserved as reported; cumulative compute accounting uses elapsed monotonic wall time.
