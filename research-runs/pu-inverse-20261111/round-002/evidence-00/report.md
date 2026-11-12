# Round 002: endpoint and access corrections

Code C: `843384339cb4af923f077e9e567a648a3a727725`. Actual cn2 job `round-002-e0b5d020` completed in 27.881088990950957 seconds. Cumulative budget: 2/10 rounds, 55.47320987191051/259200 seconds. Formal model fitting remains zero; success contract remains DRAFT_NOT_FROZEN.

## Executed and independently checked

- 21 targeted tests passed. The sequential runner stops on any failed check; all three subprocess checks exited 0.
- Recovered 561 atomic Tg observations from 246 cross-measurement means. All 4218 adjudication IDs close against round 001, without resplitting.
- Current provisional classifications: 1178 primary, 5 auxiliary, 1935 unresolved, 1100 rejected. There are 239 provisionally approved grades and 19 quarantined grades; 882 template/grade-eligible materials.
- Independent readback recomputed each split/target observation, material and closure-group count from measurement_adjudication.csv. Non-training value fields are blank. All 28 received artifacts match worker size/SHA-256; transfer total 22176988 bytes.
- The first/second-heating Tg counterexample remains primary with dsc_inflection (not relabelled midpoint), different condition hashes, actual raw_prompt_v2 cache keys, numeric scan tags 1/2 and train assay indices 5/6. No condition embeddings or supervised heads were fitted.
- Both unidentified Tm-peak counterexamples and both strength-negation counterexamples are now UNRESOLVED.
- Normal test reader and underlying file-open attempts are both 0. Reader/direct-open negative controls are blocked. A train entry pointing to sealed test is refused before hash/content access.
- All 1162 original graph tensors were re-audited; one independently rebuilt real training graph exactly matches every tensor. Four random no-gradient forward batches are finite. These are engineering checks, not training performance.

## Current evidence denominators

| Split | Target | Observations | Materials | Independent closure groups |
|---|---|---:|---:|---:|
| train | elongation_at_break | 391 | 360 | 84 |
| train | tensile_strength_ultimate | 200 | 192 | 37 |
| train | tg | 67 | 38 | 7 |
| train | tm | 85 | 61 | 23 |
| validation | elongation_at_break | 77 | 77 | 7 |
| validation | tensile_strength_ultimate | 71 | 71 | 4 |
| validation | tg | 32 | 16 | 1 |
| validation | tm | 7 | 7 | 3 |
| test | elongation_at_break | 121 | 114 | 9 |
| test | tensile_strength_ultimate | 110 | 99 | 4 |
| test | tg | 6 | 3 | 1 |
| test | tm | 11 | 11 | 3 |

Tg validation/test contain only 1 independent closure group each; test has 3 materials. All three Tg test materials are unseen grades under the supervised definitions. Tm validation/test have 3 closure groups each. These limits cannot support a general population claim or a meaningful group-bootstrap interval for Tg; neither rows nor model seeds count as independent sources.

## Scope and uncertainty decisions requiring Pro adjudication

- Four-target and three-target approved common states remain zero in every split. Approved two-target states are train 177 / validation 61 / test 106. Cross-campaign states remain unresolved; no union creates joint labels.
- All four targets have zero qualified independently identified replicate pairs, so repeatability estimates are null. Round 001 Tg=5 K / Tm=46.7 K estimates are explicitly superseded. Training uncertainty records comprise 4 source-reported SD, 9 ranges, 193 unspecified plus/minus and 537 missing; no SE/SD conversion was made.
- Material-equal quantiles use each material total weight 1 and the inverse weighted empirical CDF; source labels are never averaged across protocols. Per-endpoint/condition strata are retained.
- IMPORTANT candidate-scope question: the revised alcohol-OH SMARTS counts O-H on sp3 carbon. Consequently phenolic bifunctional extenders (e.g. bisphenol A, resorcinol and bisphenol AF) are now conservatively UNRESOLVED, although the broader frozen scope may permit their source-proven PU reactivity. This is a potential over-exclusion, not proof that these source molecules are non-bifunctional. Pro must adjudicate whether the candidate predicate should count two source-qualified non-carboxylic hydroxyls with alcohol/phenol subtype recorded, before freezing the domain. DMPA is separately quarantined for extra carboxylic-acid reactivity, with two alcohol OH and one carboxyl OH correctly distinguished.
- Thermal collision grouping is intentionally conservative and does not select peaks by numerical rank. Review source phase assignments and quarantine rules rather than accepting the reduced counts as proof of correctness.
- Full train/design/evaluate, optimizer updates, fitted weights, calibration, unified-budget baselines, ablations and seed comparisons do not yet exist. The final one-time test window remains unopened. Old task-supervised assets remain prohibited.
- A first collect attempt occurred before final receipt availability and returned download_failed. Read-only inspection found the same original job had completed successfully; its result was then collected. No job was resubmitted.

## Requested decision

Read the fixed code and evidence, adjudicate the identified phenolic-extender domain question and any remaining necessary corrections. If evidence is sufficient, provide the complete numeric pre-training success contract for computational four-target search and finite heldout single-target anchors, distinguishing design tolerances from measurement noise. A lack of repeatability estimates alone must not cause indefinite audit loops. If a genuine requirement cannot be supported, name the specific blocker without changing splits, thresholds after testing, the four-target scope or the no-new-synthesis boundary. No research PASS is requested at this stage.
