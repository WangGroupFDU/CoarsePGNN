# Data card

## Identity and intended use

This card describes the repository-compatible v1.0.1 OpenPoly/PU assets accompanying 85 regression checkpoints. The archive author is Jifeng Wang, Fudan University. Data curation and derived assets use CC BY 4.0, subject to third-party rights and attribution.

The data support reconstruction of the specified cross-validation experiments and inference within a scientifically assessed domain. They do not establish prospective experimental validation or comprehensive chemical coverage.

## Published v1.1.0 archive

[Zenodo v1.1.0](https://zenodo.org/records/22741277) preserves these regression data and adds 3,900 BCDB phase observations. Its six tasks total 15,239 task records, with 90 full-row train/validation/test tables. Each task has `data.csv`, `splits.csv`, and `folds/fold_1` through `fold_5`. New `archive_graph_file` paths are relative to its `CoarsePGNN/` root; historical provenance paths remain unchanged.

The phase class counts are 2,122 lamellar, 764 cylinder, 559 gyroid, 389 disordered and 66 sphere. Phase splits contain 2,496/624/780 records per fold. Random record splitting permits material overlap; approximately 94–96% of phase test observations have materials also present in training, as documented in the archive README. The task-record total is not an independent-material count, and these evaluations do not demonstrate unseen-material generalization.

COF/dendrimer literature data remain supplementary and outside the six benchmarks. See [Chapter 3](03_assets.md), [Chapter 10](10_supplementary_data.md), and the archive's `documentation/DATA_DICTIONARY.md` for current paths and fields. The remainder of this card retains the original v1.0.1 data definitions.

## OpenPoly

A frozen `experiment_polymer_data.xlsx` snapshot contains 21,394 input rows, of which 12,943 have finite Tg values. The recorded canonicalization and validity filters retain 12,894 member observations belonging to 7,682 canonical connected homopolymer repeat-unit structures. Each accepted structure has two monovalent connection sites. Arithmetic-mean Tg in kelvin is the target; `tg_std_K` is a population standard deviation. There are 8,500 excluded source rows.

`structures_tg.csv` contains one row per canonical structure. `structure_members.csv` retains the associated input observations and provenance. `exclusions.csv` retains excluded-row identities and reasons. `graph_index.csv` maps each structure to one fixed Chain100 tensor and its hydrogen-capped SMILES. Canonical structure IDs, source row numbers, source IDs, names, and references are distinct concepts.

The target-sorted rank-decile stratification and seed 20260817 were fixed before training. All observations of the same canonical structure are aggregated before splitting. The five outer test partitions are disjoint, and validation is 10% of the outer training pool.

## PU

The selected interface contains 3,657 measurements: 803 Tg, 954 Tm, 1,032 elongation-at-break, and 868 ultimate-tensile-strength observations. Its graphs occupy 1,162 unique material/graph paths; a graph basename alone is not guaranteed to identify one file.

`dataset_manifest.csv` preserves observation, measurement, material, formulation, source, campaign, and graph identities. It also retains raw and normalized conditions, structural components, assay definitions, label units, source hashes, and screening flags. Historical adaptability flags describe original screening and do not add other property models to this release.

`fold_manifest.csv` contains the two fixed five-fold strategies. Grouped rows carry a group ID that enforces formulation/graph isolation between train, validation, and test. The random strategy permits related structures across partitions. Neither strategy is asserted to exclude every same-paper relationship.

`provenance/` contains the selected sources, materials, components, material-component amounts, measurement values and original units, test campaigns, graph components, and evidence locators. These are relational exports from the immutable database snapshot, restricted to the selected training observations and their dependencies. The original database hash and table counts are recorded in `export_provenance.json`.

## Conditions and missingness

Raw text is retained separately from standardized JSON. Numerical features are admitted through reported-field masks and explicit units. Missingness is not replaced with an observed zero. Conflicts and source-status flags remain auditable. The source observation itself is not rewritten when creating a model vector.

The 48-dimensional PU cache uses the early encoder and alpha 0.5. Numeric-head training records, benchmark development/test assignments, PLE knots, prompts, and tuning results are in the Hybrid part. Property-training folds and encoder-development folds are different partitions for different tasks.

## Quality checks and limitations

Release checks verify hashes, referenced graph files, label counts, fold coverage, grouped isolation, checkpoint identities, and archived metrics. Raw source records and references are preserved rather than retrospectively curated to improve scores. Missing or ambiguous scientific context can limit transfer to new samples even when a graph is technically valid.

Component molecular weights and soft-segment Mn values must not be interpreted as experimentally established final-PU Mn or Mw. Chemical charge and ionic behavior require separate assessment; absence of an explicit charged node does not prove that a source experiment was nonionic.

