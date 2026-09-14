# 10. Supplementary COF and dendrimer literature data

## Scope and access

The [Zenodo draft 22741277](https://zenodo.org/uploads/22741277) also contains two supplementary literature datasets. The draft requires owner access until publication. Its supplementary files are not yet public; the fixed OpenPoly/PU assets remain publicly available from GitHub release v1.0.1.

The supplements preserve the validated `STANDARDIZED_20260910T054902Z` sources. They provide no additional property-model checkpoints and do not alter the 85 CoarsePGNN property models, their splits, their condition features, or reported metrics. The `coarsepgnn_release download` command continues to select the four core reproducibility archives.

| Dataset | Materials | Parent measurements | Explicit curve points | All standardized observations | Eligible observations |
|---|---:|---:|---:|---:|---:|
| COF | 3,827 | 35,930 | 0 | 35,930 | 453 |
| Dendrimer | 409 | 992 | 195 | 1,187 | 303 |

The dendrimer archive additionally contains `unadmitted_candidates/`: 208 source records and 244 extracted observations. The archived admission decision admitted no candidate bundle. These candidates are not included in the standardized or eligible counts above.

## Files and extraction

After the Zenodo record is published, download the relevant ZIP and `SUPPLEMENT_MANIFEST.json`, `SUPPLEMENT_SHA256SUMS.csv`, and `SUPPLEMENT_README.md` from that record. Validate the archive size and SHA-256 against the manifest before extraction.

| File | Bytes | SHA-256 |
|---|---:|---|
| `CoarsePGNN_cof_literature_v1.0.1.zip` | 45,728,402 | `f8ce58d60c957b64da1c120d2a8eeeacab40cd4f8f9e879e5de2dc19c226ca9e` |
| `CoarsePGNN_dendrimer_literature_v1.0.1.zip` | 8,468,055 | `a7583b14e0be390791f98a6ee7d39eab3b7ecc47f55e3a6a6eee5c626fde55d2` |

```bash
python -m zipfile -e downloads/CoarsePGNN_cof_literature_v1.0.1.zip supplementary_data
python -m zipfile -e downloads/CoarsePGNN_dendrimer_literature_v1.0.1.zip supplementary_data
python supplementary_data/cof/scripts/inspect_dataset.py --root supplementary_data/cof --verify
python supplementary_data/dendrimer/scripts/inspect_dataset.py --root supplementary_data/dendrimer --verify
```

Python 3.11 and its standard library are sufficient to verify the archives and read SQLite/CSV/JSON data. The archived standardization and validation routines require RDKit 2025.03.2. Each archive contains its own English README, detailed `USAGE.md`, `DATA_DICTIONARY.md`, complete schema, original scripts, and environment record.

## Scientific interpretation

Raw and standardized SQLite files retain their exact source bytes. All observations, including incomplete records, remain archived. The standardized `training_eligible` flag uses the frozen `user-definition-20260910-v1` rule; historical raw-table acceptance flags are not equivalent.

The rule requires usable synthesis-monomer structures, an explicit feed-ratio basis, numeric property information with interpretable units, relevant test conditions, and either COF topology or dendrimer generation. Eligibility does not imply independent observations, a complete three-dimensional structure, an exact scalar label, or a validated predictive model. Preserve ranges and bounds, distinguish curve points from parent measurements, and group correlated records by material/source/route/curve when creating new evaluations.

Unresolved SMILES, uncertain units, missing conditions, source locators, and exclusion reasons remain explicit. Reported `m/z` values must not be interpreted as neutral molar mass without the necessary ion/adduct assignment. The unadmitted candidate directory must remain outside supervised training until its unresolved evidence is addressed.

## Verification and citation

Release preparation reran the archived validators on independent copies. SQLite integrity and foreign keys, original-table preservation, observation coverage, independent numeric conversions, eligibility, and CSV counts passed for both datasets. The new manifests and ZIP CRC checks passed. The complete standardization process and model training were not rerun; no new environment or cross-platform installation was tested.

Archive creator: Jifeng Wang, Fudan University. Original curation is CC BY 4.0 and original scripts are MIT. Third-party source rights remain unchanged. Cite the underlying literature when reusing individual measurements. A final Zenodo DOI will be added after approval and publication of this specific draft.

