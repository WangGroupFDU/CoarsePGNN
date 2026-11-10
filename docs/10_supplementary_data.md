# 10. Supplementary COF and dendrimer literature data

## Scope and access

The public [Zenodo v1.1.0 record](https://zenodo.org/records/22741277), DOI [10.5281/zenodo.22741277](https://doi.org/10.5281/zenodo.22741277), contains these literature collections inside `CoarsePGNN_supplementary_v1.1.0.zip`. They are not separate top-level COF/dendrimer ZIP downloads in this published record.

The collections preserve the validated `STANDARDIZED_20260910T054902Z` sources. They do not participate in the six manuscript prediction benchmarks and add no COF/dendrimer property checkpoints. The same supplementary ZIP also retains **40 grouped-validation PU checkpoints and their splits** under `supplementary/grouped_validation`; these are separate from the literature collections. Across the complete v1.1.0 archive there are 95 property checkpoints (55 main and 40 grouped).

The embedded collection READMEs retain their historical supplement version 1.0.1 and references to the earlier 85-checkpoint release. Those frozen statements describe source-package history, not the version or total model count of the enclosing Zenodo v1.1.0 archive. The repository's legacy `coarsepgnn_release download` does not retrieve this supplementary ZIP.

| Dataset | Materials | Parent measurements | Explicit curve points | All standardized observations | Eligible observations |
|---|---:|---:|---:|---:|---:|
| COF | 3,827 | 35,930 | 0 | 35,930 | 453 |
| Dendrimer | 409 | 992 | 195 | 1,187 | 303 |

The dendrimer archive additionally contains `unadmitted_candidates/`: 208 source records and 244 extracted observations. The archived admission decision admitted no candidate bundle. These candidates are not included in the standardized or eligible counts above.

## Files and extraction

Follow [Chapter 3](03_assets.md#zenodo-v110-download-and-extraction) to download and verify the v1.1.0 archives. The supplementary ZIP is **822,156,436 bytes**, with SHA-256 `0f18e57c4881ba39247b6a9278717df66809a0e4cf205814572b4b469eef117d`.

If it has not already been extracted with the other v1.1.0 parts, extract it into the same parent directory:

```bash
python -m zipfile -e downloads/v1.1.0/CoarsePGNN_supplementary_v1.1.0.zip archive_v1.1.0
```

From the repository root, inspect and verify the actual extracted directories (note the plural `dendrimers`):

```bash
python archive_v1.1.0/CoarsePGNN/supplementary/cof/scripts/inspect_dataset.py \
  --root archive_v1.1.0/CoarsePGNN/supplementary/cof --verify
python archive_v1.1.0/CoarsePGNN/supplementary/dendrimers/scripts/inspect_dataset.py \
  --root archive_v1.1.0/CoarsePGNN/supplementary/dendrimers --verify
```

The scripts check each collection's `RELEASE_MANIFEST.json` and then report database counts. Each collection is under the shared `CoarsePGNN/` archive root, not the old `supplementary_data/` directory. Do not overwrite an existing extracted copy.

Python 3.11 and its standard library are sufficient to verify the archives and read SQLite/CSV/JSON data. The archived standardization and validation routines require RDKit 2025.03.2. Each collection contains its own English README, detailed `USAGE.md`, `DATA_DICTIONARY.md`, complete schema, original scripts, and environment record.

## Scientific interpretation

Raw and standardized SQLite files retain their exact source bytes. All observations, including incomplete records, remain archived. The standardized `training_eligible` flag uses the frozen `user-definition-20260910-v1` rule; historical raw-table acceptance flags are not equivalent.

The rule requires usable synthesis-monomer structures, an explicit feed-ratio basis, numeric property information with interpretable units, relevant test conditions, and either COF topology or dendrimer generation. Eligibility does not imply independent observations, a complete three-dimensional structure, an exact scalar label, or a validated predictive model. Preserve ranges and bounds, distinguish curve points from parent measurements, and group correlated records by material/source/route/curve when creating new evaluations.

Unresolved SMILES, uncertain units, missing conditions, source locators, and exclusion reasons remain explicit. Reported `m/z` values must not be interpreted as neutral molar mass without the necessary ion/adduct assignment. The unadmitted candidate directory must remain outside supervised training until its unresolved evidence is addressed.

## Verification and citation

The historical supplement preparation reran the archived validators on independent copies. SQLite integrity and foreign keys, original-table preservation, observation coverage, independent numeric conversions, eligibility, and CSV counts passed for both datasets. The historical supplement manifests and ZIP CRC checks passed. The complete standardization process and model training were not rerun; no new environment or cross-platform installation was tested.

Archive creator: Jifeng Wang, Fudan University. Original curation is CC BY 4.0 and original scripts are MIT. Third-party source rights remain unchanged. Cite the underlying literature when reusing individual measurements. Cite the enclosing dataset archive as v1.1.0 using [doi:10.5281/zenodo.22741277](https://doi.org/10.5281/zenodo.22741277). The public v1.1.0 validation report describes archive-wide checks; this documentation update does not claim a new standardization, model-training, or full dataset-validation run.

