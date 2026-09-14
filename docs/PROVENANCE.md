# Release provenance and source reconciliation

## Frozen inputs

| Component | Source identity |
|---|---|
| PU experiments | `PU_V24_8MODEL_5FOLD_ALPHA50_20260901T140034Z_68742fed` |
| OpenPoly five-fold experiments | `OPENPOLY_TG_COARSEPGNNV2_CHAIN100_5FOLD_20260902T003458Z_8a43fabd` |
| Early Hybrid encoder | `PU_CONDITION_ENCODERS_V2_EXT16_20260824T104653Z_0fb2aa90` |
| PU underlying database export | `PU_DATABASE_V2_4_B0024_20260901T060554Z_68742fed` |
| Qwen revision | `959bed3521cc1e8714392c77e15ae213b00aa921` |

The release was assembled in a separate directory. Original source, weights, data, logs, and experiment directories were retained. File copies were checked against their recorded sources. Archive manifests provide the final released identities.

## PU source reconciliation

The current working code had evolved after the selected training run. The original training environment recorded hashes that matched the historical `pu_v24_text_only_20260902T004200Z` backup for the following three files. The backup label does not describe the released model scope; the selected file contents implement structure-only and full Hybrid modes.

| Original file | SHA-256 before portability edits |
|---|---|
| `data.py` | `db81014a8749c2102a186c157304216026a083e27e15d064bc917729c15a04a3` |
| `model.py` | `b1455453e2d16d39405bfe6d6ad0b21b033c1f491493ff5e8f975c600380f830` |
| `train.py` | `5b815f054070cbe238b96da4bfa3071545e658b49c3386cdaa56b0ea6229a553` |

## Release edits

The release adds explicit path/device interfaces, mean-of-five prediction, cached condition encoding, metric reconstruction, asset download/integrity checks, and documentation. PU training resolves historical graph paths within the extracted asset directory while preserving manifest bytes and checkpoint hashes. It does not change the structural or Hybrid architecture.

OpenPoly training, fold loading, analysis, and acceptance loops now accept folds 0–4. Its analysis and acceptance reports cover only the released CoarsePGNN model and do not require out-of-scope benchmark models. Validation reads the assets without rewriting their manifests. Graph reconstruction uses the archived hydrogen-capped SMILES index and no longer requires another benchmark model's input directory. Package validation uses the released source manifest and the unchanged graph schema.

The topology cache is placed under the selected output directory. An external OpenBabel executable is resolved through PATH rather than a private installation path. Public command descriptions and generated release explanations are presented in English.

The final code-to-source correspondence is recorded in the release source manifest. Historical paths inside original checkpoint/provenance metadata are descriptive; public execution does not access those servers.

