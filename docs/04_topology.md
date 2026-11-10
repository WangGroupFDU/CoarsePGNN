# 4. Generating polymer topology

**Scope:** This chapter describes the repository software and GitHub assets at v1.0.1. For the separately packaged Zenodo v1.1.0 archive and its archived-ID interface, see [Chapter 3](03_assets.md).

## Prerequisites

Install the pinned RDKit, qmdesc, SMiPoly, PyTorch, PyG, NetworkX, and plotting dependencies. Use neutral, single-fragment **raw monomer SMILES** for qmdesc. Wildcard PSMILES and raw SMILES have different meanings and must remain separate fields.

## Automatic tree generation

`examples/topology/system_PU.json` contains a diisocyanate and two diols with relative molecule counts of 10:5:5. Its target is 100 nodes, its minimum completion ratio is 0.90, and its random seed is 7.

```bash
python -m polymer_workflow generate \
  --input examples/topology/system_PU.json \
  --output-dir outputs/topology_tree --device cpu
```

To prepare another input, use this JSON structure:

```json
{
  "polymer_id": "diisocyanate_diol_example",
  "monomers": [
    {"name": "HDI", "smiles": "O=C=NCCCCCCN=C=O", "ratio": 1},
    {"name": "BDO", "smiles": "OCCCCO", "ratio": 1}
  ],
  "target_nodes": 100,
  "min_completion_ratio": 0.90,
  "random_seed": 7,
  "reaction_rule_id": 111
}
```

Ratios determine integer monomer inventories by the archived allocation rule. They are molecule ratios, not mass fractions. Supply distinct monomer names and an explicit reaction rule if chemical classification yields more than one valid route.

SMiPoly first identifies compatible functional groups. The generator then evaluates qmdesc descriptors on the original monomers, retains atom identity through `orig_id`, labels electrophilic and nucleophilic ports, and scores only chemically permitted site pairs. Each growth step attaches one unused monomer with one coarse edge, giving a connected acyclic graph. The output is not a prediction of reaction rate, conversion, or equilibrium product distribution.

### Supported reaction templates

| Rule | Template |
|---|---|
| 6, 105 | Hydroxy-acid condensation |
| 7, 109 | Amino-acid condensation |
| 104 | Diacid–diol polycondensation |
| 108 | Diacid–diamine polycondensation |
| 110 | Dianhydride–diamine polyimide formation |
| 111 | Diisocyanate–diol polyurethane formation |
| 112 | Cyclic-anhydride/epoxide ring-opening route |
| 113 | Diepoxide–diisocyanate polyoxazolidone formation |
| 114, 115 | Activated aromatic substitution with diols |

These are encoded templates with explicit applicability constraints. qmdesc scores do not override a missing template or create a general reaction predictor. Free-radical and other unsupported routes can return `classified_not_generated`, which is a classification result with no released polymer graph.

### Output interpretation

The command prints a status and an exclusive run directory. Inspect `classification.json`, `monomer_sites.json`, `polymer_graph.json`, the PyG graph file, all-atom structure files, `coarse_graph.png`, `validation_report.json`, and the run manifest. Exact filenames are listed in that manifest. Accept a graph only when the validation report passes and the status is `generated`.

The graph records component identity, site ownership and state, reaction-rule identity, directed message edges, and chemical atom mappings. The qmdesc cache is written beneath the selected output directory. Original monomer Hirshfeld charges are descriptors; they are not charges calculated for the final polymer environment.

## Explicit topology constraints

The separate backend defines finite idealized topologies through a monomer catalog, node instances, named ports, and prescribed reaction events:

```bash
python -m topology_models write-builtins --output outputs/explicit_models.json
python -m topology_models validate-config --config outputs/explicit_models.json
python -m topology_models generate --config outputs/explicit_models.json \
  --output-root outputs/topology_explicit --device cpu --conformers 8 --seed 20260819
```

`write-builtins` emits ten fully specified examples. Edit a copy to change a topology, then validate it before generation. Explicit edges must use available compatible ports. Preserve parallel edges for composite reactions instead of collapsing a multigraph into a simple graph.

Expected output is a timestamped structure showcase containing topology JSON/PT, two-dimensional PNGs, site records, validation reports, and finite MOL2 conformers. Conformer generation uses the archived RDKit procedure and may invoke an existing OpenBabel executable found on `PATH`; if its required geometry step fails, inspect the error rather than treating an incomplete showcase as passed.

The showcase marks its graphs `training_eligible=false`. Structural validation does not validate CoarsePGNN predictions on rings, networks, dendrimers, bottlebrushes, or disconnected systems. MOL2 files are visualization conformers with `NO_CHARGES`, not force-field-parametrized or equilibrated MD inputs. Periodic topology is unsupported.

