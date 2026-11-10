# Third-party sources and rights

These notices describe the repository software and GitHub assets at v1.0.1. For the published v1.1.0 dataset archive, including BCDB phase data, also read its [THIRD_PARTY_NOTICES](https://zenodo.org/records/22741277/files/CoarsePGNN_THIRD_PARTY_NOTICES_v1.1.0.md?download=1). Upstream rights and attribution remain applicable.

Original CoarsePGNN release code is distributed under MIT. Original data curation, derived graph assets, and model weights are offered under CC BY 4.0. These grants do not replace rights held by third parties.

## External software and base weights

The release imports third-party packages through their supported interfaces. Install the pinned distributions rather than treating those packages as original CoarsePGNN code.

- [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B): official base model, fixed revision `959bed3521cc1e8714392c77e15ae213b00aa921`; its model card specifies Apache-2.0. Base weights are not redistributed here.
- [qmdesc](https://github.com/yanfeiguan/qmdesc): descriptor model and bundled pretrained checkpoint; retain the upstream distribution's license and citation.
- [SMiPoly](https://github.com/PEJpOhno/SMiPoly): functional-group classification and official reaction templates; the installed 1.1.2 package supplies the templates.
- [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), and [RDKit](https://www.rdkit.org/): external scientific software with their respective licenses.
- Arial is a proprietary font and is not supplied. OpenBabel is an optional external executable, not bundled code.

## Experimental sources

OpenPoly input identity is the dated workbook SHA-256 `8a43fabdf0bad810c8379fddac025ff75cd35a486f4ccb937c54665a4e6ea525`. The original member rows and reference strings are retained. This input is larger than the earlier public OpenPoly benchmark and must be identified by its hash and this release's filtering rules.

The [OpenPoly publication](https://doi.org/10.1007/s10118-025-3402-y) and [author-maintained repository](https://github.com/WangGroupFDU/Openpoly_benchmark) provide context and citation. Individual source references in the snapshot may point to other datasets or articles, including PolyMetriX; their original provenance is preserved rather than silently reassigned.

PU `provenance/sources.csv` supplies bibliographic metadata for the selected records. Measurement values, units, formulations, and source locators are retained for scientific traceability. Article full-text PDFs, abstracts, and long evidence excerpts are not redistributed in this release. Original reference strings and bibliographic metadata have not all been independently revalidated during packaging; do not interpret a source string as a newly verified DOI.

Cite the underlying works when reusing their measurements, preserve notices, and describe the transformations applied to derived data.

