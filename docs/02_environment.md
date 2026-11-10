# 2. Environment configuration

**Scope:** This chapter describes the repository software and GitHub assets at v1.0.1. For the separately packaged Zenodo v1.1.0 archive and its archived-ID interface, see [Chapter 3](03_assets.md).

## Prerequisites and recorded platform

The experiments used Python **3.11.8**, PyTorch **2.5.1 + CUDA 12.1**, PyTorch Geometric **2.6.1**, and RDKit **2025.3.2** on Linux with glibc 2.28 and an NVIDIA RTX 4090 (24 GB). Direct dependencies are pinned in `requirements.txt`. This documents an observed training environment; it is not a claim that every supported operating system was tested.

Use a Python 3.11 environment under your control. The following commands are installation instructions for the reader. Release preparation did not create an environment, install packages, or perform cross-platform installation trials.

## Linux with NVIDIA CUDA

From the repository root, with your intended Python interpreter active:

```bash
python --version
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python -m coarsepgnn_release --help
```

PyTorch includes the CUDA runtime libraries in its distribution; an appropriate system NVIDIA driver is required. Consult the [official PyTorch version-specific instructions](https://pytorch.org/get-started/previous-versions/) and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html). The release uses the standard PyG operations; it does not require separately installing every optional PyG extension.

## CPU, macOS, and Windows

For macOS use the official CPU-capable PyTorch distribution:

```bash
python -m pip install torch==2.5.1
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python -m coarsepgnn_release predict --help
```

On Windows PowerShell, the corresponding source-path setting is:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m coarsepgnn_release --help
```

Use `--device cpu` for prediction on systems without CUDA. MPS is not exposed by this frozen interface. Windows users may use WSL2 for the multi-command shell training examples. The recorded Linux system corresponds to a CentOS 8 generation userland. CentOS 7 has an older system C library and is not asserted to support the pinned binary packages; use a compatible user-controlled Linux runtime rather than changing package versions silently.

Alternatively, after installing dependencies, `python -m pip install --no-deps -e .` adds the `coarsepgnn` command. It is optional; all tutorials work with the source-path method.

## Additional requirements

New condition text requires the fixed [Qwen3-Embedding-4B snapshot](https://huggingface.co/Qwen/Qwen3-Embedding-4B) described in Chapter 5. Cached PU inference and numerical-head training do not load the base model. Four billion float32 parameters alone occupy approximately 16 GB; CPU Qwen inference also needs working memory. CUDA Qwen inference uses bfloat16, whereas its archived prompt cache was generated on CPU with batch size 4 and 16 threads.

The property workflows load one checkpoint at a time. CPU inference is supported but may be slow for large graphs. Archive extraction needs approximately 15 GB before additional training outputs; allow extra space for downloads and, if needed, Qwen weights.

Reproducing the publication-style PNGs requires a locally licensed Arial font. Pass `--arial-font /path/to/Arial.ttf` if the font is not registered. Do not substitute another font while claiming identical typography. The explicit topology backend may use an existing `obabel` executable for conformer generation; see Chapter 4. No OpenBabel executable or proprietary font is distributed.

## Reader-side verification

```bash
python -c "import torch, torch_geometric, rdkit; print(torch.__version__, torch_geometric.__version__, rdkit.__version__); print('CUDA:', torch.cuda.is_available())"
python -m coarsepgnn_release verify-assets --assets assets
```

Expected output includes the installed versions, CUDA availability, and an asset verification status of `passed`. A different platform or floating-point reduction order can produce small numerical differences; bitwise reproduction of training trajectories is not guaranteed.

