"""Portable entrypoints for the unchanged property-model training procedures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import schema_path


def train(args: Any) -> None:
    """Train selected outer folds using the archived split and target transforms.

    PU Hybrid initializes from a structure checkpoint with the same target,
    strategy, fold, seed and data-manifest hashes. Output paths are exclusive;
    an existing training run is never silently resumed or replaced.
    """
    import torch
    torch.set_num_threads(args.threads)
    folds = range(5) if args.fold is None else [args.fold]
    if args.dataset == "pu":
        from coarsepgnn_hybrid_v3.train import train_fold
        for fold in folds:
            structure = None
            if args.model == "hybrid":
                root = args.structure_models or args.assets / "models/pu"
                structure = root / args.strategy / "CoarsePGNN-structure-only" / args.target / f"fold_{fold}/best_checkpoint.pt"
            train_fold(release_dir=args.assets / "pu", schema_path=schema_path(),
                       runs_root=args.output,target=args.target,strategy=args.strategy,
                       fold=fold,mode="full" if args.model=="hybrid" else "structure_only",
                       structure_checkpoint=structure,device_name=args.device,smoke=False)
    else:
        if args.model != "structure" or args.target != "tg":
            raise ValueError("OpenPoly supports only --model structure --target tg.")
        from openpoly_coarsepgnn_v2_harness.train import train_fold
        for fold in folds:
            train_fold(release_dir=args.assets / "openpoly",coarse_root=schema_path().parent,
                       runs_root=args.output,fold=fold,seed=20260817+fold,
                       device_name=args.device,epochs=200,patience=20,
                       learning_rate=3e-4,weight_decay=1e-4,max_nodes=3200,max_edges=6400)
