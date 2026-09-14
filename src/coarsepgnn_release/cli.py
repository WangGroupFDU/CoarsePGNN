"""Documented public command line for CoarsePGNN reproducibility."""
from __future__ import annotations

import argparse
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    """Define explicit dataset, model, path and device choices without hidden defaults."""
    root = argparse.ArgumentParser(prog="coarsepgnn",description="Frozen CoarsePGNN training, inference and research assets.")
    root.add_argument("--version",action="version",version="1.0.0")
    sub = root.add_subparsers(dest="command",required=True)
    train = sub.add_parser("train",help="Train one or all five outer folds.")
    pred = sub.add_parser("predict",help="Predict independently with five models, then average in original units.")
    for p in [train,pred]:
        p.add_argument("--assets",type=Path,default=Path("assets"))
        p.add_argument("--dataset",choices=["pu","openpoly"],required=True)
        p.add_argument("--model",choices=["structure","hybrid"],required=True)
        p.add_argument("--target",choices=["tg","tm","elongation_at_break","tensile_strength_ultimate"],required=True)
        p.add_argument("--strategy",choices=["record_random_interpolation","formulation_graph_grouped"],
                       default="formulation_graph_grouped" if p is pred else "record_random_interpolation")
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--device",choices=["cpu","cuda"],default="cpu" if p is pred else "cuda")
        p.add_argument("--threads",type=int,default=4)
    train.add_argument("--fold",type=int,choices=range(5),help="Omit to train folds 0 through 4.")
    train.add_argument("--structure-models",type=Path,help="PU model root containing strategy/model/target/fold directories.")
    pred.add_argument("--input",type=Path,required=True,help="JSON Lines with unique sample_id values.")
    pred.add_argument("--batch-size",type=int,default=16)
    pred.add_argument("--condition-vectors",type=Path,help="Explicit ID-aligned NPZ from the encode command.")
    pred.add_argument("--qwen-model",type=Path,help="Local fixed-revision Qwen snapshot for new conditions.")
    enc = sub.add_parser("encode",help="Encode explicit test conditions with the frozen Hybrid encoder.")
    enc.add_argument("--assets",type=Path,default=Path("assets"))
    enc.add_argument("--input",type=Path,required=True)
    enc.add_argument("--output",type=Path,required=True)
    enc.add_argument("--qwen-model",type=Path)
    enc.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    enc.add_argument("--batch-size",type=int,default=4)
    enc_train = sub.add_parser("train-encoder",help="Rebuild the selected Hybrid head or rerun its development CV.")
    enc_train.add_argument("--assets",type=Path,default=Path("assets"))
    enc_train.add_argument("--output",type=Path,required=True)
    enc_train.add_argument("--tune",action="store_true",help="Run the archived 20 configurations x 5 folds.")
    enc_train.add_argument("--workers",type=int,default=1)
    evaluate = sub.add_parser("evaluate",help="Recompute fold and pooled OOF metrics from archived predictions.")
    evaluate.add_argument("--results",type=Path,default=Path("results"))
    evaluate.add_argument("--output",type=Path,required=True)
    evaluate.add_argument("--plots",action="store_true",help="Also write Arial 7 pt, 600 DPI PNG figures.")
    evaluate.add_argument("--arial-font",type=Path,help="Path to a locally licensed Arial TrueType font.")
    dl = sub.add_parser("download",help="Download and verify published Zenodo assets.")
    dl.add_argument("--source",choices=["zenodo","github"],default="github",help="Select the published asset host explicitly; default github.")
    dl.add_argument("--record-id",type=int)
    dl.add_argument("--download-dir",type=Path,default=Path("downloads"))
    dl.add_argument("--output",type=Path,default=Path("assets"))
    verify = sub.add_parser("verify-assets",help="Verify all extracted asset files against their SHA-256 manifest.")
    verify.add_argument("--assets",type=Path,default=Path("assets"))
    for p in [dl,verify]:
        p.add_argument("--part",choices=["all","openpoly","pu","hybrid","models"],default="all")
    prepare = sub.add_parser("prepare-openpoly",help="Rebuild Chain100 graphs from the archived structure index.")
    prepare.add_argument("--assets",type=Path,default=Path("assets"))
    prepare.add_argument("--output",type=Path,required=True)
    return root


def main() -> None:
    """Dispatch lazily so reading help does not require a configured ML environment."""
    p=parser()
    args=p.parse_args()
    for field in ["threads","batch_size","workers"]:
        if hasattr(args,field) and getattr(args,field)<=0:
            p.error(f"--{field.replace('_','-')} must be positive")
    if args.command=="train":
        from .training import train
        train(args)
    elif args.command=="predict":
        from .prediction import predict
        predict(args)
    elif args.command=="encode":
        from .conditions import encode_file
        encode_file(args)
    elif args.command=="train-encoder":
        from .conditions import train_encoder
        train_encoder(args)
    elif args.command=="evaluate":
        from .evaluation import evaluate
        evaluate(args)
    elif args.command=="download":
        from .download import download
        download(args)
    elif args.command=="verify-assets":
        from .download import verify_assets
        verify_assets(args)
    elif args.command=="prepare-openpoly":
        from openpoly_coarsepgnn_v2_harness.adapter import build_release
        from .io import schema_path,require_new
        require_new(args.output)
        build_release(args.assets/"openpoly",schema_path().parent,args.output.parent,args.output.name)
