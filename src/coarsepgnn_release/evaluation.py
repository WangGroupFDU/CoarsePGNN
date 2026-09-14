"""Recompute research metrics from held-out fold predictions, without retraining."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json, read_rows, require_new, write_json, write_rows

COUNTS = {"tg":803,"tm":954,"elongation_at_break":1032,"tensile_strength_ultimate":868}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate original-unit regression metrics; retain negative and undefined R²."""
    from scipy.stats import spearmanr
    true=np.array([float(r["y_true"]) for r in rows],dtype=np.float64)
    pred=np.array([float(r["y_pred"]) for r in rows],dtype=np.float64)
    if len(true)==0 or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric input is empty or nonfinite.")
    delta=pred-true
    ss=float(np.sum((true-true.mean())**2))
    rho=float(spearmanr(true,pred).statistic) if np.std(true)>0 and np.std(pred)>0 else None
    return {"n":len(true),"mae":float(np.abs(delta).mean()),
            "rmse":float(np.sqrt(np.mean(delta**2))),
            "r2":1-float(np.sum(delta**2))/ss if ss>0 else None,"spearman":rho}


def _figures(output: Path, groups: dict, histories: dict, arial: Path | None) -> None:
    """Export fixed-size parity and five-fold loss PNGs using the archived values."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    if arial is not None:
        font_manager.fontManager.addfont(str(arial))
    font_manager.findfont("Arial",fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial","font.size":7,"axes.labelsize":7,
                         "axes.titlesize":7,"xtick.labelsize":7,"ytick.labelsize":7,
                         "legend.fontsize":7,"axes.linewidth":0.75,
                         "xtick.major.width":0.75,"ytick.major.width":0.75,
                         "figure.facecolor":"white","savefig.facecolor":"white"})
    for key,rows in groups.items():
        dataset,strategy,model,target=key
        folder=output/"figures"/dataset/strategy/model
        folder.mkdir(parents=True,exist_ok=True)
        unit="K" if dataset=="openpoly" else {"tg":"°C","tm":"°C","elongation_at_break":"%","tensile_strength_ultimate":"MPa"}[target]
        true=np.array([float(r["y_true"]) for r in rows])
        pred=np.array([float(r["y_pred"]) for r in rows])
        fig,ax=plt.subplots(figsize=(2.3,2.3))
        ax.scatter(true,pred,s=5,alpha=0.4,edgecolors="none",color="#38789A")
        lo=min(true.min(),pred.min());hi=max(true.max(),pred.max())
        ax.plot([lo,hi],[lo,hi],color="#444444",linewidth=0.75,linestyle="--")
        m=metrics(rows)
        ax.text(0.04,0.96,f"MAE = {m['mae']:.3g} {unit}\n$R^2$ = {m['r2']:.3f}",transform=ax.transAxes,va="top")
        ax.set(xlabel=f"Observed ({unit})",ylabel=f"Predicted ({unit})")
        fig.subplots_adjust(left=0.24,bottom=0.22,right=0.96,top=0.96)
        fig.savefig(folder/f"{target}_oof_600dpi.png",dpi=600)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(4.6,2.3))
        colors=["#315B7D","#A34C39","#5A8262","#9B7B2F","#75648F"]
        for fold,history in histories[key]:
            epoch=[int(r["epoch"]) for r in history]
            ax.plot(epoch,[float(r["train_loss"]) for r in history],color=colors[fold],lw=0.75,label=f"Fold {fold+1}")
            ax.plot(epoch,[float(r["validation_loss"]) for r in history],color=colors[fold],lw=0.75,ls="--")
        ax.set(xlabel="Epoch",ylabel="Loss in transformed target space")
        ax.legend(frameon=False,ncol=5,loc="upper right")
        ax.text(0.98,0.85,"Solid: training; dashed: validation",transform=ax.transAxes,ha="right",va="top")
        fig.subplots_adjust(left=0.15,bottom=0.22,right=0.98,top=0.96)
        fig.savefig(folder/f"{target}_fivefold_loss_600dpi.png",dpi=600)
        plt.close(fig)


def evaluate(args: Any) -> None:
    """Verify 85 runs and export 17 pooled OOF summaries and reproducible figures.

    A test sample appears exactly once in its model/strategy OOF group. Fold
    metrics are also compared with the original run report before aggregation.
    """
    require_new(args.output)
    groups=defaultdict(list)
    histories=defaultdict(list)
    fold_rows=[]
    split_rows=[]
    run_count=0
    max_delta=0.0
    for path in sorted((args.results/"pu").glob("*/*/*/fold_*/predictions.csv")):
        folder=path.parent
        fold=int(folder.name.split("_")[-1]);target=folder.parent.name
        model=folder.parent.parent.name;strategy=folder.parent.parent.parent.name
        if model not in {"CoarsePGNN-structure-only","CoarsePGNN-FULL"}:
            raise ValueError(f"Out-of-scope PU model in result tree: {model}")
        key=("pu",strategy,model,target)
        rows=read_rows(path)
        saved=read_json(folder/"run_report.json")["metrics"]
        histories[key].append((fold,read_rows(folder/"history.csv")))
        ids=[r["observation_id"] for r in rows]
        if len(ids)!=COUNTS[target] or len(ids)!=len(set(ids)):
            raise ValueError(f"Invalid prediction coverage: {path}")
        for split in ["train","validation","test"]:
            selected=[r for r in rows if r["split"]==split]
            value=metrics(selected)
            for name in ["mae","rmse","r2"]:
                difference=abs(value[name]-saved[split][name])
                max_delta=max(max_delta,difference)
                if difference>1e-8:
                    raise ValueError(f"Saved metric mismatch: {path} {split} {name}")
            info={"dataset":"pu","strategy":strategy,"model":model,"target":target,
                  "fold":fold,"split":split,"error_unit":"K" if target in ["tg","tm"] else selected[0]["unit"],**value}
            split_rows.append(info)
            if split=="test":
                fold_rows.append(info)
                groups[key].extend({"sample_id":r["observation_id"],"fold":fold,
                                    "y_true":r["y_true"],"y_pred":r["y_pred"],"unit":r["unit"]} for r in selected)
        run_count+=1
    for folder in sorted((args.results/"openpoly").glob("fold_*")):
        fold=int(folder.name.split("_")[-1]);key=("openpoly","structure_disjoint","CoarsePGNN-v2-Chain100","tg")
        histories[key].append((fold,read_rows(folder/"training_history.csv")))
        saved=read_json(folder/"metrics.json")["metrics_original_scale"]
        all_ids=[]
        for split in ["train","validation","test"]:
            selected=read_rows(folder/f"predictions_{split}.csv")
            all_ids.extend(r["structure_id"] for r in selected)
            value=metrics(selected)
            for name in ["mae","rmse","r2"]:
                difference=abs(value[name]-saved[split][name]);max_delta=max(max_delta,difference)
                if difference>1e-8:raise ValueError(f"Saved OpenPoly metric mismatch: fold {fold}")
            info={"dataset":"openpoly","strategy":"structure_disjoint","model":key[2],"target":"tg",
                  "fold":fold,"split":split,"error_unit":"K",**value}
            split_rows.append(info)
            if split=="test":
                fold_rows.append(info)
                groups[key].extend({"sample_id":r["structure_id"],"fold":fold,"y_true":r["y_true"],
                                    "y_pred":r["y_pred"],"unit":"K"} for r in selected)
        if len(all_ids)!=7682 or len(set(all_ids))!=7682:
            raise ValueError(f"OpenPoly split coverage failed: {folder}")
        run_count+=1
    if run_count!=85 or len(groups)!=17:
        raise ValueError(f"Expected 85 complete runs and 17 groups; found {run_count} and {len(groups)}.")
    summaries=[];oof=[];fold_summary=[]
    for key,rows in sorted(groups.items()):
        dataset,strategy,model,target=key
        expected=7682 if dataset=="openpoly" else COUNTS[target]
        if len(rows)!=expected or len({r["sample_id"] for r in rows})!=expected or {r["fold"] for r in rows}!=set(range(5)):
            raise ValueError(f"OOF coverage failed: {key}")
        identity=dict(zip(["dataset","strategy","model","target"],key))
        summaries.append({**identity,"error_unit":"K" if target in ["tg","tm"] else rows[0]["unit"],**metrics(rows)})
        oof.extend({**identity,**r} for r in rows)
        selected=[r for r in fold_rows if tuple(r[k] for k in identity)==key]
        summary={**identity}
        for name in ["mae","rmse","r2","spearman"]:
            values=np.array([r[name] for r in selected],dtype=float)
            summary[name+"_mean"]=float(values.mean());summary[name+"_std"]=float(values.std(ddof=1))
        fold_summary.append(summary)
    args.output.mkdir(parents=True)
    write_rows(args.output/"metrics_by_split.csv",split_rows)
    write_rows(args.output/"metrics_by_fold.csv",fold_rows)
    write_rows(args.output/"metrics_oof.csv",summaries)
    write_rows(args.output/"metrics_fold_mean_std.csv",fold_summary)
    write_rows(args.output/"oof_predictions.csv",oof)
    if args.plots:_figures(args.output,groups,histories,args.arial_font)
    write_json(args.output/"verification_report.json",{"status":"passed","runs":run_count,
               "pooled_oof_groups":len(groups),"test_samples_across_groups":len(oof),
               "maximum_saved_metric_difference":max_delta,"folds_per_group":5,
               "new_training_performed":False,"figures_generated":bool(args.plots)})
