"""Five-fold OpenPoly-only OOF analysis; no external benchmark reports are required."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .train import MODEL_NAME
from coarsepgnn_release.io import read_rows, write_rows, write_json, require_new
from coarsepgnn_release.evaluation import metrics

def collect_coarse_oof(release_dir: Path, runs_root: Path) -> list[dict[str, Any]]:
    """Collect each structure's unique held-out prediction from all five runs."""
    folds=read_rows(release_dir/"folds_tg.csv")
    result=[]
    for fold in range(5):
        expected={x["structure_id"] for x in folds if int(x["fold"])==fold and x["split"]=="test"}
        rows=read_rows(runs_root/release_dir.name/MODEL_NAME/f"fold_{fold}/predictions_test.csv")
        if len(rows)!=len(expected) or {x["structure_id"] for x in rows}!=expected:
            raise ValueError(f"Test-fold coverage mismatch: {fold}")
        result.extend({**x,"fold":fold} for x in rows)
    if len(result)!=7682 or len({x["structure_id"] for x in result})!=7682:
        raise ValueError("OOF must cover exactly 7,682 unique structures.")
    return result

def analyze(release_dir: Path, runs_root: Path, report_dir: Path) -> Path:
    """Export five test-fold metrics and pooled OOF metrics to a fresh directory."""
    require_new(report_dir)
    rows=collect_coarse_oof(release_dir,runs_root)
    summaries=[{"model":MODEL_NAME,"fold":str(f),"unit":"K",**metrics([x for x in rows if x["fold"]==f])} for f in range(5)]
    summaries.append({"model":MODEL_NAME,"fold":"overall","unit":"K",**metrics(rows)})
    write_rows(report_dir/"oof_predictions.csv",rows)
    write_rows(report_dir/"metrics.csv",summaries)
    write_json(report_dir/"analysis_summary.json",{"status":"analysis_completed","folds":5,"model":MODEL_NAME,"oof_rows":len(rows)})
    (report_dir/"report.md").write_text("# OpenPoly five-fold OOF results\n\nFive independently held-out test partitions cover 7,682 canonical structures. "
        "Targets and errors use kelvin. The overall row is calculated on pooled OOF predictions, not on an ensemble of models that saw those samples.\n",encoding="utf-8")
    return report_dir

def build_parser(subparsers: Any) -> None:
    """Register the OpenPoly-only result reconstruction command."""
    p=subparsers.add_parser("analyze",help="Reconstruct five-fold and pooled OOF metrics.")
    p.add_argument("--release-dir",type=Path,required=True)
    p.add_argument("--runs-root",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.set_defaults(command="analyze")

