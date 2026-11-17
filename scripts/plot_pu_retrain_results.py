#!/usr/bin/env python3
"""绘制已完成的同一次 PU 训练结果；不拟合、不生成预测、不择优挑种子。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


TASKS = ("tg", "tm", "tensile_strength_ultimate", "elongation_at_break")
LABELS = {"tg": "Tg", "tm": "Tm", "tensile_strength_ultimate": "Ultimate tensile strength",
          "elongation_at_break": "Elongation at break",
          "auxiliary::tensile_strength_unspecified": "Unspecified tensile strength",
          "auxiliary::thermal_event": "Thermal event",
          "auxiliary::crystallization_temperature": "Crystallization temperature"}
UNITS = {"tg": "°C", "tm": "°C", "tensile_strength_ultimate": "MPa", "elongation_at_break": "%"}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    if value in (None, "", "None", "null"):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("plot inputs must contain finite numbers or explicit missing values")
    return result


def _style(fontfamily, fontsize, linewidth):
    """记录实际字体；Arial 缺失时不能把替代字体标成 Arial。"""
    for family in dict.fromkeys((fontfamily, "Liberation Sans", "DejaVu Sans")):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=[family]), fallback_to_default=False)
            actual = font_manager.FontProperties(fname=path).get_name()
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("no usable publication font found")
    settings = {"font.family": actual, "font.size": fontsize, "axes.titlesize": fontsize,
                "axes.labelsize": fontsize, "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
                "legend.fontsize": fontsize, "axes.linewidth": linewidth,
                "xtick.major.width": linewidth, "ytick.major.width": linewidth,
                "xtick.major.size": 3, "ytick.major.size": 3,
                "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False}
    return settings, {"requested": fontfamily, "actual": actual, "font_file": path,
                      "font_size_pt": fontsize, "axes_linewidth_pt": linewidth,
                      "fallback_used": actual.casefold() != fontfamily.casefold()}


def _finish(fig, *, output, figname):
    if output:
        prefix = Path(figname)
        if prefix.suffix.lower() in {".pdf", ".png"}:
            prefix = prefix.with_suffix("")
        paths = [prefix.with_suffix(extension) for extension in (".pdf", ".png")]
        if any(path.exists() for path in paths):
            raise FileExistsError("plot output already exists")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        # PDF 保留散点、线段和文字矢量；PNG 仅供预览。
        fig.savefig(paths[0], format="pdf", facecolor="white")
        fig.savefig(paths[1], format="png", dpi=300, facecolor="white")


def plot_data_coverage(audit, *, output=True, figszie=(7.2, 3.4), figname="data_coverage",
                       fontfamily="Arial", fontsize=7, linewidth=.75):
    """原始计数直接取自母表审计，派生父记录保留但不重复充当监督标签。"""
    style, font = _style(fontfamily, fontsize, linewidth)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=figszie, gridspec_kw={"width_ratios": [1, 1.3]})
        stages = ["Imported rows", "After atomic recovery", "Forward supervision", "Inverse supervision"]
        counts = [audit["input_rows"], audit["retained_observations"],
                  audit["shared_trainable_observations"], audit["shared_trainable_observations"]]
        axes[0].barh(range(4), counts, color=["#969696", "#56B4E9", COLORS[0], COLORS[0]], height=.6)
        axes[0].set_yticks(range(4), stages)
        axes[0].invert_yaxis()
        for index, count in enumerate(counts):
            axes[0].text(count + max(counts) * .025, index, f"{count:,}", va="center")
        axes[0].set_xlim(0, max(counts) * 1.25)
        axes[0].set_xlabel("Observation records")
        axes[0].set_title("a  Data retention", loc="left", fontweight="bold")
        order = [task for task in TASKS if task in audit["retained_by_target"]]
        order += sorted(set(audit["retained_by_target"]) - set(order))
        for index, task in enumerate(order):
            used = audit["trainable_by_target"].get(task, 0)
            retained = audit["retained_by_target"][task]
            if used > retained:
                raise ValueError("supervised count exceeds retained count")
            axes[1].barh(index, used, color=COLORS[0] if task in TASKS else COLORS[1], height=.65)
            axes[1].barh(index, retained - used, left=used, color="#D6D6D6", height=.65)
            axes[1].text(retained + max(audit["retained_by_target"].values()) * .025, index,
                         f"{used:,} / {retained:,}", va="center")
        axes[1].set_yticks(range(len(order)), [LABELS.get(task, task) for task in order])
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, max(audit["retained_by_target"].values()) * 1.48)
        axes[1].set_xlabel("Supervised / retained records")
        axes[1].set_title("b  Task coverage", loc="left", fontweight="bold")
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left=.17, right=.98, bottom=.24, top=.90, wspace=.95)
        derived = audit["retained_observations"] - audit["shared_trainable_observations"]
        fig.text(.02, .08, f"{audit['restored_atomic_observations']:,} atomic observations restored; "
                 f"{derived:,} records retained without independent supervision.", fontsize=fontsize)
        fig.text(.02, .025, "Blue: primary labels. Orange: auxiliary labels. Grey: retained summaries. "
                 "Record counts are not independent material counts.", fontsize=fontsize)
        _finish(fig, output=output, figname=figname)
    return fig, font


def plot_oof_parity(rows, metrics, *, seed, output=True, figszie=(7.2, 6.2), figname="oof_parity",
                    fontfamily="Arial", fontsize=7, linewidth=.75):
    """仅显示预声明首种子的内层选择管线；全折基线复用必须在图内说明。"""
    style, font = _style(fontfamily, fontsize, linewidth)
    selected = [row for row in rows if int(row["seed"]) == seed and row["target"] in TASKS]
    fold_families = {}
    for row in selected:
        fold, family = int(row["outer_fold"]), row["forward_family"]
        if fold in fold_families and fold_families[fold] != family:
            raise ValueError("one selected pipeline fold contains conflicting forward families")
        fold_families[fold] = family
    baseline_folds = sum(family == "baseline" for family in fold_families.values())
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=figszie)
        for letter, task, ax in zip("abcd", TASKS, axes.flat):
            task_rows = [row for row in selected if row["target"] == task]
            plotted = [row for row in task_rows if _number(row["y_pred"]) is not None]
            if not task_rows:
                raise ValueError(f"missing primary OOF task: {task}")
            values = [_number(row[key]) for row in plotted for key in ("y_true", "y_pred")]
            if any(value is None for value in values):
                raise ValueError("observed OOF labels cannot be missing")
            if values:
                lower, upper = min(values), max(values)
                pad = max(upper - lower, 1.) * .06
                ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad],
                        color="#555555", linewidth=linewidth, linestyle="--", zorder=0)
                for fold in sorted(fold_families):
                    group = [row for row in plotted if int(row["outer_fold"]) == fold]
                    ax.scatter([float(row["y_true"]) for row in group], [float(row["y_pred"]) for row in group],
                               s=9, alpha=.45, linewidths=0, color=COLORS[fold % len(COLORS)],
                               label=f"Outer fold {fold + 1}", rasterized=False)
                ax.set_xlim(lower - pad, upper + pad)
                ax.set_ylim(lower - pad, upper + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"Observed {LABELS[task]} ({UNITS[task]})")
            ax.set_ylabel(f"Predicted {LABELS[task]} ({UNITS[task]})")
            ax.set_title(f"{letter}  {LABELS[task]}", loc="left", fontweight="bold")
            metric = [item for item in metrics if item["method"] == "inner_selected_pipeline"
                      and str(item["seed"]) == str(seed) and item["task"] == task]
            if len(metric) != 1:
                raise ValueError("exactly one recorded pipeline metric required per primary task")
            mae = _number(metric[0]["mae"])
            annotation = f"n = {len(plotted):,} / {len(task_rows):,}"
            annotation += f"\nMAE = {mae:.3g} {UNITS[task]}" if mae is not None else "\nMAE unavailable"
            ax.text(.04, .96, annotation, transform=ax.transAxes, va="top")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=5, frameon=False)
        fig.suptitle(f"Inner-selected pipeline · predeclared first seed {seed}", fontsize=fontsize, y=.99)
        fig.text(.5, .038, f"Baseline used in {baseline_folds}/{len(fold_families)} outer folds; "
                 "reused baseline predictions are not independent neural-seed fits.", ha="center", fontsize=fontsize)
        fig.text(.5, .010, "Retrospective grouped out-of-fold predictions; no new blind-test claim.",
                 ha="center", fontsize=fontsize)
        fig.subplots_adjust(left=.10, right=.98, top=.93, bottom=.19, hspace=.43, wspace=.40)
        _finish(fig, output=output, figname=figname)
    return fig, {**font, "displayed_seed": seed, "baseline_outer_folds": baseline_folds,
                 "total_outer_folds": len(fold_families)}


def plot_model_mae(metrics, *, seeds, output=True, figszie=(7.2, 4.4), figname="model_mae",
                    fontfamily="Arial", fontsize=7, linewidth=.75):
    """各神经种子独立展示；经典基线只展示其一套实际运行，不复制成三次。"""
    style, font = _style(fontfamily, fontsize, linewidth)
    methods = [("train_median", "single_classical_run", "Train median"),
               ("selected_tabular", "single_classical_run", "Selected tabular")]
    methods += [("neural", str(seed), f"Neural, seed {seed}") for seed in seeds]
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=figszie)
        for letter, task, ax in zip("abcd", TASKS, axes.flat):
            for index, (method, seed, _) in enumerate(methods):
                selected = [row for row in metrics if row["method"] == method
                            and str(row["seed"]) == seed and row["task"] == task]
                if len(selected) != 1:
                    raise ValueError(f"missing or duplicated recorded comparison metric: {method}/{seed}/{task}")
                row = selected[0]
                mae, lower, upper = (_number(row[key]) for key in ("mae", "ci95_low", "ci95_high"))
                color = "#8A8A8A" if index == 0 else COLORS[(index - 1) % len(COLORS)]
                if (lower is None) != (upper is None):
                    raise ValueError("confidence interval must have two endpoints or be absent")
                if lower is not None:
                    if lower > upper:
                        raise ValueError("confidence interval endpoints are reversed")
                    ax.hlines(index, lower, upper, color=color, linewidth=linewidth)
                    ax.vlines([lower, upper], index - .08, index + .08, color=color, linewidth=linewidth)
                if mae is not None:
                    ax.plot(mae, index, "o", color=color, markersize=3.5)
                else:
                    ax.text(.02, index, "Unavailable", transform=ax.get_yaxis_transform(), va="center")
            ax.set_yticks(range(len(methods)), [item[2] for item in methods])
            ax.set_ylim(len(methods) - .5, -.5)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(f"Mean absolute error ({UNITS[task]})")
            ax.set_title(f"{letter}  {LABELS[task]}", loc="left", fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left=.22, right=.98, top=.92, bottom=.20, hspace=.55, wspace=1.0)
        fig.text(.02, .085, "Classical baselines are shown once. Whiskers show recorded 95% closure-bootstrap intervals where available.",
                 fontsize=fontsize)
        fig.text(.02, .035, "Intervals are conditional on stored OOF predictions and exclude training / model-selection uncertainty.", fontsize=fontsize)
        _finish(fig, output=output, figname=figname)
    return fig, font


def plot_inverse_scope(metrics, *, seeds, output=True, figszie=(7.2, 3.2), figname="inverse_scope",
                       fontfamily="Arial", fontsize=7, linewidth=.75):
    """分别显示目录覆盖和受限恢复；两个比例都保留全部留出观察为分母。"""
    style, font = _style(fontfamily, fontsize, linewidth)
    index = {(str(row["seed"]), row["task"]): row for row in metrics}
    if len(index) != len(metrics):
        raise ValueError("duplicate inverse metric identity")
    reference = []
    for task in TASKS:
        values = [index[str(seed), task] for seed in seeds]
        identities = {(int(row["n"]), int(row["complete_n"])) for row in values}
        if len(identities) != 1:
            raise ValueError("inverse catalogue coverage changed between seeds")
        total, complete = next(iter(identities))
        if not 0 <= complete <= total or total == 0:
            raise ValueError("invalid inverse observation denominator")
        for row in values:
            coverage = _number(row["complete_coverage"])
            recovery = _number(row["component_set_at_true_count"])
            if coverage is None or abs(coverage - complete / total) > 1e-10:
                raise ValueError("inverse coverage does not match recorded counts")
            if recovery is None or not 0 <= recovery <= 1:
                raise ValueError("invalid recorded inverse recovery rate")
        reference.append((total, complete))
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=figszie)
        labels = ["Tg", "Tm", "Ultimate tensile strength", "Elongation at break"]
        for i, (task, (total, complete)) in enumerate(zip(TASKS, reference)):
            coverage = 100 * complete / total
            axes[0].barh(i, 100, color="#E7E7E7", height=.56)
            axes[0].barh(i, coverage, color=COLORS[0], height=.56)
            axes[0].text(102, i, f"{complete:,} / {total:,}", va="center")
            for j, seed in enumerate(seeds):
                recovery = 100 * _number(index[str(seed), task]["component_set_at_true_count"])
                offset = (j - (len(seeds) - 1) / 2) * .18
                axes[1].plot(recovery, i + offset, "o", color=COLORS[j], markersize=3.6,
                             label=f"Seed {seed}" if i == 0 else None)
        for ax in axes:
            ax.set_yticks(range(len(TASKS)), labels)
            ax.set_ylim(len(TASKS) - .55, -.55)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(left=-1)
        axes[0].set_xlim(-1, 141)
        axes[0].set_xticks([0, 50, 100])
        axes[0].set_xlabel("Complete catalogue coverage (%)")
        axes[0].set_title("a  Retained evaluation denominator", loc="left", fontweight="bold")
        rates = [100 * _number(row["component_set_at_true_count"]) for row in metrics if row["task"] in TASKS]
        axes[1].set_xlim(-max(.2, max(rates) * .04), max(5, max(rates) * 1.15))
        axes[1].xaxis.set_major_locator(MaxNLocator(4))
        axes[1].set_xlabel("Exact component-set recovery (%)")
        axes[1].set_title("b  Recovery with true component count", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.25), ncol=1)
        fig.subplots_adjust(left=.19, right=.98, top=.87, bottom=.32, wspace=1.0)
        fig.text(.02, .09, "Both denominators include all held-out observations. Grey bars: incomplete training-catalogue support.")
        fig.text(.02, .025, "True component count is provided for this diagnostic; recovery is not free-generation Hit@20 or experimental success.")
        _finish(fig, output=output, figname=figname)
    return fig, {**font, "denominator": "all held-out observations", "true_component_count_used": True,
                 "experimental_success_claim": False}


def build_figures(run_dir, output_dir, *, fontfamily="Arial", fontsize=7, linewidth=.75):
    """入口只接收完成的单个运行；输出绘图数据和原件散列，不改科研结果。"""
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    workflow = run / "workflow"
    paths = {"workflow_result": workflow / "workflow_result.json", "audit": run / "dataset/audit.json",
             "summary": run / "delivery_summary/summary.json",
             "forward_metrics": run / "delivery_summary/forward_metrics.csv",
             "inverse_metrics": run / "delivery_summary/inverse_metrics.csv",
             "neural_oof": workflow / "evaluation/neural/oof_predictions.csv",
             "selected_oof": workflow / "evaluation/champion/selected_oof.csv"}
    if _json(paths["workflow_result"])["status"] != "completed":
        raise ValueError("plotting requires a fully completed workflow")
    if output.exists():
        raise FileExistsError("plot output must be a new directory")
    audit, summary = _json(paths["audit"]), _json(paths["summary"])
    if audit["forward_observation_ids"] != audit["inverse_observation_ids"]:
        raise ValueError("forward and inverse supervised identities differ")
    seeds = summary["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("predeclared seeds must be present and unique")
    expected = set(audit["forward_observation_ids"])
    selected, neural, metrics = _csv(paths["selected_oof"]), _csv(paths["neural_oof"]), _csv(paths["forward_metrics"])
    # 完整逐种子分母先核对；图只取首种子，不根据外层误差挑图。
    indices = []
    for rows in (selected, neural):
        index = {}
        for row in rows:
            key = (int(row["seed"]), row["observation_id"])
            if key in index:
                raise ValueError("duplicated OOF observation within a seed")
            index[key] = row
        if set(index) != {(int(seed), oid) for seed in seeds for oid in expected}:
            raise ValueError("OOF does not cover every supervised observation once per declared seed")
        indices.append(index)
    for key, row in indices[0].items():
        reference = indices[1][key]
        if (row["target"], row["outer_fold"], _number(row["y_true"])) != (
                reference["target"], reference["outer_fold"], _number(reference["y_true"])):
            raise ValueError("selected and neural OOF label/fold identity mismatch")
    output.mkdir(parents=True)
    coverage = [{"task": task, "retained_n": count, "supervised_n": audit["trainable_by_target"].get(task, 0)}
                for task, count in audit["retained_by_target"].items()]
    parity = [{key: row[key] for key in ("observation_id", "seed", "outer_fold", "target", "y_true", "y_pred", "forward_family")}
              for row in selected if int(row["seed"]) == int(seeds[0]) and row["target"] in TASKS]
    for row in parity:
        row["plot_status"] = "plotted" if _number(row["y_pred"]) is not None else "unavailable_prediction"
    comparison = [row for row in metrics if row["task"] in TASKS
                  and row["method"] in ("train_median", "selected_tabular", "neural")]
    inverse = [row for row in _csv(paths["inverse_metrics"]) if row["task"] in TASKS]
    if {(str(row["seed"]), row["task"]) for row in inverse} != {(str(seed), task) for seed in seeds for task in TASKS}:
        raise ValueError("inverse metrics do not cover the predeclared seeds and primary tasks")
    _write_csv(output / "data_coverage.csv", coverage)
    _write_csv(output / "oof_parity_data.csv", parity)
    _write_csv(output / "model_mae_data.csv", comparison)
    _write_csv(output / "inverse_scope_data.csv", inverse)
    options = {"fontfamily": fontfamily, "fontsize": fontsize, "linewidth": linewidth}
    records = {}
    for name, function, args, extra in (
        ("data_coverage", plot_data_coverage, (audit,), {}),
        ("oof_parity", plot_oof_parity, (selected, metrics), {"seed": int(seeds[0])}),
        ("model_mae", plot_model_mae, (metrics,), {"seeds": seeds}),
        ("inverse_scope", plot_inverse_scope, (inverse,), {"seeds": seeds}),
    ):
        fig, record = function(*args, **extra, **options, figname=output / name)
        plt.close(fig)
        records[name] = record
    manifest = {"schema_version": "pu-retrain-plots-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "run": str(run), "sources": {key: {"path": str(path.relative_to(run)), "sha256": _sha(path)} for key, path in paths.items()},
                "script_sha256": _sha(__file__), "figures": records, "displayed_seed_rule": "first predeclared seed, never best OOF seed",
                "statistical_scope": "existing retrospective grouped OOF metrics; no new predictions, fitting, bootstrap or model selection",
                "supervised_observations_per_seed": len(expected),
                "unresolved_publication_families": audit.get("publication_identity_audit", {}).get("unresolved_family_sources", []),
                "outputs": {path.name: _sha(path) for path in sorted(output.iterdir())}}
    with (output / "plot_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font", default="Arial")
    parser.add_argument("--fontsize", type=float, default=7)
    parser.add_argument("--linewidth", type=float, default=.75)
    args = parser.parse_args()
    result = build_figures(args.run, args.output, fontfamily=args.font, fontsize=args.fontsize, linewidth=args.linewidth)
    print(json.dumps({"status": "completed", "figures": result["figures"]}, ensure_ascii=False))
