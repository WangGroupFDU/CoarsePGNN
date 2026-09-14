"""V2三任务性能图、t-SNE坐标、稳定性与中文分析报告。"""

from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .encoders import ENCODER_NAMES
from .io_utils import read_csv, read_json, read_jsonl, utc_now, write_csv, write_hash_manifest, write_json
from .v2_core import V2Record, record_from_dict_v2


TSNE_SEEDS = (20260825, 20260826, 20260827)
TSNE_DATASETS = (
    "numeric_sensitivity",
    "paraphrase_invariance",
    "semantic_role_discrimination",
)
SCHEME_LABELS = {
    "raw_text_mrl32": "Scheme 1 · Raw text",
    "numeric_tag_text_mrl32": "Scheme 2 · Numeric tags",
    "raw_plus_tag_mrl32": "Scheme 3 · Raw + tags",
    "hybrid_raw_ple_set48": "Scheme 4 · Hybrid",
}
SCHEME_SHORT = {
    "raw_text_mrl32": "S1 Raw",
    "numeric_tag_text_mrl32": "S2 Tags",
    "raw_plus_tag_mrl32": "S3 Raw+tags",
    "hybrid_raw_ple_set48": "S4 Hybrid",
}
SCHEME_COLORS = {
    "raw_text_mrl32": "#7C8DA6",
    "numeric_tag_text_mrl32": "#69A59B",
    "raw_plus_tag_mrl32": "#D5A14A",
    "hybrid_raw_ple_set48": "#9E2A2B",
}
SCHEME_MARKERS = {
    "raw_text_mrl32": "o",
    "numeric_tag_text_mrl32": "s",
    "raw_plus_tag_mrl32": "^",
    "hybrid_raw_ple_set48": "D",
}
DATASET_LABELS = {
    "numeric_sensitivity": "Numeric ordering accuracy",
    "paraphrase_invariance": "Paraphrase retrieval AUC",
    "semantic_role_discrimination": "Semantic triplet accuracy",
}
FIELD_LABELS = {
    "test_temperature_c": "Test temperature",
    "crosshead_speed_mm_min": "Crosshead speed",
    "heating_rate_c_min": "Heating rate",
    "frequency_hz": "Frequency",
    "specimen_thickness_mm": "Specimen thickness",
}


def _load_final(benchmark_dir: Path, dataset: str) -> tuple[list[V2Record], dict[str, np.ndarray]]:
    """加载一个数据集的480条final-test记录和四个表示矩阵。"""

    root = benchmark_dir / "embeddings" / dataset
    all_records = [record_from_dict_v2(row) for row in read_jsonl(root / "records.jsonl")]
    selected = [index for index, record in enumerate(all_records) if record.split == "final_test"]
    records = [all_records[index] for index in selected]
    if len(records) != 480:
        raise RuntimeError(f"TSNE_FINAL_ROW_COUNT:{dataset}:{len(records)}")
    matrices = {
        scheme: np.load(root / f"embeddings_{scheme}.npy", allow_pickle=False)[selected]
        for scheme in ENCODER_NAMES
    }
    return records, matrices


def _tsne_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """独立拟合单个数据集、方案和随机种子的t-SNE。"""

    from sklearn.manifold import TSNE, trustworthiness

    matrix = np.load(str(payload["matrix_file"]), allow_pickle=False)
    selected = np.load(str(payload["selected_file"]), allow_pickle=False)
    matrix = matrix[selected]
    seed = int(payload["seed"])
    model = TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12,
        learning_rate="auto",
        init="pca",
        max_iter=2000,
        metric="cosine",
        random_state=seed,
        method="barnes_hut",
    )
    coordinates = model.fit_transform(matrix).astype(np.float32)
    if coordinates.shape != (480, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError(f"INVALID_TSNE_COORDINATES:{payload['dataset']}:{payload['scheme']}:{seed}")
    output_file = Path(str(payload["output_file"]))
    np.save(output_file, coordinates, allow_pickle=False)
    return {
        "dataset_type": payload["dataset"],
        "scheme": payload["scheme"],
        "seed": seed,
        "row_count": len(coordinates),
        "kl_divergence": float(model.kl_divergence_),
        "cosine_trustworthiness_at_15": float(
            trustworthiness(matrix, coordinates, n_neighbors=15, metric="cosine")
        ),
        "coordinate_file": output_file.name,
    }


def _neighbor_sets(coordinates: np.ndarray, neighbors: int = 15) -> list[set[int]]:
    """返回二维坐标中每个点的近邻集合，不把自身计入。"""

    from sklearn.neighbors import NearestNeighbors

    indices = NearestNeighbors(n_neighbors=neighbors + 1).fit(coordinates).kneighbors(
        coordinates, return_distance=False
    )
    return [set(int(value) for value in row if int(value) != index) for index, row in enumerate(indices)]


def _neighbor_stability(coordinate_sets: Sequence[np.ndarray]) -> tuple[float, float]:
    """以三seed二维近邻集合的平均Jaccard衡量邻域稳定性。"""

    neighbors = [_neighbor_sets(matrix) for matrix in coordinate_sets]
    values: list[float] = []
    for left, right in itertools.combinations(range(len(neighbors)), 2):
        for point in range(len(neighbors[left])):
            union = neighbors[left][point] | neighbors[right][point]
            values.append(len(neighbors[left][point] & neighbors[right][point]) / len(union))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def run_tsne_analysis(benchmark_dir: Path, output_dir: Path, *, max_workers: int = 12) -> dict[str, Any]:
    """并行运行36个t-SNE任务，保存长表坐标、质量和三seed稳定性。"""

    work = output_dir / "tsne_work"
    work.mkdir(parents=True, exist_ok=False)
    payloads: list[dict[str, Any]] = []
    records_by_dataset: dict[str, list[V2Record]] = {}
    for dataset in TSNE_DATASETS:
        root = benchmark_dir / "embeddings" / dataset
        records = [record_from_dict_v2(row) for row in read_jsonl(root / "records.jsonl")]
        selected = np.asarray([index for index, record in enumerate(records) if record.split == "final_test"], dtype=np.int64)
        if len(selected) != 480:
            raise RuntimeError(f"TSNE_FINAL_ROW_COUNT:{dataset}:{len(selected)}")
        records_by_dataset[dataset] = [records[index] for index in selected]
        selected_file = work / f"selected_{dataset}.npy"
        np.save(selected_file, selected, allow_pickle=False)
        for scheme in ENCODER_NAMES:
            for seed in TSNE_SEEDS:
                payloads.append(
                    {
                        "dataset": dataset,
                        "scheme": scheme,
                        "seed": seed,
                        "matrix_file": root / f"embeddings_{scheme}.npy",
                        "selected_file": selected_file,
                        "output_file": work / f"tsne_{dataset}_{scheme}_{seed}.npy",
                    }
                )
    quality_rows: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(payloads), os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_tsne_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            quality_rows.append(future.result())
    quality_rows.sort(key=lambda row: (row["dataset_type"], row["scheme"], int(row["seed"])))

    stability_rows: list[dict[str, Any]] = []
    for dataset in TSNE_DATASETS:
        for scheme in ENCODER_NAMES:
            coordinate_sets = [np.load(work / f"tsne_{dataset}_{scheme}_{seed}.npy") for seed in TSNE_SEEDS]
            mean, std = _neighbor_stability(coordinate_sets)
            stability_rows.append(
                {
                    "dataset_type": dataset,
                    "scheme": scheme,
                    "neighbor_count": 15,
                    "seed_count": len(TSNE_SEEDS),
                    "mean_pairwise_neighbor_jaccard": mean,
                    "std_pairwise_neighbor_jaccard": std,
                }
            )

    for dataset in TSNE_DATASETS:
        coordinate_rows: list[dict[str, Any]] = []
        records = records_by_dataset[dataset]
        for scheme in ENCODER_NAMES:
            for seed in TSNE_SEEDS:
                matrix = np.load(work / f"tsne_{dataset}_{scheme}_{seed}.npy")
                for index, (record, xy) in enumerate(zip(records, matrix)):
                    coordinate_rows.append(
                        {
                            "dataset_type": dataset,
                            "scheme": scheme,
                            "seed": seed,
                            "point_index": index,
                            "case_id": record.case_id,
                            "family_id": record.family_id,
                            "method_family": record.method_family,
                            "variant_index": record.variant_index,
                            "variant_type": record.variant_type,
                            "changed_field": record.changed_field,
                            "template_family": record.template_family,
                            "difficulty_tier": record.difficulty_tier,
                            "value_canonical": "" if record.value_canonical is None else record.value_canonical,
                            "tsne_x": float(xy[0]),
                            "tsne_y": float(xy[1]),
                        }
                    )
        write_csv(output_dir / f"tsne_coordinates_{dataset}.csv", coordinate_rows, tuple(coordinate_rows[0]))
    write_csv(output_dir / "tsne_quality.csv", quality_rows, tuple(quality_rows[0]))
    write_csv(output_dir / "tsne_stability.csv", stability_rows, tuple(stability_rows[0]))
    return {"task_count": len(payloads), "worker_count": workers, "quality_rows": quality_rows}


def _configure_matplotlib(font_file: Path) -> tuple[Any, Any]:
    """显式注册Arial文件并设置固定论文图规范。"""

    if not font_file.is_file():
        raise FileNotFoundError(f"ARIAL_FONT_NOT_FOUND:{font_file}")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(font_file))
    family = font_manager.FontProperties(fname=str(font_file)).get_name()
    if family != "Arial":
        raise RuntimeError(f"FONT_FILE_IS_NOT_ARIAL:{family}:{font_file}")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return mpl, plt


def _save_png(fig: Any, path: Path, expected_inches: tuple[float, float]) -> dict[str, Any]:
    """按固定物理尺寸保存600 DPI PNG并核对像素和DPI。"""

    from PIL import Image

    fig.set_size_inches(*expected_inches, forward=True)
    fig.savefig(path, dpi=600, format="png", facecolor="white")
    with Image.open(path) as image:
        expected_pixels = (round(expected_inches[0] * 600), round(expected_inches[1] * 600))
        dpi = image.info.get("dpi", (0.0, 0.0))
        valid_dpi = all(abs(float(value) - 600.0) < 1.0 for value in dpi[:2])
        if image.size != expected_pixels or not valid_dpi:
            raise RuntimeError(f"PNG_GEOMETRY_INVALID:{path.name}:{image.size}:{dpi}")
        return {"file": path.name, "pixels": list(image.size), "dpi": [float(dpi[0]), float(dpi[1])]}


def _style_axis(axis: Any) -> None:
    """统一坐标轴线宽和刻度方向。"""

    for spine in axis.spines.values():
        spine.set_linewidth(0.75)
    axis.tick_params(direction="out", length=2.5, width=0.75, pad=1.5)


def _plot_scorecard(analysis_dir: Path, output_dir: Path, plt: Any) -> dict[str, Any]:
    """绘制三任务点—区间主结果图。"""

    rows = read_csv(analysis_dir / "encoder_metrics_primary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    y_positions = np.arange(len(ENCODER_NAMES))[::-1]
    for axis, dataset in zip(axes, TSNE_DATASETS):
        lookup = {row["scheme"]: row for row in rows if row["dataset_type"] == dataset}
        for y, scheme in zip(y_positions, ENCODER_NAMES):
            row = lookup[scheme]
            value = float(row["value"])
            lower, upper = float(row["ci95_lower"]), float(row["ci95_upper"])
            axis.errorbar(
                value, y, xerr=[[value - lower], [upper - value]], fmt=SCHEME_MARKERS[scheme],
                color=SCHEME_COLORS[scheme], markerfacecolor=SCHEME_COLORS[scheme],
                markeredgecolor="white" if scheme == "hybrid_raw_ple_set48" else SCHEME_COLORS[scheme],
                markeredgewidth=0.45, markersize=5.2 if scheme == "hybrid_raw_ple_set48" else 4.2,
                linewidth=1.25 if scheme == "hybrid_raw_ple_set48" else 0.85, capsize=1.8,
                zorder=5 if scheme == "hybrid_raw_ple_set48" else 3,
            )
        axis.axvline(0.5, color="#777777", linewidth=0.75, linestyle="--", zorder=1)
        axis.set_xlim(0.35, 1.02)
        axis.set_ylim(-0.7, 3.7)
        axis.set_title(DATASET_LABELS[dataset], pad=4)
        axis.set_xlabel("Final-test score")
        axis.set_yticks(y_positions)
        axis.set_yticklabels([SCHEME_SHORT[name] for name in ENCODER_NAMES] if axis is axes[0] else [])
        _style_axis(axis)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.82, wspace=0.27)
    result = _save_png(fig, output_dir / "figure_1_three_task_scorecard.png", (7, 2.3))
    plt.close(fig)
    return result


def _numeric_gap_values(records: Sequence[V2Record], matrix: np.ndarray) -> dict[str, dict[int, list[float]]]:
    """按字段和等级差收集逐family平均距离，供family Bootstrap使用。"""

    index = {record.case_id: position for position, record in enumerate(records)}
    output: dict[str, dict[int, list[float]]] = {}
    for family in sorted({record.family_id for record in records}):
        rows = sorted(
            [record for record in records if record.family_id == family],
            key=lambda record: (float(record.value_canonical), record.case_id),
        )
        field = rows[0].changed_field
        family_values: dict[int, list[float]] = {}
        for left, right in itertools.combinations(range(len(rows)), 2):
            distance = 1.0 - float(np.clip(np.dot(matrix[index[rows[left].case_id]], matrix[index[rows[right].case_id]]), -1.0, 1.0))
            family_values.setdefault(right - left, []).append(distance)
        for gap, distances in family_values.items():
            output.setdefault(field, {}).setdefault(gap, []).append(float(np.mean(distances)))
    return output


def _plot_numeric_response(benchmark_dir: Path, output_dir: Path, plt: Any) -> dict[str, Any]:
    """绘制五个数值字段随等级差变化的距离响应。"""

    records, matrices = _load_final(benchmark_dir, "numeric_sensitivity")
    values = {scheme: _numeric_gap_values(records, matrix) for scheme, matrix in matrices.items()}
    fields = [field for field in FIELD_LABELS if any(record.changed_field == field for record in records)]
    if len(fields) != 5:
        raise RuntimeError(f"NUMERIC_FIVE_FIELD_EXPECTATION:{fields}")
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.6))
    axes_flat = list(axes.ravel())
    for axis, field in zip(axes_flat, fields):
        for scheme in ENCODER_NAMES:
            gaps = sorted(values[scheme][field])
            means, lowers, uppers = [], [], []
            rng = np.random.default_rng(20260901)
            for gap in gaps:
                array = np.asarray(values[scheme][field][gap], dtype=np.float64)
                means.append(float(array.mean()))
                samples = array[rng.integers(0, len(array), size=(2000, len(array)))].mean(axis=1)
                lowers.append(float(np.quantile(samples, 0.025)))
                uppers.append(float(np.quantile(samples, 0.975)))
            color = SCHEME_COLORS[scheme]
            axis.fill_between(gaps, lowers, uppers, color=color, alpha=0.08 if scheme != "hybrid_raw_ple_set48" else 0.13, linewidth=0)
            axis.plot(
                gaps, means, color=color, marker=SCHEME_MARKERS[scheme],
                linewidth=1.35 if scheme == "hybrid_raw_ple_set48" else 0.8,
                markersize=3.4 if scheme == "hybrid_raw_ple_set48" else 2.7,
                zorder=5 if scheme == "hybrid_raw_ple_set48" else 3,
                label=SCHEME_SHORT[scheme],
            )
        axis.set_title(FIELD_LABELS[field], pad=3)
        axis.set_xlabel("Numeric level gap")
        axis.set_ylabel("Cosine distance")
        axis.set_xticks(range(1, 6))
        _style_axis(axis)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.975, 0.075), frameon=False, ncol=1)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.94, wspace=0.31, hspace=0.43)
    result = _save_png(fig, output_dir / "figure_2_numeric_response_by_field.png", (7, 4.6))
    plt.close(fig)
    return result


def _boxplot(axis: Any, groups: Sequence[Sequence[float]], *, zero_line: bool = False) -> None:
    """绘制按方案着色的轻量箱线图。"""

    box = axis.boxplot(groups, patch_artist=True, widths=0.6, showfliers=False, medianprops={"color": "white", "linewidth": 0.9})
    for patch, scheme in zip(box["boxes"], ENCODER_NAMES):
        patch.set_facecolor(SCHEME_COLORS[scheme])
        patch.set_alpha(0.95 if scheme == "hybrid_raw_ple_set48" else 0.55)
        patch.set_linewidth(1.0 if scheme == "hybrid_raw_ple_set48" else 0.65)
    for key in ("whiskers", "caps"):
        for line in box[key]:
            line.set_linewidth(0.65)
    if zero_line:
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=0.75, zorder=0)
    axis.set_xticks(range(1, 5))
    axis.set_xticklabels(["S1", "S2", "S3", "S4"])
    _style_axis(axis)


def _plot_paraphrase(analysis_dir: Path, output_dir: Path, plt: Any) -> dict[str, Any]:
    """绘制同义检索分离度、AUC和Recall@1。"""

    family = [row for row in read_csv(analysis_dir / "family_level_metrics.csv") if row["dataset_type"] == "paraphrase_invariance"]
    primary = [row for row in read_csv(analysis_dir / "encoder_metrics_primary.csv") if row["dataset_type"] == "paraphrase_invariance"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    groups = [[float(row["retrieval_separation"]) for row in family if row["scheme"] == scheme] for scheme in ENCODER_NAMES]
    _boxplot(axes[0], groups, zero_line=True)
    axes[0].set_ylabel("Nearest wrong family − within family")
    axes[0].set_title("Retrieval separation", pad=3)
    lookup = {row["scheme"]: row for row in primary}
    for position, scheme in enumerate(ENCODER_NAMES, 1):
        row = lookup[scheme]
        value, lower, upper = (float(row[key]) for key in ("value", "ci95_lower", "ci95_upper"))
        axes[1].errorbar(position, value, yerr=[[value-lower], [upper-value]], fmt=SCHEME_MARKERS[scheme], color=SCHEME_COLORS[scheme], markersize=5 if scheme.endswith("set48") else 4, capsize=2)
    axes[1].axhline(0.5, color="#666666", linestyle="--", linewidth=0.75)
    axes[1].set_xticks(range(1, 5)); axes[1].set_xticklabels(["S1", "S2", "S3", "S4"])
    axes[1].set_ylabel("Retrieval AUC"); axes[1].set_title("Family retrieval", pad=3)
    _style_axis(axes[1])
    recall = [[float(row["recall_at_1"]) for row in family if row["scheme"] == scheme] for scheme in ENCODER_NAMES]
    _boxplot(axes[2], recall)
    axes[2].set_ylabel("Tie-adjusted Recall@1"); axes[2].set_title("Nearest-neighbor retrieval", pad=3)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.85, wspace=0.37)
    result = _save_png(fig, output_dir / "figure_3_paraphrase_retrieval.png", (7, 2.3))
    plt.close(fig)
    return result


def _plot_semantic(analysis_dir: Path, output_dir: Path, plt: Any) -> dict[str, Any]:
    """按相同单位和不同单位绘制语义三元组margin分布。"""

    rows = [row for row in read_csv(analysis_dir / "family_level_metrics.csv") if row["dataset_type"] == "semantic_role_discrimination"]
    tiers = ("A_same_number_and_unit", "B_same_number_different_unit")
    titles = ("Same number and unit", "Same number, different unit")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
    for axis, tier, title in zip(axes, tiers, titles):
        groups = [[float(row["triplet_margin"]) for row in rows if row["scheme"] == scheme and row["stratum"] == tier] for scheme in ENCODER_NAMES]
        _boxplot(axis, groups, zero_line=True)
        axis.set_ylabel("Triplet margin, d(a,n) − d(a,p)")
        axis.set_title(title, pad=3)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.85, wspace=0.28)
    result = _save_png(fig, output_dir / "figure_4_semantic_triplet_margins.png", (7, 2.3))
    plt.close(fig)
    return result


def _read_main_tsne(output_dir: Path, dataset: str) -> dict[str, list[dict[str, str]]]:
    """读取第一个冻结seed的坐标并按方案分组。"""

    rows = [row for row in read_csv(output_dir / f"tsne_coordinates_{dataset}.csv") if int(row["seed"]) == TSNE_SEEDS[0]]
    grouped = {scheme: [row for row in rows if row["scheme"] == scheme] for scheme in ENCODER_NAMES}
    if any(len(values) != 480 for values in grouped.values()):
        raise RuntimeError(f"TSNE_MAIN_COORDINATE_COUNT:{dataset}")
    return grouped


def _tsne_figure(benchmark_dir: Path, output_dir: Path, dataset: str, plt: Any) -> dict[str, Any]:
    """按任务语义绘制1×4独立t-SNE模型面板。"""

    from matplotlib.lines import Line2D

    grouped = _read_main_tsne(output_dir, dataset)
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.3))
    field_colors = {
        field: color for field, color in zip(FIELD_LABELS, ("#4477AA", "#66A61E", "#CC6677", "#AA4499", "#DDCC77"))
    }
    method_markers = {"tensile": "o", "DSC": "s", "DMA": "^"}
    role_colors = {"anchor": "#355C9A", "positive_paraphrase": "#4FAE83", "hard_negative": "#C43D3D"}
    for axis, scheme in zip(axes, ENCODER_NAMES):
        rows = grouped[scheme]
        x = np.asarray([float(row["tsne_x"]) for row in rows])
        y = np.asarray([float(row["tsne_y"]) for row in rows])
        if dataset == "numeric_sensitivity":
            by_family: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                by_family.setdefault(row["family_id"], []).append(index)
            for indices in by_family.values():
                ordered = sorted(indices, key=lambda index: float(rows[index]["value_canonical"]))
                axis.plot(x[ordered], y[ordered], color="#B0B0B0", alpha=0.22, linewidth=0.45, zorder=1)
            for field, color in field_colors.items():
                indices = [index for index, row in enumerate(rows) if row["changed_field"] == field]
                ranks = np.asarray([int(rows[index]["variant_index"]) for index in indices], dtype=float)
                alpha = 0.25 + 0.7 * (ranks - ranks.min()) / max(1.0, ranks.max() - ranks.min())
                for index, opacity in zip(indices, alpha):
                    axis.scatter(x[index], y[index], s=6, color=color, alpha=float(opacity), linewidths=0, zorder=2)
        elif dataset == "paraphrase_invariance":
            # 中文注释：variant_index对应六种冻结改写模板，method_family只控制点形。
            palette = plt.get_cmap("tab10")
            template_colors = {index: palette(index % 10) for index in range(6)}
            by_family: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                by_family.setdefault(row["family_id"], []).append(index)
            for indices in by_family.values():
                centroid = np.asarray([x[indices].mean(), y[indices].mean()])
                for index in indices:
                    axis.plot((x[index], centroid[0]), (y[index], centroid[1]), color="#A8A8A8", alpha=0.15, linewidth=0.35)
            for index, row in enumerate(rows):
                style_index = int(row["variant_index"]) % 6
                axis.scatter(x[index], y[index], s=6, marker=method_markers.get(row["method_family"], "o"), color=template_colors[style_index], alpha=0.58, linewidths=0)
        else:
            by_family: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                by_family.setdefault(row["family_id"], []).append(index)
            for indices in by_family.values():
                axis.plot(x[indices], y[indices], color="#9A9A9A", alpha=0.18, linewidth=0.4, zorder=1)
            for role, color in role_colors.items():
                indices = [index for index, row in enumerate(rows) if row["variant_type"] == role]
                marker = {"anchor": "o", "positive_paraphrase": "^", "hard_negative": "X"}[role]
                axis.scatter(x[indices], y[indices], s=7, marker=marker, color=color, alpha=0.62, linewidths=0, zorder=2)
        axis.set_title(SCHEME_LABELS[scheme], pad=3, color=SCHEME_COLORS[scheme] if scheme.endswith("set48") else "black", fontweight="bold" if scheme.endswith("set48") else "normal")
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_xlabel("t-SNE 1"); axis.set_ylabel("t-SNE 2" if axis is axes[0] else "")
        _style_axis(axis)
    if dataset == "numeric_sensitivity":
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgewidth=0, markersize=4, label=FIELD_LABELS[field])
            for field, color in field_colors.items()
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=5, frameon=False, handletextpad=0.3, columnspacing=0.9)
        bottom = 0.24
    elif dataset == "paraphrase_invariance":
        style_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=plt.get_cmap("tab10")(index), markeredgewidth=0, markersize=4, label=f"Rewrite {index + 1}")
            for index in range(6)
        ]
        method_handles = [
            Line2D([0], [0], marker=marker, color="#555555", linestyle="none", markersize=4, label=method)
            for method, marker in method_markers.items()
        ]
        fig.legend(handles=style_handles + method_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=9, frameon=False, handletextpad=0.25, columnspacing=0.65)
        bottom = 0.24
    else:
        handles = [
            Line2D([0], [0], marker={"anchor": "o", "positive_paraphrase": "^", "hard_negative": "X"}[role], color="none", markerfacecolor=color, markeredgecolor=color, markersize=4, label={"anchor": "Anchor", "positive_paraphrase": "Positive", "hard_negative": "Hard negative"}[role])
            for role, color in role_colors.items()
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
        bottom = 0.24
    fig.subplots_adjust(left=0.055, right=0.99, bottom=bottom, top=0.88, wspace=0.18)
    suffix = {"numeric_sensitivity": "5a", "paraphrase_invariance": "5b", "semantic_role_discrimination": "5c"}[dataset]
    stem = {"numeric_sensitivity": "numeric_sensitivity", "paraphrase_invariance": "paraphrase_invariance", "semantic_role_discrimination": "semantic_roles"}[dataset]
    result = _save_png(fig, output_dir / f"figure_{suffix}_tsne_{stem}.png", (7, 2.3))
    plt.close(fig)
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    """把报告数值格式化为固定小数，空值保持短横线。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _write_chinese_report(benchmark_dir: Path, output_dir: Path) -> None:
    """由冻结统计表生成可复用中文结果报告。"""

    analysis = benchmark_dir / "analysis"
    primary = read_csv(analysis / "encoder_metrics_primary.csv")
    balanced = read_csv(analysis / "encoder_balanced_scores.csv")
    differences = read_csv(analysis / "pairwise_scheme4_differences.csv")
    collision = read_csv(analysis / "embedding_collision_audit.csv")
    config = read_json(benchmark_dir / "benchmark_config.json")
    release = Path(config["encoder_release"])
    tuning = read_json(release / "tuning_report.json")
    lines = [
        "# 四种测试条件编码器三任务测试报告",
        "",
        f"生成时间：{utc_now()}。所有主指标来自封存的 final-test；每套测试均为 480 条记录。",
        "主指标以 family 为统计单位，95% 区间由 2,000 次配对 family Bootstrap 得到。",
        "",
        "## 主要结果",
        "",
        "| 方案 | 数值顺序准确率 | 同义检索 AUC | 语义三元组准确率 | Balanced score | 总排名 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    balanced_lookup = {row["scheme"]: row for row in balanced}
    for scheme in ENCODER_NAMES:
        metric_lookup = {row["dataset_type"]: row for row in primary if row["scheme"] == scheme}
        lines.append(
            "| {label} | {numeric} | {para} | {semantic} | {balanced} | {rank} |".format(
                label=SCHEME_LABELS[scheme],
                numeric=_fmt(metric_lookup["numeric_sensitivity"]["value"]),
                para=_fmt(metric_lookup["paraphrase_invariance"]["value"]),
                semantic=_fmt(metric_lookup["semantic_role_discrimination"]["value"]),
                balanced=_fmt(balanced_lookup[scheme]["balanced_score"]),
                rank=balanced_lookup[scheme]["rank"],
            )
        )
    selected = tuning["selected"]
    lines.extend(
        [
            "",
            "## 方案4调优",
            "",
            (
                f"方案4通过 development 五折选择：原文分支权重 α={_fmt(selected['alpha_raw'], 3)}，"
                f"学习率={float(selected['learning_rate']):.1e}，语义三元组 margin={_fmt(selected['semantic_margin'], 2)}，"
                f"最终训练 epoch={selected['median_best_epoch']}。final-test 未参与边界、超参数或 epoch 选择。"
            ),
            "",
            "## 方案4相对差值",
            "",
            "| 任务 | 对照方案 | 主指标差值 | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in differences:
        lines.append(
            f"| {DATASET_LABELS[row['dataset_type']]} | {SCHEME_SHORT[row['comparator_scheme']]} | "
            f"{_fmt(row['difference'])} | [{_fmt(row['ci95_lower'])}, {_fmt(row['ci95_upper'])}] |"
        )
    lines.extend(
        [
            "",
            "## 表示质量与可视化口径",
            "",
            "- 精确碰撞、近碰撞和有效秩见 `embedding_collision_audit.csv`；t-SNE 不参与主指标计算。",
            "- t-SNE 对每个“数据集×方案×种子”独立拟合；主图使用种子 20260825，三种子邻域稳定性另表报告。",
            "- 方案2在同义改写任务中使用相同的金标准字段序列，因此该结果同时反映了结构化输入消除措辞差异的构造特性。",
            "- 方案4为48维，方案1–3为32维；分数比较保留这一表示容量差异。",
            "",
            "## 数据与配置追溯",
            "",
            f"- 编码器 release：`{release.name}`",
            f"- 测试 benchmark：`{benchmark_dir.name}`",
            f"- 碰撞审计行数：{len(collision)}",
            "- Qwen 分支：冻结 Qwen3-Embedding-4B、last-token pooling、MRL-32、condition-only prompt。",
        ]
    )
    (output_dir / "三任务测试与可视化报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def visualize_behavior_benchmark(
    benchmark_dir: Path,
    *,
    font_file: Path,
    max_workers: int = 12,
) -> dict[str, Any]:
    """生成36个t-SNE结果、七张600 DPI主图、中文报告和图形QC。"""

    benchmark_dir = benchmark_dir.resolve()
    analysis_dir = benchmark_dir / "analysis"
    if not (analysis_dir / "analysis_report.json").is_file():
        raise FileNotFoundError(f"ANALYSIS_REQUIRED:{analysis_dir}")
    output_dir = benchmark_dir / "visualization"
    if output_dir.exists():
        raise FileExistsError(f"VISUALIZATION_OUTPUT_EXISTS:{output_dir}")
    output_dir.mkdir()
    try:
        tsne_report = run_tsne_analysis(benchmark_dir, output_dir, max_workers=max_workers)
        _mpl, plt = _configure_matplotlib(font_file.resolve())
        images = [
            _plot_scorecard(analysis_dir, output_dir, plt),
            _plot_numeric_response(benchmark_dir, output_dir, plt),
            _plot_paraphrase(analysis_dir, output_dir, plt),
            _plot_semantic(analysis_dir, output_dir, plt),
        ]
        images.extend(_tsne_figure(benchmark_dir, output_dir, dataset, plt) for dataset in TSNE_DATASETS)
        _write_chinese_report(benchmark_dir, output_dir)
        qc = {
            "schema_version": "condition_encoder_visualization_qc_v2",
            "created_at_utc": utc_now(),
            "passed": len(images) == 7 and tsne_report["task_count"] == 36,
            "font_file": str(font_file.resolve()),
            "font_family": "Arial",
            "matplotlib_font_fallback_allowed": False,
            "image_count": len(images),
            "images": images,
            "tsne": {
                "task_count": tsne_report["task_count"],
                "worker_count": tsne_report["worker_count"],
                "rows_per_task": 480,
                "seeds": list(TSNE_SEEDS),
                "configuration": {
                    "metric": "cosine", "perplexity": 30, "early_exaggeration": 12,
                    "learning_rate": "auto", "init": "pca", "max_iter": 2000,
                },
            },
        }
        write_json(output_dir / "visualization_qc.json", qc)
        write_hash_manifest(output_dir)
        return qc
    except Exception:
        # 保留失败目录便于诊断，但显式写入失败标记，绝不将半成品当作通过结果。
        write_json(output_dir / "visualization_failed.json", {"failed_at_utc": utc_now()})
        raise
