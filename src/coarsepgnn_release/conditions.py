"""Construct the frozen, equally weighted 48-dimensional Hybrid representation."""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .io import read_json, read_samples, require_new, write_json


def encode_conditions(samples: Sequence[dict[str, Any]], assets: Path, *,
                      model_dir: Path | None = None, device: str = "cpu",
                      batch_size: int = 4) -> np.ndarray:
    """Encode explicit text and numeric fields; load Qwen only for uncached prompts.

    Each sample provides sample_id and condition={raw_condition_text,numeric_tags}.
    Numeric tags require key, value and unit. Raw text is preserved verbatim; no
    property label, sample identifier or target value is serialized into a prompt.
    Missing prompts require an explicitly supplied local Qwen snapshot.
    """
    import torch
    from condition_encoding_benchmark.encoders import load_hybrid_checkpoint
    from condition_encoding_benchmark.qwen import PromptCache, QwenMRL32Encoder, load_prompt_cache, unique_prompts
    from condition_encoding_benchmark.v2_core import (
        FIELD_REGISTRY_V2, PLEConfigV2, PiecewiseLinearEncoderV2, V2Record,
        field_prompt_v2, make_tag_v2, raw_prompt_v2,
    )
    from pu_v24_benchmark_harness.condition_cache import _encode_hybrid

    records = []
    for sample in samples:
        data = sample["condition"]
        if not isinstance(data.get("raw_condition_text"), str):
            raise ValueError("condition.raw_condition_text must be a string.")
        tags = []
        for tag in data.get("numeric_tags", []):
            key = tag["key"]
            spec = FIELD_REGISTRY_V2[key]
            tags.append(make_tag_v2(key, tag["value"], tag["unit"],
                                    tag.get("label_text") or spec.display_name))
        if len({tag.key for tag in tags}) != len(tags):
            raise ValueError(f"Duplicate numeric fields for {sample['sample_id']}")
        records.append(V2Record(case_id=sample["sample_id"],
                                raw_condition_text=data["raw_condition_text"], numeric_tags=tuple(tags)))
    if not records:
        raise ValueError("No conditions were supplied.")
    encoder_dir = assets / "hybrid"
    cache_values = dict(load_prompt_cache(encoder_dir / "qwen_cache").vectors)
    property_cache = assets / "pu/condition_cache/qwen_cache"
    if property_cache.is_dir():
        cache_values.update(load_prompt_cache(property_cache).vectors)
    prompts = unique_prompts([prompt for record in records
                              for prompt in [raw_prompt_v2(record), *[field_prompt_v2(t) for t in record.numeric_tags]]])
    missing = [prompt for prompt in prompts if prompt.sha256 not in cache_values]
    if missing:
        if model_dir is None:
            raise ValueError(f"{len(missing)} prompts are not cached. Supply --qwen-model with the documented fixed snapshot.")
        encoder = QwenMRL32Encoder(model_dir, device_name=device)
        vectors, _counts = encoder.encode(missing, batch_size=batch_size)
        cache_values.update({prompt.sha256: vectors[i] for i, prompt in enumerate(missing)})
        del encoder
    head, _metadata = load_hybrid_checkpoint(encoder_dir / "hybrid_set_head_v2.pt", device="cpu")
    ple = PiecewiseLinearEncoderV2(PLEConfigV2.from_dict(read_json(encoder_dir / "ple_config_v2.json")))
    from condition_encoding_benchmark.v2_core import hybrid_forward_v2
    return _encode_hybrid(records, api={"FIELD_REGISTRY_V2": FIELD_REGISTRY_V2,
                                       "hybrid_forward_v2": hybrid_forward_v2},
                          cache=PromptCache(cache_values), ple=ple, head=head,
                          alpha_raw=0.5, device=device, batch_size=batch_size)


def encode_file(args: Any) -> None:
    """Write aligned IDs and Hybrid vectors for subsequent five-fold inference."""
    samples = read_samples(args.input)
    require_new(args.output)
    vectors = encode_conditions(samples, args.assets, model_dir=args.qwen_model,
                                 device=args.device, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, observation_ids=np.array([s["sample_id"] for s in samples]), hybrid48=vectors)
    write_json(args.output.with_suffix(".json"), {"rows":len(samples), "dimension":48,
                "alpha_raw":0.5, "numeric_fields":"explicit tags", "device":args.device})


def train_encoder(args: Any) -> None:
    """Reproduce selected-head training or the archived 20-configuration search.

    Copies only the required frozen input records and prompt cache into a new
    directory. The published encoder and its PLE knots are never overwritten.
    """
    from condition_encoding_benchmark.encoders import save_hybrid_checkpoint
    from condition_encoding_benchmark.v2_training import _train_final_head, tune_and_train_v2
    require_new(args.output)
    source = args.assets / "hybrid"
    args.output.mkdir(parents=True)
    for name in ["source_records.jsonl", "benchmark_records.jsonl", "development_folds.json",
                 "field_registry_v2.json", "prompt_templates_v2.json"]:
        shutil.copy2(source / name, args.output / name)
    shutil.copytree(source / "qwen_cache", args.output / "qwen_cache")
    if args.tune:
        report = tune_and_train_v2(args.output, max_workers=args.workers, max_epochs=100, patience=10)
    else:
        selected = read_json(source / "manifest.json")["selected_hybrid_config"]
        head, ple, metadata = _train_final_head(
            release_dir=args.output, alpha_raw=float(selected["alpha_raw"]),
            learning_rate=float(selected["learning_rate"]),
            semantic_margin=float(selected["semantic_margin"]),
            epochs=int(selected["median_best_epoch"]))
        save_hybrid_checkpoint(args.output / "hybrid_set_head_v2.pt", head,
                              metadata={**metadata,"cv_selected_config":selected})
        write_json(args.output / "ple_config_v2.json", ple.to_dict())
        report = {"selection":"archived development-CV selection", "configuration":selected,
                  "training_metadata":metadata}
    write_json(args.output / "reproduction_report.json", report)
