"""Five-fold prediction with inverse transformation before arithmetic averaging."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json, read_samples, require_new, schema_path, sha256, write_json, write_rows

PU_MODELS = {"structure": "CoarsePGNN-structure-only", "hybrid": "CoarsePGNN-FULL"}
PU_UNITS = {"tg": "degC", "tm": "degC", "elongation_at_break": "%", "tensile_strength_ultimate": "MPa"}


def _path(value: str, input_file: Path) -> Path:
    """Resolve an input graph relative to its JSONL file, without server mappings."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else input_file.resolve().parent / path


def _condition_matrix(args: Any, samples: list[dict[str, Any]]) -> np.ndarray:
    """Load explicitly referenced vectors and encode explicitly supplied conditions."""
    from .conditions import encode_conditions
    result = np.empty((len(samples), 48), dtype=np.float32)
    cached = []
    inline = []
    for i, sample in enumerate(samples):
        if ("condition_id" in sample) == ("condition" in sample):
            raise ValueError("Hybrid input requires exactly one of condition_id or condition.")
        (cached if "condition_id" in sample else inline).append(i)
    if cached:
        path = args.condition_vectors or args.assets / "pu/condition_cache/condition_embeddings.npz"
        with np.load(path, allow_pickle=False) as payload:
            ids = [str(x) for x in payload["observation_ids"]]
            vectors = np.asarray(payload["hybrid48"], dtype=np.float32)
        if len(ids) != len(set(ids)) or vectors.shape != (len(ids), 48) or not np.isfinite(vectors).all():
            raise ValueError("Invalid condition ID/vector archive.")
        index = {value:i for i,value in enumerate(ids)}
        for i in cached:
            result[i] = vectors[index[samples[i]["condition_id"]]]
    if inline:
        result[inline] = encode_conditions([samples[i] for i in inline], args.assets,
                                           model_dir=args.qwen_model, device=args.device,
                                           batch_size=args.batch_size)
    return result


def predict(args: Any) -> None:
    """Load all five matching models and write per-fold and ensemble predictions.

    PU graph JSONs use the frozen featurizer. OpenPoly accepts an archived graph
    or a canonical PSMILES together with its hydrogen-capped repeat-unit SMILES.
    Missing folds, incompatible identities and invalid graph inputs are errors.
    """
    import torch
    from coarsepgnn import GraphFeaturizer, batch_graphs, build_model
    from openpoly_coarsepgnn_v2_harness.adapter import _chain_record, _molecular_weight, _validate_psmiles
    from pu_v24_benchmark_harness.common import TargetTransform
    from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS
    from coarsepgnn_hybrid_v3.train import _reload_model

    require_new(args.output)
    samples = read_samples(args.input)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    schema = read_json(schema_path())
    featurizer = GraphFeaturizer.from_sample_schema(schema)
    graphs = []
    for sample in samples:
        if args.dataset == "pu":
            graph = featurizer.prepare_graph(read_json(_path(sample["graph_file"], args.input))).data
        elif "graph_file" in sample:
            graph = torch.load(_path(sample["graph_file"], args.input), map_location="cpu", weights_only=False)
        else:
            _validate_psmiles(sample["canonical_psmiles"])
            capped = sample["h_capped_smiles"]
            record = _chain_record({"structure_id":sample["sample_id"],
                                    "canonical_psmiles":sample["canonical_psmiles"]}, capped, _molecular_weight(capped))
            graph = featurizer.prepare_graph(record).data
        graphs.append(graph)
    if args.dataset == "openpoly":
        if args.model != "structure" or args.target != "tg":
            raise ValueError("OpenPoly supports only --model structure --target tg.")
        folder = args.assets / "models/openpoly"
        checkpoints = [folder / f"fold_{fold}/best_model.pt" for fold in range(5)]
        unit = "K"
        condition = None
    else:
        folder = args.assets / "models/pu" / args.strategy / PU_MODELS[args.model] / args.target
        checkpoints = [folder / f"fold_{fold}/best_checkpoint.pt" for fold in range(5)]
        unit = PU_UNITS[args.target]
        condition = _condition_matrix(args,samples) if args.model == "hybrid" else None
    manifest = read_json(args.assets / "models/RELEASE_MANIFEST.json")["sha256"]
    for path in checkpoints:
        key = str(path.relative_to(args.assets / "models"))
        if sha256(path) != manifest[key]:
            raise ValueError(f"Checkpoint checksum mismatch: {key}")
    values = np.empty((len(samples), 5), dtype=np.float64)
    for fold, checkpoint in enumerate(checkpoints):
        if args.dataset == "pu":
            model, payload = _reload_model(checkpoint, schema=schema, device=device)
            identity = (payload["target"],payload["strategy"],payload["outer_fold"],payload["mode"])
            expected = (args.target,args.strategy,fold,"full" if args.model=="hybrid" else "structure_only")
            if identity != expected:
                raise ValueError(f"Checkpoint identity mismatch: {identity} != {expected}")
            inverse = TargetTransform(**payload["target_transform"]).inverse
            vocab = payload.get("assay_vocab",{})
        else:
            payload = torch.load(checkpoint,map_location="cpu",weights_only=True)
            if (payload["fold"],payload["target"],payload["mode"]) != (fold,"tg_mean_K","structure_only"):
                raise ValueError("OpenPoly checkpoint identity mismatch.")
            model = build_model(target="tg_mean_K",schema=schema,mode="structure_only").to(device)
            model.load_state_dict(payload["state_dict"],strict=True)
            stats = payload["target_stats"]
            inverse = lambda x: np.asarray(x,dtype=np.float64)*float(stats["std"])+float(stats["mean"])
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(samples),args.batch_size):
                stop = min(start+args.batch_size,len(samples))
                graph_batch = batch_graphs(graphs[start:stop]).to(device)
                if args.dataset == "openpoly":
                    output = model(graph_batch,torch.zeros((stop-start,512),device=device))
                elif args.model == "structure":
                    output = model(graph_batch)
                else:
                    assay = torch.tensor([[vocab[field].get(str(sample.get("assay",{}).get(field,"")).strip(),0)
                                           for field in ASSAY_FIELDS] for sample in samples[start:stop]],device=device)
                    output = model(graph_batch,torch.from_numpy(condition[start:stop]).to(device),assay)
                # 先逐折逆变换，再在原单位下求均值；非线性变换与平均不可交换。
                values[start:stop,fold] = inverse(output.reshape(-1).cpu().numpy())
        del model
    if not np.isfinite(values).all():
        raise ValueError("Model predictions contain nonfinite values.")
    rows = [{"sample_id":sample["sample_id"],"dataset":args.dataset,"model":args.model,
             "target":args.target,"strategy":args.strategy if args.dataset=="pu" else "structure_disjoint",
             **{f"prediction_fold_{f}":float(values[i,f]) for f in range(5)},
             "prediction_mean":float(values[i].mean()),"prediction_std":float(values[i].std(ddof=1)),"unit":unit}
            for i,sample in enumerate(samples)]
    write_rows(args.output,rows)
    write_json(args.output.with_suffix(".receipt.json"),{"sample_count":len(samples),"fold_count":5,
               "aggregation":"arithmetic mean after per-fold inverse transformation",
               "standard_deviation_ddof":1,"device":args.device,"input_sha256":sha256(args.input),
               "checkpoint_sha256":{str(p.relative_to(args.assets)):sha256(p) for p in checkpoints}})
