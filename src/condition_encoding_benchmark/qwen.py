"""冻结 Qwen3-Embedding-4B 的 MRL-32 推理与提示缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io_utils import sha256_file, write_csv, write_json
from .prompts import QWEN_MAX_LENGTH, QWEN_MODEL_REVISION, QWEN_MRL_DIMENSION, Prompt


@dataclass(frozen=True)
class PromptCache:
    """内存中的 prompt 哈希到 MRL-32 向量映射。"""

    vectors: Mapping[str, np.ndarray]

    def get(self, prompt: Prompt) -> np.ndarray:
        """按哈希取出向量并验证维度。"""

        if prompt.sha256 not in self.vectors:
            raise KeyError(f"PROMPT_NOT_CACHED:{prompt.sha256}")
        vector = np.asarray(self.vectors[prompt.sha256], dtype=np.float32)
        if vector.shape != (QWEN_MRL_DIMENSION,) or not np.isfinite(vector).all():
            raise RuntimeError(f"INVALID_CACHED_VECTOR:{prompt.sha256}:{vector.shape}")
        return vector


def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """取每个序列最后一个非 padding token。"""

    import torch

    if attention_mask[:, -1].sum().item() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[indices, lengths]


class QwenMRL32Encoder:
    """冻结 Qwen3-Embedding-4B，并输出重新归一化的前32维。"""

    def __init__(self, model_dir: Path, *, device_name: str = "cuda") -> None:
        """从本地固定 revision 加载模型，禁止联网和自动降级。"""

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:  # pragma: no cover - 远程环境门禁
            raise RuntimeError("MISSING_TORCH_OR_TRANSFORMERS") from error
        model_dir = model_dir.resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"QWEN_MODEL_NOT_FOUND:{model_dir}")
        if model_dir.name != QWEN_MODEL_REVISION:
            raise RuntimeError(
                f"QWEN_REVISION_MISMATCH:expected={QWEN_MODEL_REVISION}:actual={model_dir.name}"
            )
        if device_name not in {"cuda", "cpu"}:
            raise ValueError(f"UNKNOWN_DEVICE:{device_name}")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
        self.model_dir = model_dir
        self.device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, prompts: Sequence[Prompt], *, batch_size: int = 4) -> tuple[np.ndarray, list[int]]:
        """编码 prompt，拒绝超过1024 token的输入且不静默截断。"""

        import torch
        import torch.nn.functional as functional

        if not prompts:
            return np.empty((0, QWEN_MRL_DIMENSION), dtype=np.float32), []
        if batch_size <= 0:
            raise ValueError("BATCH_SIZE_MUST_BE_POSITIVE")
        outputs: list[np.ndarray] = []
        token_counts: list[int] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            tokenized = self.tokenizer(
                [item.text for item in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            counts = [int(item) for item in tokenized["attention_mask"].sum(dim=1).tolist()]
            if any(count > QWEN_MAX_LENGTH for count in counts):
                offenders = [
                    batch[index].sha256 for index, count in enumerate(counts) if count > QWEN_MAX_LENGTH
                ]
                raise RuntimeError(f"TEXT_TOO_LONG:{offenders}")
            token_counts.extend(counts)
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
            with torch.inference_mode():
                hidden = self.model(**tokenized).last_hidden_state
                pooled = _last_token_pool(hidden, tokenized["attention_mask"])
                if pooled.shape[1] < QWEN_MRL_DIMENSION:
                    raise RuntimeError(f"QWEN_OUTPUT_TOO_SMALL:{pooled.shape}")
                full = functional.normalize(pooled.float(), p=2, dim=1)
                mrl = functional.normalize(full[:, :QWEN_MRL_DIMENSION], p=2, dim=1)
            outputs.append(mrl.float().cpu().numpy())
        matrix = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if matrix.shape != (len(prompts), QWEN_MRL_DIMENSION):
            raise RuntimeError(f"QWEN_VECTOR_SHAPE:{matrix.shape}")
        if not np.isfinite(matrix).all() or not np.allclose(
            np.linalg.norm(matrix, axis=1), 1.0, atol=2e-4
        ):
            raise RuntimeError("QWEN_VECTOR_FINITE_OR_NORM_FAILURE")
        return matrix, token_counts


def unique_prompts(prompts: Iterable[Prompt]) -> list[Prompt]:
    """按哈希去重并验证同一哈希对应同一文本。"""

    by_hash: dict[str, Prompt] = {}
    for prompt in prompts:
        previous = by_hash.get(prompt.sha256)
        if previous is not None and previous.text != prompt.text:
            raise RuntimeError(f"PROMPT_HASH_COLLISION:{prompt.sha256}")
        by_hash[prompt.sha256] = prompt
    return [by_hash[key] for key in sorted(by_hash)]


def write_prompt_cache(
    output_dir: Path,
    prompts: Sequence[Prompt],
    matrix: np.ndarray,
    token_counts: Sequence[int],
    *,
    model_dir: Path,
    device: str,
    hash_model_weights: bool = True,
) -> PromptCache:
    """写入 prompt 注册表、向量矩阵和冻结推理配置。"""

    if matrix.shape != (len(prompts), QWEN_MRL_DIMENSION) or len(token_counts) != len(prompts):
        raise ValueError("PROMPT_CACHE_SHAPE_MISMATCH")
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / "qwen_mrl32_embeddings.npy"
    np.save(vector_path, matrix.astype(np.float32, copy=False), allow_pickle=False)
    rows = [
        {
            "embedding_index": index,
            "prompt_sha256": prompt.sha256,
            "prompt_kind": prompt.kind,
            "token_count": int(token_counts[index]),
            "prompt": prompt.text,
        }
        for index, prompt in enumerate(prompts)
    ]
    write_csv(
        output_dir / "qwen_prompt_registry.csv",
        rows,
        ("embedding_index", "prompt_sha256", "prompt_kind", "token_count", "prompt"),
    )
    important_names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    )
    model_hashes = {
        name: sha256_file(model_dir / name)
        for name in important_names
        if (model_dir / name).is_file()
    }
    # 中文注释：revision 目录名不足以证明本地权重未被替换；正式 release 对全部
    # safetensors 分片逐文件计算哈希，代价只在构建阶段发生一次。
    weight_paths = sorted(model_dir.glob("*.safetensors"))
    if not weight_paths:
        raise RuntimeError(f"QWEN_WEIGHT_FILES_NOT_FOUND:{model_dir}")
    weight_hashes = (
        {path.name: sha256_file(path) for path in weight_paths} if hash_model_weights else {}
    )
    config = {
        "schema_version": "qwen_mrl32_cache_v1",
        "model_path": str(model_dir.resolve()),
        "model_revision": model_dir.name,
        "model_metadata_sha256": model_hashes,
        "model_weight_sha256": weight_hashes,
        "model_weights_hashed_in_this_cache": hash_model_weights,
        "device": device,
        "dtype": "bfloat16" if device.startswith("cuda") else "float32",
        "attention_implementation": "sdpa",
        "padding_side": "left",
        "pooling": "last_valid_token",
        "full_output_normalization": "L2",
        "mrl_dimension": QWEN_MRL_DIMENSION,
        "final_normalization": "L2",
        "max_length": QWEN_MAX_LENGTH,
        "prompt_count": len(prompts),
        "token_count_min": min(token_counts) if token_counts else None,
        "token_count_max": max(token_counts) if token_counts else None,
        "vector_sha256": sha256_file(vector_path),
    }
    write_json(output_dir / "qwen_config.json", config)
    return PromptCache({prompt.sha256: matrix[index] for index, prompt in enumerate(prompts)})


def load_prompt_cache(root: Path) -> PromptCache:
    """严格读取 release 中的 prompt 注册表和向量。"""

    from .io_utils import read_csv, read_json

    rows = read_csv(root / "qwen_prompt_registry.csv")
    matrix = np.load(root / "qwen_mrl32_embeddings.npy", allow_pickle=False)
    config = read_json(root / "qwen_config.json")
    if config.get("model_revision") != QWEN_MODEL_REVISION:
        raise RuntimeError("QWEN_CACHE_REVISION_MISMATCH")
    if matrix.dtype != np.float32 or matrix.shape != (len(rows), QWEN_MRL_DIMENSION):
        raise RuntimeError(f"QWEN_CACHE_MATRIX_INVALID:{matrix.shape}:{matrix.dtype}")
    if sha256_file(root / "qwen_mrl32_embeddings.npy") != config.get("vector_sha256"):
        raise RuntimeError("QWEN_CACHE_HASH_MISMATCH")
    mapping: dict[str, np.ndarray] = {}
    for expected, row in enumerate(rows):
        index = int(row["embedding_index"])
        if index != expected:
            raise RuntimeError("QWEN_CACHE_INDEX_NOT_CONTIGUOUS")
        mapping[row["prompt_sha256"]] = matrix[index]
    return PromptCache(mapping)
