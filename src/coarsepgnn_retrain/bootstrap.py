"""获取固定公开入口，核验原字节；不使用旧资格子集或旧监督权重。"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE_COMMIT = "5484c79cc35fe590760b2b72ef68fde184199910"
RELEASE = "https://github.com/WangGroupFDU/CoarsePGNN/releases/download/v1.0.1/CoarsePGNN_pu_v1.0.1.zip"
PINNED_FILES = {
    "CoarsePGNN_pu_v1.0.1.zip": (RELEASE, "3be3776737896ac82a3ff85a79012ac403000d61a664a5371ace98e5734614b7"),
    "tg_mean_membership_r4.csv": (
        f"https://raw.githubusercontent.com/WangGroupFDU/CoarsePGNN/{BASE_COMMIT}/research-runs/pu-inverse-20261111/round-004/evidence-00/tg_mean_membership.csv",
        "4eb5c95e122c428c014f8d40b926f48ba71831a71d609b30f02ca79f40813e4c"),
    "training_evidence_r4.json": (
        f"https://raw.githubusercontent.com/WangGroupFDU/CoarsePGNN/{BASE_COMMIT}/research-runs/pu-inverse-20261111/round-004/evidence-00/training_evidence.json",
        "e64a64857651a0dada7230dc7deb7848b57012bdb55584fe2e518070e29d91d5"),
}


def fetch_inputs(output_dir, *, local_zip=None):
    """已存在输入也必须再次核对散列；下载中断不冒充完整文件。"""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, (url, expected) in PINNED_FILES.items():
        destination = root / name
        if not destination.exists():
            temporary = root / (name + ".partial")
            if local_zip and name.endswith(".zip"):
                shutil.copyfile(local_zip, temporary)
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "CoarsePGNN-retrain/2"})
                with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"source digest mismatch: {name}")
            temporary.rename(destination)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"existing source digest mismatch: {name}")
        receipts[name] = {"url": url, "sha256": actual, "bytes": destination.stat().st_size}
    assets = root / "assets"
    with zipfile.ZipFile(root / "CoarsePGNN_pu_v1.0.1.zip") as archive:
        for info in archive.infolist():
            destination = (assets / info.filename).resolve()
            if not destination.is_relative_to(assets):
                raise ValueError("unsafe archive member path")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            original = archive.read(info)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(original).digest():
                    raise ValueError(f"extracted file differs from pinned archive: {info.filename}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(original)
    result = {"base_commit": BASE_COMMIT, "files": receipts,
              "manifest": str(assets / "pu/dataset_manifest.csv"),
              "provenance": str(assets / "pu/provenance"),
              "membership": str(root / "tg_mean_membership_r4.csv"),
              "evidence": str(root / "training_evidence_r4.json"),
              "scope": "complete published PU entry; entire private/original SQLite not included"}
    receipt_path = root / "input_receipt.json"
    if not receipt_path.exists():
        receipt_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
