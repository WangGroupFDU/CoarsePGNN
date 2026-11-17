#!/usr/bin/env python3
"""仅归档已完成运行：数据与模型分包，逐文件校验，不执行拟合或预测。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def files_under(root, prefix=""):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            # 不允许符号链接把归档范围扩展到指定输入根目录之外。
            path.resolve().relative_to(root.resolve())
            yield path, (Path(prefix) / path.relative_to(root)).as_posix()


def archive(path, entries, identity):
    entries = sorted(entries, key=lambda item: item[1])
    names = [name for _, name in entries]
    manifest_name = "MANIFEST_" + path.stem + ".json"
    if len(names) != len(set(names)) or manifest_name in names:
        raise ValueError("duplicate archive entry")
    manifest = {"identity": identity, "scope": "existing bytes only; no fitting or prediction",
                "files": [{"path": name, "bytes": source.stat().st_size, "sha256": digest(source)}
                          for source, name in entries]}
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as output:
        for source, name in entries:
            output.write(source, name)
        output.writestr(manifest_name, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # 以实际归档内的原字节逐项回读；清单自身不计入被绑定文件的分母。
    with zipfile.ZipFile(path) as output:
        if set(output.namelist()) != set(names) | {manifest_name}:
            raise ValueError("archive member set differs from its manifest")
        for item in manifest["files"]:
            value, size = hashlib.sha256(), 0
            with output.open(item["path"]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
                    size += len(block)
            if size != item["bytes"] or value.hexdigest() != item["sha256"]:
                raise ValueError(f"archive readback mismatch: {item['path']}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
            "payload_files": len(entries), "archive_members": len(entries) + 1,
            "internal_manifest": manifest_name,
            "all_members_size_and_sha256_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="包含 dataset/ 与 workflow/ 的运行父目录")
    parser.add_argument("--source-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    workflow = run / "workflow"
    result = json.loads((workflow / "workflow_result.json").read_text())
    if result.get("status") != "completed":
        raise ValueError("only a completed workflow may use this delivery packager")
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((workflow / "workflow_lock.json").read_text())
    runtime = json.loads((workflow / "runtime.json").read_text())
    identity = {"training_code_commit": runtime["base_git_head"],
                "dataset_sha256": lock["dataset_file_sha256"],
                "configuration_sha256": lock["config_file_sha256"],
                "splits_sha256": digest(workflow / "splits.json"),
                "new_blind_test_claim": False,
                "deployment_has_independent_test": False}
    data = list(files_under(run / "dataset", "dataset"))
    data += list(files_under(args.source_evidence, "source_evidence"))
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in {".json", ".xml"}:
            data.append((path, "audit/" + path.name))
    for name in ("splits.json", "workflow_lock.json", "runtime.json", "dataset_identity.json"):
        data.append((workflow / name, "workflow/" + name))
    models = [(path, name) for path, name in files_under(workflow, "workflow")
              if name != "workflow/inputs/dataset.jsonl"]
    for directory in ("independent_verification", "delivery_summary", "inference_examples"):
        if (run / directory).is_dir():
            models.extend(files_under(run / directory, directory))
    # 两包共享的冻结回执可以保留，但共同解压路径必须逐字节一致。
    data_paths, model_paths = dict((name, path) for path, name in data), dict((name, path) for path, name in models)
    shared_members = []
    for name in sorted(set(data_paths) & set(model_paths)):
        first, second = data_paths[name], model_paths[name]
        checksum = digest(first)
        if first.stat().st_size != second.stat().st_size or checksum != digest(second):
            raise ValueError(f"cross-archive shared member differs: {name}")
        shared_members.append({"path": name, "bytes": first.stat().st_size, "sha256": checksum})
    artifacts = [archive(output / "pu_retrain_data_and_audit.zip", data, identity),
                 archive(output / "pu_retrain_models_and_oof.zip", models, identity)]
    delivery = {"identity": identity, "artifacts": artifacts,
                "shared_members_identical": shared_members,
                "usage": "extract both archives into one empty directory; each MANIFEST_<archive stem>.json is retained; "
                         "use dataset/dataset.jsonl for verification, not the omitted workflow/inputs/dataset.jsonl copy; "
                         "obtain code at training_code_commit",
                "limitations": "public input only; unresolved publication families remain documented; no experimental success claim"}
    with (output / "delivery_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(delivery, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(delivery, ensure_ascii=False))


if __name__ == "__main__":
    main()
