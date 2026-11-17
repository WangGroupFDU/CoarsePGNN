"""从已保存的专利家族检索正文编译来源映射；不联网、不查看性质数值、不拟合。

检索正文为工具返回的带行号文本，不冒称原网页 HTML。未读到目标家族正文的
专利明确保留 unresolved；引用页、相似标题和相同性质数值均不能代替家族证据。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_mapping(source_path, evidence_dir, second_scan):
    with Path(source_path).open(encoding="utf-8-sig", newline="") as stream:
        sources = list(csv.DictReader(stream))
    patents = [r for r in sources if r.get("patent_number")]
    if len(sources) != 202 or len(patents) != 56:
        raise ValueError("Unexpected bound source inventory")
    first_patents = {s["patent_number"]: s for s in patents[:28]}
    evidence_dir = Path(evidence_dir)
    inputs = [Path(source_path), Path(second_scan)]
    # 仅解析确实打开目标正文并定位 ID 字段的批次，排除搜索结果里的引用页面。
    filenames = [f"first28_batch_{i}.json" for i in range(1, 5)] + [
        "family_first28_retry.json", "family_first28_final_attempt.json"]
    found = {}
    for name in filenames:
        path = evidence_dir / name
        inputs.append(path)
        document = json.loads(path.read_text())
        text = "\n".join(item["text"] for item in document["content"] if item["type"] == "text")
        for block in re.split(r"\n-{10,}\n", text):
            header = re.search(r"https://patents\.google\.com/patent/([A-Z0-9]+)(?:/[^)\s]*)?", block)
            identity = re.search(r"(L\d+: ## ID=(\d+))", block)
            if not header or not identity or header[1] not in first_patents:
                continue
            patent = header[1]
            reference = re.search(r"【(turn\d+(?:view|search)\d+)】", block)
            previous = found.get(patent)
            if previous and previous["family_id"] != identity[2]:
                raise ValueError(f"Conflicting family identities: {patent}")
            found[patent] = {
                "source_id": first_patents[patent]["source_id"], "patent_number": patent,
                "family_id": identity[2], "status": "read", "url": header[0],
                "source_ref": reference[1] if reference else None,
                "family_field": identity[1], "evidence_file": name,
                "evidence_file_sha256": sha(path),
                "retrieved_text_sha256": hashlib.sha256(block.encode()).hexdigest(),
            }
    second = json.loads(Path(second_scan).read_text())
    for item in second["entries"]:
        if item["patent_number"] not in {s["patent_number"] for s in patents[28:]}:
            raise ValueError("Second batch is not the expected publication inventory")
        if item["status"] != "read" or not re.search(r"## ID=" + re.escape(item["family_id"]), item["family_excerpt"]):
            raise ValueError("Second batch lacks an actual family identity field")
        found[item["patent_number"]] = {**item, "evidence_file": Path(second_scan).name,
                                      "evidence_file_sha256": sha(second_scan)}
    if len(found) != 54:
        raise ValueError(f"Expected 54 read families, observed {len(found)}")
    mapping = {}
    families = defaultdict(list)
    for source in sources:
        sid, patent = source["source_id"], source.get("patent_number")
        identities = []
        if source.get("doi_norm"):
            doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", source["doi_norm"].strip(), flags=re.I).lower()
            identities.append("DOI:" + doi)
        if patent:
            identities.append(patent.upper())
        entry = {"publication_identifiers": sorted(identities), "source_family_ids": [],
                 "status": "publication_identifier_only", "evidence_ids": ["sources.csv:" + sid]}
        if patent and patent in found:
            receipt = found[patent]
            if receipt["source_id"] != sid:
                raise ValueError("Source ID differs from original metadata")
            family = "patents.google.family:" + receipt["family_id"]
            entry.update(source_family_ids=[family], status="family_verified", family_evidence=receipt)
            entry["evidence_ids"].append(receipt["evidence_file"] + ":" + patent)
            families[family].append({"source_id": sid, "patent_number": patent})
        elif patent:
            entry.update(status="family_unresolved", reason=(
                "Target patent family metadata unavailable after bounded direct-page and primary-source lookup; "
                "no family identity inferred from citing patents, matching values or titles."))
        mapping[sid] = entry
    return {
        "schema_version": "pu_source_family_mapping_v1", "source_table_sha256": sha(source_path),
        "method": "Publication identity and directly read patent Family ID for conservative split closure; no label-value grouping or deletion.",
        "retrieval_representation": "Saved structured tool responses and extracted line-numbered text; hashes are not original HTML hashes.",
        "audit_summary": {"source_count": len(sources), "patent_count": len(patents),
                          "patent_families_read": len(found), "unresolved_patents": sorted(set(first_patents) - set(found)),
                          "multiple_source_family_groups": {key: values for key, values in families.items() if len(values) > 1}},
        "compilation_inputs": [{"name": p.name, "sha256": sha(p)} for p in inputs],
        "sources": mapping,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("sources", "evidence-dir", "second-scan", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = compile_mapping(args.sources, args.evidence_dir, args.second_scan)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(target), "sha256": sha(target), **result["audit_summary"]}, ensure_ascii=False))
