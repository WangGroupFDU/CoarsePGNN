"""Download, verify and unpack the immutable public release assets."""
from __future__ import annotations

import json
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .io import read_json, sha256


def download(args: Any) -> None:
    """Fetch selected published files and verify their release hashes before extraction.

    The command uses the selected public host API and never needs a personal token.
    A draft has no public files and therefore cannot satisfy this command.
    Existing local archives are reused only when their exact hash matches.
    """
    config = read_json(Path(__file__).with_name("assets.json"))
    if args.source == "github":
        url = "https://api.github.com/repos/WangGroupFDU/CoarsePGNN/releases/tags/v1.0.1"
        request = urllib.request.Request(url,headers={"User-Agent":"CoarsePGNN/1.0.1","Accept":"application/vnd.github+json"})
        with urllib.request.urlopen(request,timeout=60) as response:
            release=json.load(response)
        by_key={entry["name"]:{"links":{"content":entry["browser_download_url"]}} for entry in release["assets"]}
    else:
        record_id = args.record_id or config.get("zenodo_record_id")
        if not record_id:
            raise RuntimeError("The Zenodo deposit is unavailable; use --source github or see docs/03_assets.md.")
        request = urllib.request.Request(f"https://zenodo.org/api/records/{record_id}",
                                         headers={"User-Agent":"CoarsePGNN/1.0.1"})
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                record = json.load(response)
        except Exception as error:
            raise RuntimeError(f"Cannot read public Zenodo record {record_id}. A private draft is not downloadable.") from error
        entries = record.get("files",[])
        if isinstance(entries,dict):
            entries = list(entries.get("entries",{}).values())
        by_key = {entry["key"]:entry for entry in entries}
    args.download_dir.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    selected = config["archives"] if args.part=="all" else [x for x in config["archives"] if x["part"]==args.part]
    for item in selected:
        entry = by_key[item["filename"]]
        archive = args.download_dir/item["filename"]
        if not archive.exists():
            partial = archive.with_suffix(archive.suffix+".partial")
            if partial.exists():
                raise FileExistsError(f"An incomplete download exists; inspect or remove it before retrying: {partial}")
            url = entry["links"].get("content") or entry["links"].get("self")
            with urllib.request.urlopen(url,timeout=120) as response, partial.open("xb") as handle:
                shutil.copyfileobj(response,handle,length=8*1024*1024)
            if partial.stat().st_size!=item["size_bytes"] or sha256(partial)!=item["sha256"]:
                raise ValueError(f"Downloaded archive failed its size/SHA-256 check: {partial}")
            partial.rename(archive)
        if archive.stat().st_size!=item["size_bytes"] or sha256(archive)!=item["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        destination = args.output/item["part"]
        if destination.exists():
            raise FileExistsError(f"Asset directory exists; use verify-assets or choose another output: {destination}")
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if len(names) != len({name.casefold() for name in names}):
                raise ValueError("Archive filenames conflict on a case-insensitive file system.")
            for info in handle.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0]!=item["part"]:
                    raise ValueError(f"Invalid archive member: {info.filename}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Archive symlink is unsupported: {info.filename}")
            handle.extractall(args.output)
        print(f"Verified and extracted: {item['filename']}",flush=True)


def verify_assets(args: Any) -> None:
    """Check every file listed in the chosen extracted asset manifests."""
    parts = ["openpoly","pu","hybrid","models"] if args.part=="all" else [args.part]
    checked = 0
    for part in parts:
        root = args.assets/part
        manifest = read_json(root/"RELEASE_MANIFEST.json")
        for relative,expected in manifest["sha256"].items():
            path=root/relative
            if not path.is_file() or sha256(path)!=expected:
                raise ValueError(f"Missing or altered asset: {part}/{relative}")
            checked+=1
    print(json.dumps({"status":"passed","files_checked":checked,"parts":parts}))
