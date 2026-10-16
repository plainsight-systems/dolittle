#!/usr/bin/env python3
"""Fetch the pinned reference inputs; never execute downloaded model code.

Contract: Python 3.12+, a trusted manifest (adjacent models.json by default),
and a writable --root. --manifest selects an independent candidate bundle.
Keep verified files, stream new downloads into temporary files, and publish
only after byte count and SHA-256 match. Failed downloads remain .part files;
reruns restart those files. This is reference tooling, not the product launcher.
Optimization: stream bounded chunks and fetch at most three files concurrently.
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def fetch(root, model, item):
    dest = root / "models" / model["repo"].replace("/", "--") / item["path"]
    if dest.exists() and dest.stat().st_size == item["bytes"]:
        with dest.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == item["sha256"]:
                return f"Verified cached {model['repo']}/{item['path']}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{item['path']}"
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise ValueError(f"Download verification failed: {dest}")
    partial.replace(dest)
    return f"Downloaded and verified {model['repo']}/{item['path']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models.json"))
    args = parser.parse_args()
    source = args.manifest
    manifest = json.loads(source.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    target = args.root / "models.json"
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    jobs = [(model, item) for model in manifest["models"] for item in model["files"]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for message in pool.map(lambda job: fetch(args.root, *job), jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
