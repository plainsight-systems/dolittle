#!/usr/bin/env python3
"""Acquire declared target/AV artifacts into a revision-aware cache.

Manifest paths and digests are validated before I/O. Bounded streaming writes
exclusive temporary files, verifies size/hash, then atomically publishes. No
model code is executed; weights never enter image builds. Standalone acquisition
hashes existing files and publishes a preparation receipt. Launch reuses it
when the manifest and file fingerprints still match, without reading payloads.
This is trusted local setup state, not a fresh cryptographic content check.
Symlinked cache components are rejected. Files stay immutable while in use.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 * 1024 * 1024
RECEIPT = ".dolittle-prepared.json"
RECEIPT_LIMIT = 1024 * 1024


def manifest(path):
    return validate(json.loads(path.read_text()))


def validate(data):
    if data.get('schema') != 1 or data.get('pair') != 'qwen3-8b-nla':
        raise ValueError('Unsupported model manifest')
    roles = set()
    for model in data['models']:
        role = model['role']
        if role not in {'target', 'verbalizer', 'reconstructor'} or role in roles:
            raise ValueError('Invalid or duplicate role')
        roles.add(role)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model['repo']) or not re.fullmatch(r'[a-f0-9]{40}', model['revision']):
            raise ValueError('Invalid repository/revision')
        names = set()
        for item in model['files']:
            name = item['path']
            parts = PurePosixPath(name).parts
            if not parts or name != PurePosixPath(name).as_posix() or PurePosixPath(name).is_absolute() or '..' in parts or '\\' in name or name in names:
                raise ValueError('Invalid artifact path')
            if type(item['bytes']) is not int or item['bytes'] < 0 or not re.fullmatch(r'[a-f0-9]{64}', item['sha256']):
                raise ValueError('Invalid artifact size/digest')
            names.add(name)
        if model['index'] not in names and not (role == 'reconstructor' and model['index'] is None):
            raise ValueError('Missing declared index')
    if not {'target', 'verbalizer'} <= roles:
        raise ValueError('Both model roles are required')
    return data


def destination(root, model, item):
    path = root / model['repo'].replace('/', '--') / model['revision'] / item['path']
    current = root
    if root.is_symlink():
        raise ValueError('Cache root cannot be a symlink')
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink in artifact cache')
    return path


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest() == item['sha256']


def fingerprint(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Model artifact must be a regular file')
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino]


def snapshot(data, root):
    return {model['role'] + '/' + item['path']: fingerprint(destination(root, model, item))
            for model in data['models'] for item in model['files']}


def installation(data, files):
    """Pure versioned receipt value, binding manifest content to file metadata."""
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'schema': 1, 'manifest': digest, 'files': files}


def prepared(data, receipt, files):
    """Admit only the same manifest and unchanged complete file set."""
    return receipt == installation(data, files)


def prepare(source, root, offline=False, *, force=False):
    """Reuse setup when unchanged; otherwise fully verify/acquire and publish."""
    data = manifest(source)
    root = root.absolute()
    receipt_path = root / RECEIPT
    if not force:
        try:
            if not receipt_path.is_symlink() and receipt_path.stat().st_size <= RECEIPT_LIMIT:
                receipt = json.loads(receipt_path.read_text())
                if prepared(data, receipt, snapshot(data, root)):
                    print('Using models verified during setup.', flush=True)
                    return data
        except (OSError, ValueError):
            pass
    print('Preparing models: verifying pinned SHA-256 hashes...', flush=True)
    return acquire(source, root, offline)


def acquire(source, root, offline=False):
    """Full setup/repair. Publish a receipt only after every artifact verifies."""
    data = manifest(source)
    root = root.absolute()
    if root.is_symlink():
        raise ValueError('Cache root cannot be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / RECEIPT
    receipt_path.unlink(missing_ok=True)
    files = {}
    for model in data['models']:
        for item in model['files']:
            path = destination(root, model, item)
            key = model['role'] + '/' + item['path']
            before = fingerprint(path) if path.exists() else None
            if verified(path, item):
                after = fingerprint(path)
                if before != after:
                    raise ValueError('Artifact changed during verification')
                files[key] = after
                continue
            if offline:
                raise ValueError(f"Missing or corrupt artifact: {model['repo']}/{item['path']}")
            if shutil.disk_usage(root).free < item['bytes'] + CHUNK:
                raise ValueError('Not enough cache storage for the next artifact')
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='.download-', dir=path.parent)
            try:
                digest, size = hashlib.sha256(), 0
                url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{item['path']}"
                with os.fdopen(descriptor, 'wb') as output, urllib.request.urlopen(url, timeout=120) as response:
                    while chunk := response.read(CHUNK):
                        size += len(chunk)
                        if size > item['bytes']:
                            raise ValueError('Artifact exceeded declared size')
                        output.write(chunk)
                        digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size != item['bytes'] or digest.hexdigest() != item['sha256']:
                    raise ValueError('Artifact integrity mismatch')
                Path(temporary).replace(path)
                files[key] = fingerprint(path)
                print(f"Verified {model['repo']}/{item['path']}", flush=True)
            finally:
                Path(temporary).unlink(missing_ok=True)
    if snapshot(data, root) != files:
        raise ValueError('Artifact changed during setup')
    descriptor, temporary = tempfile.mkstemp(prefix='.prepared-', dir=root)
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(installation(data, files), output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        Path(temporary).replace(receipt_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print('Model setup complete; saved verification receipt.', flush=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'config/models.json')
    parser.add_argument('--cache', type=Path, default=ROOT / 'models')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    acquire(args.manifest, args.cache, args.offline)
