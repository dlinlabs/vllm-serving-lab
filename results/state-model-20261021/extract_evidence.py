"""Restore original file bytes from the content-addressed evidence bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) == '.':
        raise ValueError('Unsafe evidence path')
    return p


def extract(destination, bundle=None):
    bundle = Path(bundle) if bundle is not None else HERE / 'raw-bundle'
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Choose a fresh evidence directory')
    index = json.loads((bundle / 'index.json').read_text())
    expected, records = {}, index['files']
    if len({r['path'] for r in records}) != len(records):
        raise ValueError('Duplicate evidence paths')
    for r in records:
        safe_relative(r['path'])
        if not re.fullmatch('[0-9a-f]{64}', r['sha256']):
            raise ValueError('Invalid content address')
        if r['sha256'] in expected and expected[r['sha256']] != r['bytes']:
            raise ValueError('Inconsistent blob length')
        expected[r['sha256']] = r['bytes']
    if len({r['path'] for r in index['parts']}) != len(index['parts']):
        raise ValueError('Duplicate archive parts')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='evidence-', dir=destination.parent) as directory:
        temp = Path(directory); archive = temp / 'data.tar.gz'
        with archive.open('wb') as f:
            for part in index['parts']:
                p = bundle / safe_relative(part['path'])
                if p.stat().st_size != part['bytes'] or digest(p) != part['sha256']:
                    raise ValueError('Archive part checksum differs')
                with p.open('rb') as src:
                    while b := src.read(1024 * 1024):
                        f.write(b)
        if archive.stat().st_size != index['bundle_bytes'] or digest(archive) != index['bundle_sha256']:
            raise ValueError('Joined bundle differs')
        blobs = temp / 'blobs'; blobs.mkdir(); seen = set()
        with tarfile.open(archive) as t:
            for m in t:
                if not m.isfile() or not re.fullmatch(r'blobs/[0-9a-f]{64}', m.name):
                    raise ValueError('Unexpected archive member')
                h = m.name.split('/')[1]
                if h not in expected or h in seen or m.size != expected[h]:
                    raise ValueError('Duplicate or undeclared blob')
                p = blobs / h
                with t.extractfile(m) as source, p.open('wb') as target:
                    while b := source.read(1024 * 1024):
                        target.write(b)
                if digest(p) != h:
                    raise ValueError('Evidence blob differs')
                seen.add(h)
        if seen != set(expected):
            raise ValueError('Missing blob')
        staged = temp / 'restored'; staged.mkdir()
        for r in records:
            p = staged / safe_relative(r['path']); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((blobs / r['sha256']).read_bytes())
        if destination.exists():
            raise ValueError('Output appeared during verification')
        os.rename(staged, destination)
    return dict(files=len(records), unique_blobs=len(expected), all_file_bytes_verified=True,
                original_archive_sha256=index['source_archive_sha256'], transport_is_original_archive=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', required=True, type=Path)
    print(json.dumps(extract(p.parse_args().output_dir), indent=2))
