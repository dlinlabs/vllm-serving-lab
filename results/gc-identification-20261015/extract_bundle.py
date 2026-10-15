"""Verify and restore the complete original October 15 GC/identification capture."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New or empty destination for all original files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'raw-bundle'
    manifest = json.loads((root / 'manifest.json').read_text())
    chunks = []
    for part in manifest['parts']:
        name = Path(part['path'])
        if name.is_absolute() or len(name.parts) != 1:
            raise SystemExit('Invalid part path')
        chunk = (root / name).read_bytes()
        if len(chunk) != part['bytes'] or digest(chunk) != part['sha256']:
            raise SystemExit('Invalid archive part: ' + str(name))
        chunks.append(chunk)
    data = b''.join(chunks)
    if len(data) != manifest['archive_bytes'] or digest(data) != manifest['archive_sha256']:
        raise SystemExit('Archive size or SHA-256 mismatch')
    expected = {entry['path']: entry for entry in manifest['files']}
    if len(expected) != manifest['archive_regular_files']:
        raise SystemExit('Invalid original-file inventory')
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit('Output directory must be empty')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        if len(members) != manifest['archive_members']:
            raise SystemExit('Unexpected archive member count')
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise SystemExit('Unsafe archive member: ' + member.name)
        originals = [member for member in members if member.isfile()]
        if len(originals) != len(expected) or {m.name for m in originals} != expected.keys():
            raise SystemExit('Archive inventory mismatch')
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination, filter='data')
    for name, entry in expected.items():
        restored = (destination / name).read_bytes()
        if len(restored) != entry['bytes'] or digest(restored) != entry['sha256']:
            raise SystemExit('Restored file size or SHA-256 mismatch: ' + name)
    print(f"Verified archive and all {len(expected)} original files: {destination}")


if __name__ == '__main__':
    main()
