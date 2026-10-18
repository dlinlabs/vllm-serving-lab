"""Verify and extract the byte-identical uploaded failed-A/shared-A/shared-B archives."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe path: {value}')
    return path


def extract(destination, selected):
    bundle = HERE / 'raw-bundle'
    index = json.loads((bundle / 'index.json').read_text())
    available = {row['id']: row for row in index['archives']}
    if len(available) != len(index['archives']) or set(selected) - set(available):
        raise ValueError('Unknown or duplicate archive declaration')
    if destination.exists():
        raise ValueError('Output already exists; choose a fresh output directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = []
    # Stage everything and publish only after every selected file is verified.
    with tempfile.TemporaryDirectory(prefix='shared-client-verify-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'captures'
        staged.mkdir()
        for name in selected:
            row = available[name]
            pieces = []
            for part in row['parts']:
                data = (bundle / relative(part['path'])).read_bytes()
                if len(data) != part['size_bytes'] or digest(data) != part['sha256']:
                    raise ValueError(f"Part failed verification: {part['path']}")
                pieces.append(data)
            archive = b''.join(pieces)
            if len(archive) != row['size_bytes'] or digest(archive) != row['sha256']:
                raise ValueError(f'Archive failed verification: {name}')
            expected = {item['path']: item for item in row['files']}
            if len(expected) != len(row['files']):
                raise ValueError(f'Duplicate file declaration in {name}')
            seen = set()
            with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as handle:
                for member in handle.getmembers():
                    path = relative(member.name)
                    if member.isdir():
                        continue
                    if not member.isfile() or member.name in seen or member.name not in expected:
                        raise ValueError(f'Unexpected archive member: {member.name}')
                    wanted = expected[member.name]
                    if member.size != wanted['size_bytes']:
                        raise ValueError(f'File size mismatch: {member.name}')
                    data = handle.extractfile(member).read()
                    if len(data) != wanted['size_bytes'] or digest(data) != wanted['sha256']:
                        raise ValueError(f'File failed verification: {member.name}')
                    output = staged / name / path
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(data)
                    seen.add(member.name)
            if seen != set(expected):
                raise ValueError(f'Missing archive members: {name}')
            results.append(dict(archive=name, files=len(seen), sha256=row['sha256']))
        if destination.exists():
            raise ValueError('Output appeared during verification; refusing to replace it')
        os.rename(staged, destination)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--archives', nargs='+', choices=('failed-a', 'shared-a', 'shared-b'),
                        default=['failed-a', 'shared-a', 'shared-b'])
    args = parser.parse_args()
    if len(set(args.archives)) != len(args.archives):
        parser.error('Choose each archive only once')
    try:
        result = extract(args.output_dir.resolve(), args.archives)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(2, f'Extraction failed: {exc}\n')
    print(json.dumps(dict(output=str(args.output_dir.resolve()), verified=result), indent=2))


if __name__ == '__main__':
    main()
