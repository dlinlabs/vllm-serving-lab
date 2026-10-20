"""Verify and extract the original partial cap-range attempt without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(destination):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Use a fresh output directory')
    manifest = json.loads((HERE / 'raw-bundle/manifest.json').read_text())
    with tempfile.NamedTemporaryFile(suffix='.tar.gz') as joined:
        total, checksum = 0, hashlib.sha256()
        for part in manifest['parts']:
            path = HERE / 'raw-bundle' / part['path']
            if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                raise ValueError('Archive part differs: ' + part['path'])
            with path.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    joined.write(block)
                    total += len(block)
                    checksum.update(block)
        if total != manifest['archive_bytes'] or checksum.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Joined upload archive differs')
        joined.flush()
        with tarfile.open(joined.name, 'r:gz') as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive member')
            for member in members:
                if (Path(member.name).is_absolute() or '..' in Path(member.name).parts or
                        not (member.isfile() or member.isdir())):
                    raise ValueError('Unsupported archive member: ' + member.name)
            files = {member.name: member for member in members if member.isfile()}
            if set(files) != set(manifest['files']):
                raise ValueError('Archive file inventory differs')
            for name, member in files.items():
                expected = manifest['files'][name]
                with archive.extractfile(member) as stream:
                    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
                if member.size != expected['bytes'] or checksum != expected['sha256']:
                    raise ValueError('Archive member differs: ' + name)
            destination.mkdir(parents=True)
            archive.extractall(destination, filter='data')
    print(f'Verified original archive and {len(files)} files; extracted to {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    extract(parser.parse_args().output_dir)
