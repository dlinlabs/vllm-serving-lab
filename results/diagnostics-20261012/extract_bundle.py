"""Verify and extract the byte-identical uploaded diagnostics bundle (no GPU)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True,
                    help='Empty destination directory for the original 81 files')
args = parser.parse_args()
root = Path(__file__).resolve().parent / 'raw-bundle'
manifest = json.loads((root / 'manifest.json').read_text())
chunks = []
for part in manifest['parts']:
    chunk = (root / part['path']).read_bytes()
    if len(chunk) != part['bytes'] or hashlib.sha256(chunk).hexdigest() != part['sha256']:
        raise SystemExit('Invalid archive part: ' + part['path'])
    chunks.append(chunk)
data = b''.join(chunks)
if len(data) != manifest['archive_bytes'] or hashlib.sha256(data).hexdigest() != manifest['archive_sha256']:
    raise SystemExit('Archive size or SHA-256 mismatch')
destination = args.output_dir.resolve()
destination.mkdir(parents=True, exist_ok=True)
if any(destination.iterdir()):
    raise SystemExit('Output directory must be empty')
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    if len([m for m in archive.getmembers() if m.isfile()]) != 81:
        raise SystemExit('Unexpected archive file count')
    archive.extractall(destination, filter='data')
print('Archive SHA-256 verified; 81 original files extracted to:', destination)
