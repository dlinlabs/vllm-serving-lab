import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import pytest

PATH = Path(__file__).resolve().parents[1] / 'results/state-model-20261021/extract_evidence.py'
spec = importlib.util.spec_from_file_location('evidence', PATH)
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)


def bundle(tmp_path, *, duplicate=False, member=None, output_path='one/data'):
    b = tmp_path / 'bundle'; b.mkdir(); content = b'original bytes'; h = hashlib.sha256(content).hexdigest()
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode='w:gz') as t:
        for _ in range(2 if duplicate else 1):
            m = tarfile.TarInfo(member or ('blobs/' + h)); m.size = len(content); t.addfile(m, io.BytesIO(content))
    data = blob.getvalue(); (b / 'part').write_bytes(data)
    index = dict(files=[dict(path=output_path, sha256=h, bytes=len(content)),
                        dict(path='two/data', sha256=h, bytes=len(content))],
        parts=[dict(path='part', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())],
        bundle_bytes=len(data), bundle_sha256=hashlib.sha256(data).hexdigest(), source_archive_sha256='source')
    (b / 'index.json').write_text(json.dumps(index)); return b


def test_restoration_keeps_duplicate_content_independent(tmp_path):
    b = bundle(tmp_path); out = tmp_path / 'out'; result = ev.extract(out, b)
    assert result['files'] == 2 and result['unique_blobs'] == 1
    (out / 'one/data').write_text('changed')
    assert (out / 'two/data').read_bytes() == b'original bytes'
    with pytest.raises(ValueError, match='fresh'):
        ev.extract(out, b)


@pytest.mark.parametrize('kwargs,match', [({'duplicate': True}, 'Duplicate'),
    ({'member': '../escape'}, 'Unexpected'), ({'output_path': '../escape'}, 'Unsafe'),
    ({'output_path': '/escape'}, 'Unsafe')])
def test_malformed_bundle_never_publishes(tmp_path, kwargs, match):
    b = bundle(tmp_path, **kwargs); out = tmp_path / 'out'
    with pytest.raises(ValueError, match=match):
        ev.extract(out, b)
    assert not out.exists()


def test_corrupt_part_never_publishes(tmp_path):
    b = bundle(tmp_path); (b / 'part').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        ev.extract(tmp_path / 'out', b)
    assert not (tmp_path / 'out').exists()
