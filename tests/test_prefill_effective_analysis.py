"""Read-only archive/source/aggregation regression tests; no GPU qualification."""
import copy
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace as NS

import pytest
from scripts import analyze_prefill_effective_capture as a
from prefill_lab.contracts import save


def test_source_inventory_excludes_new_analysis_and_untracked_items():
    names=['prefill_lab/a.py','scripts/a.py','tests/a.py','root.py','experiments/prefill-service-v1.order.json',
           'experiments/prefill-effective-calibration-v1.json','results/report.md','scripts/nested/a.py','other/a.py']
    assert a.source_paths(names)==set(names[:6])


def packed(tmp_path):
    capture=tmp_path/'capture';capture.mkdir();save(capture/'raw.json',{'value':1})
    archive=tmp_path/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:bundle.add(capture,arcname=capture.name)
    return capture,archive


@pytest.mark.parametrize('fault',[None,'changed','extra','missing'])
def test_archive_binds_all_extracted_bytes(tmp_path,fault):
    capture,archive=packed(tmp_path)
    if fault=='changed':save(capture/'raw.json',{'value':2})
    if fault=='extra':save(capture/'extra.json',{})
    if fault=='missing':(capture/'raw.json').unlink()
    if fault:
        with pytest.raises(ValueError):a.archive_identity(archive,capture)
    else:
        meta,index=a.archive_identity(archive,capture)
        assert meta['file_count']==1 and meta['uncompressed_bytes']==(capture/'raw.json').stat().st_size
        assert set(index)=={'raw.json'}


@pytest.mark.parametrize('fault',['traversal','symlink','duplicate','root'])
def test_unsafe_archive_members_rejected(tmp_path,fault):
    capture,archive=packed(tmp_path)
    with tarfile.open(archive,'w:gz') as bundle:
        if fault=='duplicate':
            bundle.add(capture/'raw.json',arcname='capture/raw.json')
            bundle.add(capture/'raw.json',arcname='capture/raw.json')
        else:
            name={'traversal':'capture/../bad','symlink':'capture/link','root':'wrong/raw.json'}[fault]
            entry=tarfile.TarInfo(name)
            if fault=='symlink':entry.type=tarfile.SYMTYPE;entry.linkname='/etc/passwd'
            bundle.addfile(entry,io.BytesIO(b''))
    with pytest.raises(ValueError):a.archive_identity(archive,capture)


def test_endpoint_aggregation_weights_are_explicit_and_signed_bias_retained():
    cells=[]
    for metric in ('ttft_s','completion_s'):
        for case,error in [('single',.01),('multi',-.02),('multi',-.02)]:
            cells.append(dict(case=case,error_s=error,measured_median_s=1.,metric=metric,passed=True))
    result=a.endpoint_summary(cells)['ttft_s']
    assert result['request_positions']==3 and result['cases']==2
    assert result['mean_absolute_error_s']==pytest.approx(.05/3)
    assert result['case_balanced_mae_s']==pytest.approx(.015)
    assert result['signed_mean_error_s']==pytest.approx(-.01)
    assert result['max_relative_error']==.02


def test_old_source_snapshot_rejects_unreviewed_commit_before_execution(tmp_path):
    save(tmp_path/'launch.json',{'cpu_acceptance':{'commit':'a'*40}})
    with pytest.raises(ValueError,match='Unsupported'):a.snapshot_sources(tmp_path,tmp_path,tmp_path/'snapshot')


def test_source_hash_mismatch_prevents_snapshot_execution(tmp_path,monkeypatch):
    save(tmp_path/'launch.json',{'cpu_acceptance':{'commit':a.COMMIT}})
    save(tmp_path/'collection-sources.json',{'root.py':'wrong'})
    monkeypatch.setattr(a.subprocess,'check_output',lambda args,**kw:'root.py\n' if '--name-only' in args else b'print("code")')
    with pytest.raises(ValueError,match='source hash mismatch'):
        a.snapshot_sources(tmp_path,tmp_path,tmp_path/'snapshot')


def test_original_source_audit_failure_is_not_reinterpreted_as_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'snapshot_sources',lambda *args:{'a':'hash'})
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kw:NS(returncode=1,stdout='',stderr='real gate failure'))
    with pytest.raises(ValueError,match='real gate failure'):a.original_audit(tmp_path,tmp_path)


def test_unexpected_original_audit_verdict_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'snapshot_sources',lambda *args:{'a':'hash'})
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kw:NS(returncode=0,stdout=json.dumps({'audit_valid':True,'candidate_passed':False,'model_accepted':False}),stderr=''))
    with pytest.raises(ValueError,match='verdict'):a.original_audit(tmp_path,tmp_path)
