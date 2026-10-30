"""Original-source inventory/verdict and descriptive clock grouping regressions."""
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
from prefill_lab.contracts import save
from scripts import analyze_prefill_clock_capture as a


@pytest.mark.parametrize('fault',[None,'commit','missing','extra','hash','audit','verdict','acceptance','status'])
def test_original_source_inventory_and_verdict_cannot_be_promoted(tmp_path,monkeypatch,fault):
    capture=tmp_path/'capture';capture.mkdir();names=['a.py',*sorted(a.EXPLICIT)]
    raw={name:('content '+name).encode() for name in names}
    sources={name:hashlib.sha256(value).hexdigest() for name,value in raw.items()}
    if fault=='missing':sources.pop('a.py')
    if fault=='extra':sources['untracked.py']='a'*64
    if fault=='hash':sources['a.py']='a'*64
    save(capture/'collection-sources.json',sources)
    save(capture/'launch.json',dict(cpu_acceptance=dict(commit='other' if fault=='commit' else a.COMMIT)))
    def git(args,**kwargs):
        if args[1]=='ls-tree':return '\n'.join(names)+'\n'
        return raw[args[-1].split(':',1)[1]]
    monkeypatch.setattr(a.subprocess,'check_output',git)
    def audit(args,**kwargs):
        # Snapshot bytes exist independently of the current working tree.
        root=__import__('pathlib').Path(args[1]).parents[1]
        assert all((root/name).read_bytes()==value for name,value in raw.items())
        verdict=dict(audit_valid=fault!='verdict',candidate_passed=fault!='status',model_accepted=fault=='acceptance')
        return NS(returncode=1 if fault=='audit' else 0,stdout=json.dumps(verdict),stderr='')
    monkeypatch.setattr(a.subprocess,'run',audit)
    if fault:
        with pytest.raises(ValueError):a.original_audit(capture)
    else:
        result=a.original_audit(capture)
        assert result['matched_source_files']==len(names) and result['candidate_passed'] and not result['model_accepted']


def test_valid_rejected_pilot_keeps_rejected_verdict(tmp_path,monkeypatch):
    capture=tmp_path/'capture';capture.mkdir();sources={n:hashlib.sha256(n.encode()).hexdigest() for n in a.EXPLICIT}
    save(capture/'collection-sources.json',sources);save(capture/'launch.json',dict(cpu_acceptance=dict(commit=a.COMMIT)))
    monkeypatch.setattr(a.subprocess,'check_output',lambda args,**kw:'\n'.join(a.EXPLICIT) if args[1]=='ls-tree' else args[-1].split(':',1)[1].encode())
    monkeypatch.setattr(a.subprocess,'run',lambda *args,**kw:NS(returncode=2,stdout=json.dumps(dict(audit_valid=True,candidate_passed=False,model_accepted=False)),stderr=''))
    assert not a.original_audit(capture)['candidate_passed']


def test_queue_ranges_are_off_only_and_keep_posting_separate():
    rows=[dict(stage='held_delayed',observer='off',case='cell',clocks={'sentinel':dict(posting_lag_s=.0001,queue_residence_s=.020)}),
          dict(stage='held_delayed',observer='off',case='cell',clocks={'sentinel':dict(posting_lag_s=.0002,queue_residence_s=.030)}),
          dict(stage='held_delayed',observer='light'),dict(stage='train',observer='off')]
    assert a.sentinel_clocks(rows)==[dict(case='cell',repeats=2,max_posting_lag_s=.0002,min_queue_residence_s=.020,max_queue_residence_s=.030)]


def test_quality_reports_instrumentation_delta_without_physical_claim():
    quality=dict(passed=True,paired_observer=[dict(metric='ttft_s',delta_s=.002),dict(metric='completion_s',delta_s=.006)],
                 repeat_spread=[dict(metric='ttft_s',spread_s=.003),dict(metric='completion_s',spread_s=.008)])
    result=a.quality_summary(quality)
    assert result['ttft_s']['max_paired_delta_s']==.002
    assert result['completion_s']['max_repeat_spread_s']==.008
    assert 'not uninstrumented' in result['scope']
