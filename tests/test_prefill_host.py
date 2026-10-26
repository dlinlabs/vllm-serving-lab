"""Pin/resume and acceptance boundaries for short rental-host commands."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from scripts import prefill_host as host
from prefill_lab.contracts import save


def mock_host(monkeypatch,tmp_path):
    state=dict(revisions=[],validations=0,trace_checked=False)
    monkeypatch.setattr(host.subprocess,'check_output',lambda args,**kw:'' if 'diff' in args else 'a'*40)
    monkeypatch.setattr(host,'code_identity',lambda:{'source':'hash'})
    def download(**kw):state['revisions'].append(kw['revision']);return str(tmp_path/'snapshot')
    monkeypatch.setitem(sys.modules,'huggingface_hub',NS(HfApi=lambda:NS(model_info=lambda model:NS(sha='b'*40)),snapshot_download=download))
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=NS(from_pretrained=lambda *a,**kw:object())))
    monkeypatch.setattr(host,'load_trace',lambda *a:{'requests':[1,2]})
    monkeypatch.setattr(host,'verify_prompts',lambda *a:state.update(trace_checked=True))
    monkeypatch.setattr(host,'prepare',lambda a:save(a.output,{'trials':{'trial':[]}}))
    def cpu(path):
        assert state['trace_checked'];state['validations']+=1;path.mkdir(parents=True)
        save(path/'acceptance.json',dict(valid=True,commit='a'*40,source_sha256={'source':'hash'}))
        return 0
    monkeypatch.setattr(host,'cpu_check',cpu)
    return state


def test_setup_resumes_same_pin_and_reuses_only_matching_acceptance(monkeypatch,tmp_path):
    state=mock_host(monkeypatch,tmp_path);out=tmp_path/'inputs'
    first=host.prepare_host(out);second=host.prepare_host(out)
    assert first==second and state['validations']==1 and state['revisions']==['b'*40]*2
    monkeypatch.setattr(host,'code_identity',lambda:{'source':'changed'})
    # Previously valid evidence cannot survive a source change or be relabeled.
    assert not host.accepted(Path(first['cpu_acceptance']),'a'*40)
    monkeypatch.setitem(sys.modules,'huggingface_hub',NS(HfApi=lambda:pytest.fail('must retain pin'),snapshot_download=lambda **kw:(_ for _ in ()).throw(ConnectionError('interrupted'))))
    with pytest.raises(ConnectionError):host.prepare_host(out)
    assert json.loads((out/'model.json').read_text())['revision']=='b'*40


def test_setup_failure_preserves_pin_and_cannot_create_ready_evidence(monkeypatch,tmp_path):
    mock_host(monkeypatch,tmp_path);out=tmp_path/'inputs'
    monkeypatch.setattr(host,'cpu_check',lambda *a:1)
    with pytest.raises(RuntimeError,match='CPU acceptance failed'):host.prepare_host(out)
    assert (out/'model.json').exists() and not (out/'ready.json').exists()
    save(out/'model.json',dict(model=host.MODEL,revision='main'))
    with pytest.raises(ValueError,match='Invalid existing model pin'):host.prepare_host(out)


def test_collect_requires_exact_acceptance_and_delegates_frozen_runner(monkeypatch,tmp_path):
    mock_host(monkeypatch,tmp_path);ready=host.prepare_host(tmp_path/'inputs');calls=[]
    monkeypatch.setattr(host.subprocess,'run',lambda args,**kw:calls.append((args,kw)))
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path))
    host.collect_host(tmp_path/'inputs')
    args,kw=calls[0]
    assert args[:3]==[sys.executable,str(host.ROOT/'scripts/run_prefill_service.py'),'collect']
    assert args[args.index('--cpu-acceptance')+1]==ready['cpu_acceptance'] and kw['check']
    monkeypatch.setattr(host,'commit',lambda:'c'*40)
    with pytest.raises(RuntimeError,match='exact checkout'):host.collect_host(tmp_path/'inputs')
    assert len(calls)==1


def test_engine_descendant_finds_ninja_without_shell_activation(monkeypatch,tmp_path):
    import os
    import subprocess
    from scripts import run_prefill_service as runner
    real_python=sys.executable
    venv=tmp_path/'venv';bindir=venv/'bin';bindir.mkdir(parents=True)
    ninja=bindir/'ninja';ninja.write_text('#!/bin/sh\nprintf "venv-ninja\\n"\n');ninja.chmod(0o755)
    monkeypatch.setenv('PATH','/usr/bin:/bin');monkeypatch.setenv('PYTHONHOME','/invalid')
    monkeypatch.setenv('VIRTUAL_ENV','/other')
    monkeypatch.setattr(runner.sys,'executable',str(bindir/'python'))
    monkeypatch.setattr(runner.sys,'prefix',str(venv))
    env=runner.runtime_environment()
    child=subprocess.run([real_python,'-c','import subprocess; subprocess.run(["ninja", "--version"],check=True)'],
                         env=env,text=True,capture_output=True,check=True)
    assert child.stdout.strip()=='venv-ninja'
    assert env['VIRTUAL_ENV']==str(venv) and 'PYTHONHOME' not in env
    assert os.environ['PATH']=='/usr/bin:/bin' and os.environ['VIRTUAL_ENV']=='/other'


def test_failure_tail_surfaces_child_root_cause_and_redacts_key(tmp_path,capsys):
    from scripts.run_prefill_service import failure_log_tails
    secret='test-only-secret'
    (tmp_path/'server.log').write_text('old line\n'*100+
        'TypeError: TorchVersion is not serializable\n'+secret+'\n')
    # A missing/empty gateway log must not hide the server root cause.
    failure_log_tails(tmp_path,secret)
    err=capsys.readouterr().err
    assert 'TorchVersion is not serializable' in err and '[REDACTED]' in err
    assert secret not in err and err.count('old line')==78
    failure_log_tails(tmp_path/'missing',secret)


def test_targeted_diagnostic_requires_exact_acceptance_and_uses_distinct_output(monkeypatch,tmp_path):
    mock_host(monkeypatch,tmp_path);host.prepare_host(tmp_path/'inputs');calls=[]
    monkeypatch.setattr(host.subprocess,'run',lambda args,**kw:calls.append(args))
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path))
    host.collect_host(tmp_path/'inputs',diagnostics=True)
    assert '--diagnostics-only' in calls[0]
    assert '/pr32-diagnostics/' in calls[0][calls[0].index('--output')+1]
    monkeypatch.setattr(host,'commit',lambda:'c'*40)
    with pytest.raises(RuntimeError,match='exact checkout'):host.collect_host(tmp_path/'inputs',diagnostics=True)
    assert len(calls)==1


def test_calibration_host_uses_dedicated_manifest_and_archives_failed_attempt(monkeypatch,tmp_path):
    import tarfile
    import subprocess
    mock_host(monkeypatch,tmp_path);ready=host.prepare_host(tmp_path/'inputs');calls=[]
    monkeypatch.setattr(host.Path,'home',classmethod(lambda cls:tmp_path))
    def collect(args,**kw):
        calls.append(args);capture=Path(args[args.index('--output')+1]);capture.mkdir()
        (capture/'calibration.json').write_text('{"collection_complete":false}')
        raise subprocess.CalledProcessError(2,args)
    monkeypatch.setattr(host.subprocess,'run',collect)
    with pytest.raises(subprocess.CalledProcessError):host.collect_host(tmp_path/'inputs',calibration=True)
    args=calls[0]
    assert '--calibration-only' in args
    assert args[args.index('--manifest')+1]==str(tmp_path/'inputs/calibration-tokens.json')
    assert args[args.index('--cpu-acceptance')+1]==ready['cpu_acceptance']
    archives=list((tmp_path/'pr32-calibrations').glob('*.tar.gz'));assert len(archives)==1
    with tarfile.open(archives[0]) as bundle:
        assert any(n.endswith('/calibration.json') for n in bundle.getnames())


def test_setup_calibration_uses_declared_pins_and_does_not_collect(monkeypatch,tmp_path):
    from scripts import setup_prefill_calibration as setup
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    monkeypatch.setattr(setup.Path,'home',classmethod(lambda cls:tmp_path))
    monkeypatch.setattr(setup.subprocess,'check_output',lambda *a,**kw:'')
    calls=[]
    monkeypatch.setattr(setup.subprocess,'run',lambda args,**kw:calls.append(args))
    setup.main()
    assert any('ninja' in a and 'requirements-v1.txt' in a and 'requirements-cpu-tests.txt' in a for a in calls)
    assert calls[-1][-2:]==['scripts/prefill_host.py','prepare']
    assert not any('collect' in a or 'calibrate' in a for a in calls)
    pin=json.loads((tmp_path/'pr32-inputs/model.json').read_text());assert pin['revision']==setup.REVISION
    probe=next(a[-1] for a in calls if a[-2]=='-c' and 'installed_sources' in a[-1])
    # Compile the actual child probe and verify its import target exists.
    compile(probe,'setup-probe','exec')
    from prefill_lab.plugin import installed_sources
    assert callable(installed_sources)
