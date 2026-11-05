"""Same-boot controlled-to-HTTP model handoff; no held-dependent selection."""
import hashlib
from pathlib import Path
import time
from . import event_clock_calibration as clock,entry_clock,queue_confirmation as qc,submit_clock
from .contracts import digest,finite
from .live_http import clock_domain


def costs(config,runtime,*,now=None,domain=None,sources=None):
    from scripts.run_prefill_service import code_identity
    now=time.monotonic() if now is None else now;domain=clock_domain() if domain is None else domain
    if (config.get('schema')!='entry-http-model-config-v1' or
            config.get('sha256')!=digest({k:v for k,v in config.items() if k!='sha256'}) or
            config.get('clock_domain')!=domain or config.get('source_sha256')!=(code_identity() if sources is None else sources)):
        raise ValueError('Entry HTTP configuration/source/clock domain differs')
    before,at=config.get('controlled_finished_at'),config.get('configured_at')
    if not all(finite(v) for v in (before,at,now)) or not 0<=before<=at<=now:raise ValueError('Invalid entry HTTP handoff chronology')
    if config.get('controlled_candidate_passed') is not True:raise ValueError('Controlled confirmation prerequisite failed')
    p=qc.protocol();base=config['base_model'];entry=config['entry_freeze']
    if (base.get('measurement_origin')!='fresh_native_training_events' or
            entry.get('schema')!='entry-submit-clock-freeze-v1' or entry.get('readiness_passed') is not True or
            entry.get('sha256')!=digest({k:v for k,v in entry.items() if k!='sha256'}) or
            entry.get('base_model_sha256')!=base.get('sha256') or entry.get('plan_sha256')!=digest(p) or
            entry.get('runtime_sha256')!=digest(runtime) or entry.get('implementation_sha256')!=hashlib.sha256(Path(entry_clock.__file__).read_bytes()).hexdigest()):
        raise ValueError('Entry HTTP base/entry/plan/runtime identity differs')
    f0=clock.predict_frozen(base,p,runtime)
    g=entry['labels']['constants_s'];wanted=sorted(t['trial_id'] for t in p['order'] if t['stage']=='train' and t['observer']=='light')
    if entry['labels']['training_trial_ids']!=wanted:raise ValueError('Entry HTTP TRAIN membership differs')
    predicted=clock.forecasts(submit_clock.own_rollouts(base['parameters_s'],p,runtime,g['prefill'],g['decode']))
    if digest(predicted)!=digest(entry['predictions']) or digest(f0)!=digest(entry['f0_predictions']):raise ValueError('Entry HTTP own forecasts differ')
    return submit_clock.costs(base['parameters_s'],p,g['prefill'],g['decode'])


def configure(base,entry,report,runtime,*,finished_at,http_manifest_sha256):
    from scripts.run_prefill_service import code_identity
    if report.get('candidate_passed') is not True or report.get('frozen_sha256')!=entry.get('sha256'):
        raise ValueError('Entry HTTP requires the passing sealed controlled report')
    config=dict(schema='entry-http-model-config-v1',clock_domain=clock_domain(),configured_at=time.monotonic(),
        controlled_finished_at=finished_at,source_sha256=code_identity(),controlled_candidate_passed=True,
        controlled_report_sha256=digest(report),http_manifest_sha256=http_manifest_sha256,
        base_model=base,entry_freeze=entry,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    config['sha256']=digest(config);costs(config,runtime);return config
