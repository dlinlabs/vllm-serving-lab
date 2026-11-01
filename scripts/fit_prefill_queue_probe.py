"""Offline same-host TRAIN-only clock fit; receipt probes are development data."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab import queue_probe as probe
from prefill_lab.contracts import digest,save
from scripts.check_prefill_queue_probe import audit


def fit(directory,output):
    directory=Path(directory).resolve();output=Path(output).resolve()
    if output==directory or directory in output.parents:raise ValueError('Keep fit output outside immutable capture')
    audit(directory);load=lambda name:json.loads((directory/name).read_text())
    p=probe.protocol();runtime=load('runtime.json')
    rows=[r for r in load('queue-probe-rows.json') if r['stage']=='train'];events=load('training-events.json')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='queue-probe-clock-fit-v1',fit_complete=False,plan_sha256=digest(p),
        training_rows_sha256=digest(rows),training_events_sha256=digest(events),delayed_probe_labels_used=False,
        held_confirmation_performed=False,model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False,failures=[])
    try:
        frozen=probe.clock.fit(rows,events,p,runtime,deadline=time.monotonic()+180,origin='fresh_native_training_events')
        save(output/'frozen.json',frozen);save(output/'predictions.json',probe.clock.predict_frozen(frozen,p,runtime))
        result.update(fit_complete=True,frozen_sha256=frozen['sha256'])
    except Exception as exc:result['failures'].append(dict(type=type(exc).__name__,message=str(exc)));raise
    finally:save(output/'fit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(fit(args.directory,args.output)))
