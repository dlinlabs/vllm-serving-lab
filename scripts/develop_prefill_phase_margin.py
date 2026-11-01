"""Source-audited CPU development of a TRAIN-bound phase-margin proposal.

This is retrospective development, never original or fresh GPU acceptance.
Existing clock/live-http collection gates are not modified by this command.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab import queue_phase,event_clock_calibration as clock
from prefill_lab.contracts import save,digest
from scripts.analyze_prefill_queue_phase import original_audit
from scripts.analyze_prefill_effective_capture import archive_identity


def run(capture,archive,output):
    identity,index=archive_identity(archive,capture);proof=original_audit(capture)
    read=lambda name:json.loads((capture/name).read_text())
    plan=read('calibration-plan.json');runtime=read('runtime.json');model=read('frozen.json');rows=read('endpoints.json')
    predicted=clock.predict_frozen(model,plan,runtime)
    if digest(predicted)!=digest(read('pre-held-predictions.json')):raise ValueError('Development predictor differs from captured point predictions')
    # Save the TRAIN-derived freeze before held scoring. The archive and its
    # labels preexist: this replay is development, never prospective confirmation.
    frozen=queue_phase.freeze(rows,plan,runtime,model,predicted)
    save(output/'phase-freeze.json',frozen);save(output/'proposed-protocol.json',queue_phase.protocol());save(output/'policy.json',queue_phase.policy())
    report=queue_phase.evaluate(rows,plan,runtime,model,predicted,frozen) if frozen['readiness_passed'] else None
    if report is not None:save(output/'phase-report.json',report)
    summary=dict(schema='phase-margin-development-v1',archive=identity,original_audit=proof,
        native_original_candidate_passed=proof['candidate_passed'],development_candidate_passed=report['candidate_passed'] if report else False,
        retrospective_development_only=True,fresh_native_confirmation_performed=False,
        six_term_model_refitted=False,phase_training_only=True,scale_half_width=frozen['scale_half_width'],
        readiness_passed=frozen['readiness_passed'],phase_cells={k:v['status'] for k,v in frozen['cells'].items()},
        model_accepted=False,http_accepted=False,adaptive_cap_accepted=False,mpc_accepted=False)
    save(output/'summary.json',summary);save(output/'input-files.json',index)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','archive','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    print(json.dumps(run(a.capture.resolve(),a.archive.resolve(),a.output.resolve())))


if __name__=='__main__':main()
