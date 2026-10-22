"""Synthetic recalibration counterexample; no fitted model and no GPU data."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prefill_design import simulate,requests_for,COEF,contrast_eligibility
from prefill_validation import comparison_plan,paired_error_check
from scripts.build_prefill_design import cases,digest,write


def predictions(coefficients):
    return {c['id']:simulate(requests_for(c),2048,backgrounds=c['background_decoders'],coefficients=coefficients)['foreground']['sentinel']['ttft_s']
            for c in cases() if c['family']=='sentinel'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():p.error('Choose fresh output')
    raw=predictions(COEF);jitter={k:.005 for k in raw};rows=[]
    for scale in (.95,1.05,.9,1.1):
        calibrated=predictions(tuple(c*scale for c in COEF))
        old=contrast_eligibility(calibrated,raw,jitter)
        # Treat the rescaled linear toy as synthetic truth, with5 identical blocks.
        errors={k:[0.]*5 for k in raw};baseline_errors={k:[raw[k]-calibrated[k]]*5 for k in raw}
        old_check=paired_error_check(old,errors,baseline_errors)
        revised=comparison_plan('M0','M0',calibrated,calibrated,raw,jitter)
        rows.append(dict(scale=scale,delta_s={k:calibrated[k]-raw[k] for k in raw},
            old_raw_baseline_eligibility=old,old_gate3_component=old_check,revised_comparison_plan=revised))
    write(a.output,dict(role='synthetic counterexample: calibration gain is not structural gain',
        measured_data_used=False,models_fitted=False,jitter_is_synthetic=True,jitter_s=.005,
        frozen_coefficients=COEF,source_sha256={n:digest(ROOT/n) for n in ['prefill_design.py','prefill_validation.py','scripts/build_prefill_design.py','scripts/check_prefill_comparators.py']},cases=rows))
    print(json.dumps([dict(scale=r['scale'],eligible=len(r['old_raw_baseline_eligibility']['eligible_cells']),old_component=r['old_gate3_component']['component_status'],structure=r['revised_comparison_plan']['structure_status']) for r in rows],indent=2))


if __name__=='__main__':main()
