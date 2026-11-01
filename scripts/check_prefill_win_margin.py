"""Deterministic Gate3 component counterexamples, not an experiment false-positive rate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prefill_design import contrast_eligibility
from prefill_validation import paired_error_check
from scripts.build_prefill_design import digest,write


def checks():
    base={str(i):.2 for i in range(8)}
    candidate={k:v+(.06 if int(k)<4 else 0.) for k,v in base.items()}
    eligibility=contrast_eligibility(candidate,base,{k:.005 for k in base})
    rows=[]
    for name,gain,regression in [('epsilon',.000001,False),('equal_margin',.025,False),
                                 ('resolved_gain',.026,False),('noneligible_regression',.026,True)]:
        b={k:[.05]*5 for k in base};a={k:list(v) for k,v in b.items()}
        for k in ('0','1','2'):a[k]=[.05-gain]*5
        if regression:a['7']=[.1]*5
        old_wins=[k for k in eligibility['eligible_cells'] if sum(a[k])<sum(b[k])]
        rows.append(dict(case=name,old_marginless_win_count=len(old_wins),
            old_win_count_requirement_met=len(old_wins)>=eligibility['strict_wins_required'],
            revised=paired_error_check(eligibility,a,b)))
    return dict(role=__doc__,synthetic=True,eligibility=eligibility,cases=rows,
        note='No Monte Carlo, confidence interval, complete-experiment pass or experiment false-positive rate is claimed.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Choose fresh output')
    result=checks();result['source_sha256']={n:digest(ROOT/n) for n in
        ('prefill_design.py','prefill_validation.py','scripts/check_prefill_win_margin.py')}
    write(a.output,result)
    print(json.dumps({r['case']:r['revised']['component_status'] for r in result['cases']},indent=2))
