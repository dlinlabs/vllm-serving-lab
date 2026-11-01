"""First-entry estimator isolation and chronology; CPU evidence only."""
import copy
import statistics
import pytest
from prefill_lab import entry_clock as entry,submit_clock
from test_prefill_submit_clock import evidence


def captures():
    p,caps=evidence()
    for cap in caps:
        cap['release']=0.;rid=f'plab/{cap["trial_id"]}/foreground-0'
        cap['requests']={rid:dict(input_tokens=1024)}
        cap['steps'][0]['work']=[dict(request_id=rid,position=0,tokens=1024)]
        cap['steps'][1]['work']=[dict(request_id=rid,position=1024,tokens=1)]
    return p,caps


def test_entry_estimator_ignores_later_prefill_in_nominal_but_retains_domain():
    p,caps=captures();first=[]
    for cap in caps:
        rid=next(iter(cap['requests']));cap['foreground_done']=.28;cap['steps'].append(dict(step_id='2',start=.2,end=.28,prefill=512,
            work=[dict(request_id=rid,position=512,tokens=512)]))
        cap['workers'][0]['rows'].append(dict(step_id='2',host_start=.201,host_end=.249))
        # Step end encloses the worker span; later labels must not change P.
        first.append(cap['workers'][0]['rows'][0]['host_end']-cap['workers'][0]['rows'][0]['host_start'])
    f=entry.training_labels(caps,p);old=submit_clock.training_labels(caps,p)
    assert f['constants_s']['prefill']==statistics.median(first)
    assert f['predecessor_prefill_s']==old['constants_s']['prefill']
    assert f['constants_s']['decode']==old['constants_s']['decode']
    assert (f['prefill_lower_s'],f['prefill_upper_s'])==(old['prefill_lower_s'],old['prefill_upper_s'])
    assert len(f['entry_samples'])==24 and not f['probe_labels_used']
    poison=[dict(trial_id=t['trial_id'],steps='do not inspect') for t in p['order'] if t['stage']!='train']
    assert entry.training_labels(caps+poison,p)==f


@pytest.mark.parametrize('fault',['release','early','missing','background','computed','duplicate'])
def test_entry_identity_or_clock_cannot_be_imputed(fault):
    p,caps=captures();cap=caps[0]
    if fault=='release':cap['release']=float('nan')
    if fault=='early':cap['release']=.01
    if fault=='missing':cap['steps'][0]['work']=[]
    if fault=='background':cap['steps'][0]['work'][0]['request_id']='plab/x/background-0'
    if fault=='computed':cap['steps'][0]['work'][0]['position']=1024
    if fault=='duplicate':caps.append(copy.deepcopy(cap))
    with pytest.raises(ValueError):entry.training_labels(caps,p)
