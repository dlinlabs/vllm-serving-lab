"""Correlation failures must not silently become timing/coverage claims."""
from copy import deepcopy

import pytest

from scripts.analyze_prefill_diagnostic import correlate, union_us


def events():
    return [
        dict(cat='user_annotation',name='prefill_lab_step/a',pid=10,tid=10,ts=0.,dur=10.,args={'External id':1}),
        dict(cat='cuda_runtime',name='cudaGraphLaunch',pid=10,tid=10,ts=2.,dur=1.,args={'correlation':100}),
        # Two graph nodes share one launch correlation and execute after CPU return.
        dict(cat='kernel',name='node1',pid=0,tid=7,ts=20.,dur=5.,args={'correlation':100,'stream':7}),
        dict(cat='kernel',name='node2',pid=0,tid=7,ts=25.,dur=4.,args={'correlation':100,'stream':7}),
        dict(cat='gpu_user_annotation',name='prefill_lab_step/a',pid=0,tid=7,ts=20.,dur=9.,args={'External id':1}),
        dict(cat='cuda_driver',name='cuLaunchKernel',pid=10,tid=10,ts=12.,dur=1.,args={'correlation':101}),
        dict(cat='kernel',name='sample',pid=0,tid=7,ts=30.,dur=2.,args={'correlation':101,'stream':7}),
    ]


def test_graph_nodes_join_by_launch_not_gpu_timestamp_or_stream_handle():
    d=correlate(events(),['a'])
    assert d['steps'][0]['activity_counts']=={'kernel':2}
    assert d['steps'][0]['gpu_envelope_us']==9
    assert d['steps'][0]['gpu_activity_union_us']==9
    assert d['outside_forward_counts']=={'kernel':1}
    assert not d['cuda_scope_qualified'] and not d['model_accepted']


def test_same_clock_time_on_another_thread_is_not_owned_by_cpu_step():
    es=events();es[1]['tid']=11
    with pytest.raises(ValueError,match='no correlated GPU work'):
        correlate(es,['a'])


@pytest.mark.parametrize('damage',['cpu_marker','duplicate_marker','gpu_annotation','external_id','correlation','duplicate_correlation','order'])
def test_missing_or_ambiguous_trace_identity_is_rejected(damage):
    es=deepcopy(events())
    if damage=='cpu_marker':es.pop(0)
    elif damage=='duplicate_marker':es.append(deepcopy(es[0]))
    elif damage=='gpu_annotation':es.pop(4)
    elif damage=='external_id':es[4]['args']['External id']=2
    elif damage=='correlation':es[2]['args']['correlation']=999
    elif damage=='duplicate_correlation':es.append(deepcopy(es[1]))
    elif damage=='order':
        b=deepcopy(es[0]);b.update(name='prefill_lab_step/b',ts=-10.);es.append(b)
    with pytest.raises(ValueError):correlate(es,['a','b'] if damage=='order' else ['a'])


def test_gpu_annotation_is_not_assumed_to_contain_every_correlated_kernel():
    es=events();es[4]['dur']=6.
    d=correlate(es,['a'])
    assert d['steps'][0]['correlated_activities_outside_gpu_annotation']==1
    assert d['steps'][0]['gpu_envelope_us']==9
    assert not d['cuda_scope_qualified']


def test_interval_union_handles_overlap_nested_and_touching_without_double_counting():
    es=[dict(ts=a,dur=b-a) for a,b in [(0,10),(2,5),(8,12),(12,14)]]
    assert union_us(es)==14
    assert union_us([])==0
    with pytest.raises(ValueError):union_us([dict(ts=1,dur=-1)])
