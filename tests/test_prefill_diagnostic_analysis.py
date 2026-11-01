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


def test_stage_profile_tail_is_distinguished_without_accepting_gpu_scope():
    d=correlate(events(),['a'],sampling_profiled=True)
    assert not d['profile_stop_precedes_fifth_sampling']
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


def stage_fixture():
    markers=[];apis=[];gpu=[]
    windows={'forward':(0.,10.),'logits_sample':(11.,13.),'copy_enqueue':(14.,15.),'post_update':(16.,17.)}
    for i,(name,(a,b)) in enumerate(windows.items()):
        label='prefill_lab_step/a' if name=='forward' else 'prefill_lab_stage/'+name+'/a'
        markers.append(dict(cat='user_annotation',name=label,pid=10,tid=10,ts=a,dur=b-a))
        apis.append(dict(cat='cuda_runtime',name='launch',pid=10,tid=10,ts=a+.1,dur=.1,args={'correlation':i}))
        start,duration,stream=(20.,10.,7) if i==0 else (30.,1.,7) if i==1 else (31.,1.,13) if i==2 else (31.5,1.,7)
        gpu.append(dict(cat='gpu_memcpy' if i==2 else 'kernel',name=name,pid=0,tid=stream,ts=start,dur=duration,args={'correlation':i,'stream':stream}))
    # The parent must not receive the same launches as its children.
    markers.append(dict(cat='user_annotation',name='prefill_lab_stage/sample_tokens/a',pid=10,tid=10,ts=10.,dur=10.))
    row=dict(step_id='a',primary_copy_bytes=12,extra_copy_payloads=False,
        stages={n:dict(stream_handle=22 if n=='copy_enqueue' else 0,gpu_elapsed_s=.1 if n=='copy_enqueue' else .001) for n in [*windows,'sample_tokens']})
    return markers+apis+gpu,[row]


def test_leaf_join_avoids_parent_double_counting_and_measures_copy_update_overlap():
    from scripts.analyze_prefill_diagnostic import correlate_stages
    es,rows=stage_fixture();d=correlate_stages(es,['a'],rows);s=d['steps'][0]
    assert d['gpu_activity_count']==4 and d['unmatched_gpu_activities']==0
    assert s['copy_post_update_overlap_us']==.5
    assert s['stages']['copy_enqueue']['envelope_us']==1
    assert s['stages']['copy_enqueue']['event_elapsed_s']==.1
    assert s['stages']['copy_enqueue']['runtime_stream_handle']==22
    assert s['stages']['copy_enqueue']['cupti_streams']==[13]
    assert not d['cuda_scope_qualified'] and not d['model_accepted']


@pytest.mark.parametrize('damage',['missing_marker','duplicate_marker','wrong_thread','duplicate_launch','missing_launch','outside_leaf','overlapping_leaf','missing_gpu','duplicate_step'])
def test_invalid_leaf_launch_identity_is_not_guessed(damage):
    from scripts.analyze_prefill_diagnostic import correlate_stages
    es,rows=stage_fixture()
    if damage=='missing_marker':es.pop(0)
    elif damage=='duplicate_marker':es.append(deepcopy(es[0]))
    elif damage=='wrong_thread':es[5]['tid']=99
    elif damage=='duplicate_launch':es.append(deepcopy(es[5]))
    elif damage=='missing_launch':es[9]['args']['correlation']=999
    elif damage=='outside_leaf':es[5]['ts']=100.
    elif damage=='overlapping_leaf':es[1]['ts']=0.;es[1]['dur']=10.
    elif damage=='missing_gpu':es.pop(9)
    with pytest.raises(ValueError):correlate_stages(es,['a','a'] if damage=='duplicate_step' else ['a'],rows)
