import pytest

from analyze_profile import analyze_trace, union_us


def fixture():
    name = "execute_128_context_1(sq128sk128sqsq16384sqsk16384)_generation_0(sq0sk0sqsq0sqsk0)"
    return {"traceEvents":[
        dict(cat="kernel",ph="X",pid=0,tid=7,ts=11.,dur=2),
        dict(cat="kernel",ph="X",pid=0,tid=7,ts=12.,dur=3),
        dict(cat="gpu_user_annotation",ph="X",pid=0,tid=7,ts=10.,dur=6,
             name=name,args={"External id":1}),
        dict(cat="user_annotation",ph="X",pid=100,tid=100,ts=1.,dur=100,
             name=name,args={"External id":1})]}


def test_actual_gpu_events_not_cpu_duration_and_overlap_counted_once():
    result = analyze_trace(fixture())
    batch = result["batches"][0]
    assert batch["gpu_span_us"]==6 and batch["kernel_busy_union_us"]==4
    assert batch["cpu_annotation_us"]==100
    assert result["kernels_assigned_to_batch"]==2
    assert union_us([(1,5),(2,3),(4,7)])==6


def test_cpu_only_trace_cannot_qualify_gpu_service():
    document = fixture()
    document["traceEvents"] = [e for e in document["traceEvents"] if e["cat"]!="kernel"]
    with pytest.raises(ValueError,match="actual GPU"):
        analyze_trace(document)


def test_ambiguous_gpu_batch_attribution_is_rejected():
    document = fixture()
    document["traceEvents"].append(document["traceEvents"][2].copy())
    with pytest.raises(ValueError,match="ambiguous"):
        analyze_trace(document)
