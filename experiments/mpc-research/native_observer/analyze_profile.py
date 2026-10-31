"""GPU trace timing diagnostics, never CPU-duration substitution or MPC calibration."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ANNOTATION = re.compile(
    r"execute_(\d+)_context_(\d+)\(sq(\d+)sk(\d+)sqsq(\d+)sqsk(\d+)\)"
    r"_generation_(\d+)\(sq(\d+)sk(\d+)sqsq(\d+)sqsk(\d+)\)")


def union_us(intervals):
    end, total = -math.inf,0.
    for begin,finish in sorted(intervals):
        if not all(math.isfinite(x) for x in (begin,finish)) or finish<begin:
            raise ValueError("invalid GPU interval")
        total += max(0,finish-max(begin,end))
        end = max(end,finish)
    return total


def analyze_trace(document):
    events = document["traceEvents"]
    kernels = [e for e in events if e.get("cat")=="kernel" and e.get("ph")=="X"]
    if not kernels:
        raise ValueError("no actual GPU kernel events")
    annotations = [e for e in events if e.get("cat")=="gpu_user_annotation"
                   and ANNOTATION.fullmatch(e["name"])]
    if not annotations:
        raise ValueError("no GPU-side batch annotations; CPU ranges are not service measurements")
    cpu = {e["args"]["External id"]:e for e in events if e.get("cat")=="user_annotation"
           and "External id" in e.get("args",{})}
    groups, batches, assigned = {},[],set()
    for event in annotations:
        groups.setdefault((event["pid"],event["tid"]),[]).append(event)
    for group in groups.values():
        ordered = sorted(group,key=lambda e:e["ts"])
        if any(a["ts"]+a["dur"]>b["ts"]+.001 for a,b in zip(ordered,ordered[1:])):
            raise ValueError("overlapping GPU batch annotations; interval attribution is ambiguous")
    for event in sorted(annotations,key=lambda e:e["ts"]):
        counters = tuple(map(int,ANNOTATION.fullmatch(event["name"]).groups()))
        total,pcount,ptokens,pk,psq,psqsk,dcount,dtokens,dk,dsq,dsqsk = counters
        if total!=ptokens+dtokens:
            raise ValueError("batch annotation token accounting")
        external_id = event["args"]["External id"]
        host = cpu.get(external_id)
        if host is None or host["name"]!=event["name"]:
            raise ValueError("GPU annotation has no matching CPU external ID")
        inside = [(i,e) for i,e in enumerate(kernels) if e["pid"]==event["pid"]
                  and e["tid"]==event["tid"] and event["ts"]-.001<=e["ts"]
                  and e["ts"]+e["dur"]<=event["ts"]+event["dur"]+.001]
        if any(i in assigned for i,_ in inside):
            raise ValueError("GPU kernel assigned twice")
        assigned.update(i for i,_ in inside)
        busy = union_us((e["ts"],e["ts"]+e["dur"]) for _,e in inside)
        if not inside or busy>event["dur"]+.001:
            raise ValueError("missing or inconsistent in-batch GPU kernels")
        batches.append(dict(external_id=external_id,annotation=event["name"],
            gpu_trace_ts_us=event["ts"],gpu_span_us=event["dur"],
            kernel_busy_union_us=busy,cpu_annotation_us=host["dur"],kernels=len(inside),
            scheduled_tokens=total,prefill_requests=pcount,prefill_query_tokens=ptokens,
            prefill_key_tokens_sum=pk,prefill_query_squared_sum=psq,
            prefill_query_key_product_sum=psqsk,decode_requests=dcount,
            decode_query_tokens=dtokens,decode_key_tokens_sum=dk,
            decode_query_squared_sum=dsq,decode_query_key_product_sum=dsqsk))
    cells = {}
    for batch in batches:
        key = (batch["prefill_requests"],batch["prefill_query_tokens"],batch["decode_requests"])
        cells.setdefault(key,[]).append(batch)
    summaries = []
    for (pcount,ptokens,dcount),rows in sorted(cells.items()):
        summaries.append(dict(prefill_requests=pcount,prefill_query_tokens=ptokens,decode_requests=dcount,
            batches=len(rows),median_gpu_span_us=statistics.median(r["gpu_span_us"] for r in rows),
            median_kernel_busy_union_us=statistics.median(r["kernel_busy_union_us"] for r in rows),
            median_cpu_annotation_us=statistics.median(r["cpu_annotation_us"] for r in rows)))
    return dict(schema="native-gpu-profile-diagnostic-v1",gpu_kernels=len(kernels),
        kernels_assigned_to_batch=len(assigned),kernels_outside_batch_annotations=len(kernels)-len(assigned),
        gpu_batches=len(batches),cells=summaries,batches=batches,
        calibration_accepted=False,performance_accepted=False,
        limitations=["Profiling changes timing; this is not unprofiled service calibration or benchmark.",
            "GPU span includes gaps; kernel union is device active interval, not exclusive per-request demand.",
            "Labels are native scheduled batch counters, not GPU-completed progress or request-level attribution.",
            "CPU and GPU ranges join by profiler External id; no monotonic/wall-clock conversion is assumed.",
            "Only two repeats of four offered-burst conditions, max8 deliberate truncation; no held-out inference.",
            "No request-to-batch identity or mixed-batch causal decomposition has been qualified."])


def analyze(directory):
    directory = Path(directory)
    report = json.loads((directory/"native-profile-http/profile-http.json").read_text())
    if not report.get("completed") or report["profile_stop_status"]!=200:
        raise ValueError("profiler HTTP collection incomplete")
    paths = list((directory/"native-profile-traces").glob("*.pt.trace.json.gz"))
    if len(paths)!=1:
        raise ValueError("expected one rank0 native trace")
    path = paths[0]
    with gzip.open(path,"rt") as source:
        result = analyze_trace(json.load(source))
    rows = [r for g in report["groups"] for r in g["rows"]]
    if len(report["groups"])!=10 or len(rows)!=22:
        raise ValueError("profile planned cohort differs")
    if any(r["status"]!="complete" or r["response_request_id"]!=r["request_id"] for r in rows):
        raise ValueError("profile request failure")
    result.update(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        collector_sha256=report["collector_sha256"],
        complete_requests=len(rows),finish_reasons=dict(Counter(r["finish_reason"] for r in rows)),
        preludes=sum(g["prelude"] for g in report["groups"]))
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("gpu_kernels","gpu_batches","kernels_outside_batch_annotations",
                                           "complete_requests","finish_reasons","cells")},indent=2))
