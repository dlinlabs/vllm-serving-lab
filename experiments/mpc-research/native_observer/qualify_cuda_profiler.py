"""CUPTI availability smoke; not a serving measurement or controller benchmark."""
import json
from pathlib import Path
import sys
import time


def main(directory):
    import torch

    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    a = torch.ones((1024,1024),device="cuda")
    torch.mm(a,a)
    torch.cuda.synchronize()
    begin = time.monotonic()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,with_stack=False,
    ) as profiler:
        torch.mm(a,a)
        torch.cuda.synchronize()
    end = time.monotonic()
    trace = directory/"cuda-smoke.json"
    profiler.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat")=="kernel"]
    report = dict(schema="cuda-profiler-availability-v1",
        cuda_kernels=len(kernels),kernel_duration_us=sum(e["dur"] for e in kernels),
        available=bool(kernels),begin=begin,end=end,
        limitation="Explicit synchronize only in isolated smoke; no serving or latency claims.")
    (directory/"qualification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not kernels:
        raise RuntimeError("CUPTI GPU events unavailable; do not infer GPU work from CPU ranges")


if __name__=="__main__":
    main(sys.argv[1])
