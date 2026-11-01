"""Exact-deployment, loopback-only diagnostic launcher; never a benchmark."""
import json
import os
from pathlib import Path

ROOT = Path("/root/pr33-native-0588707")


if __name__=="__main__":
    config = json.loads(Path("/root/pr33-native-control.json").read_text())
    services = ROOT/"native-profile-services"
    services.mkdir(exist_ok=True)
    if (services/"vllm.pid").exists():
        raise ValueError("profile service evidence already exists")
    (services/"vllm.pid").write_text(str(os.getpid())+"\n")
    os.chdir(ROOT)
    os.environ.update(PATH=str(ROOT/".venv/bin")+":"+os.environ["PATH"],
        VLLM_PLUGINS="pr33_native_observer",PR33_NATIVE_OBSERVER_ENABLED="1",
        PR33_NATIVE_OBSERVER_KEY=config["ADMIN_API_KEY"],
        HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",VLLM_NO_USAGE_STATS="1")
    executable = str(ROOT/".venv/bin/vllm")
    profiler = dict(profiler="torch",torch_profiler_dir=str(ROOT/"native-profile-traces"),
        torch_profiler_with_stack=False,torch_profiler_record_shapes=False,
        detailed_trace_annotation=True,ignore_frontend=True)
    args = [executable,"serve",config["MODEL_NAME"],"--served-model-name",
        "Qwen/Qwen3-4B-Instruct-2507","--host","127.0.0.1","--port","8000",
        "--dtype","bfloat16","--max-model-len","8192","--max-num-seqs","32",
        "--max-num-batched-tokens","2048","--gpu-memory-utilization","0.8",
        "--no-enable-prefix-caching","--tensor-parallel-size","1",
        "--enable-request-id-headers","--profiler-config",json.dumps(profiler)]
    (services/"configuration.json").write_text(json.dumps(dict(arguments=args,
        scope="Profiler diagnostics only, not performance or model calibration acceptance."),
        indent=2)+"\n")
    os.execv(executable,args)
