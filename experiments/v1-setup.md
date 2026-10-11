# Set up the V1 core environment and start Qwen

Use a Linux x86_64 NVIDIA GPU rental template with a working host NVIDIA driver,
Python 3.12, pip and Python's `venv` module. Clone this repo and run:

```bash
bash setup_and_start.sh
```

If the template's Python 3.12 executable is named `python3`:

```bash
PYTHON_BIN=python3 bash setup_and_start.sh
```

The script checks Python, GPU/driver visibility via `nvidia-smi`, creates a local
`.venv`, installs the recorded V1 core pins, runs `pip check`, verifies the actual
Torch CUDA runtime, performs a GPU matrix multiplication, then starts vLLM.
Manual activation is not required for setup: child processes receive `.venv/bin`
at the front of `PATH`, so FlashInfer can find `ninja`. Setup checks `ninja` before
launching vLLM, including with `--skip-install`.

It does not install or replace the host NVIDIA driver or system Python. Use an
appropriate template if these prerequisites are missing. V1 used Python 3.12.3;
the script accepts Python 3.12.x and records the exact patch version.

Core requirements are `vllm==0.30.0` and `torch==2.13.0+cu130`, matching the README.
Both resolve in one pip operation; incompatible dependencies or unavailable binary
wheels cause failure rather than silently changing versions. CUDA 13.0 is the
Torch wheel's runtime, not necessarily the toolkit reported by `nvcc` or the maximum
CUDA version shown by `nvidia-smi`. The actual CUDA compute check is required.

**No full historical V1 lockfile exists.** Ancillary/transitive versions cannot be
claimed identical to V1; setup records `pip-freeze.txt` and an installation report
for subsequent reproduction. Do not install unrelated packages into this `.venv`.
The script reuses an existing Linux Python 3.12 `.venv`; use a fresh checkout if you
need to preserve an environment with other dependencies.

## Context length is fixed, explicitly and at runtime

The exact launch command is:

```bash
.venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8000 --max-model-len 8192
```

8192 is the per-request **prompt plus generated output** limit, including chat
formatting tokens. It is not 8192 input tokens plus a separate output allowance.
The model's native/default context limit must not be allowed to replace this V1
setting. Large native context limits can change memory requirements and prevent
startup on a 24 GB card. This script neither auto-expands the limit nor silently
shrinks it after an OOM. It checks the served model ID and `max_model_len=8192` in
`/v1/models`; mismatched or absent values stop startup.

First startup downloads model artifacts if absent from the Hugging Face cache.
Allow up to 30 minutes by default for download, compilation and loading:

```bash
bash setup_and_start.sh --startup-timeout 3600
```

The server must respond with the correct model/context and its tokenizer must load
from the local cache before `READY` is printed. This is readiness validation, not
a GPU load benchmark. The model revision is not historically pinned by V1.

## Logs and lifecycle

Setup writes `results/setup-<timestamp>/` with host/runtime checks, installed
versions, the exact launch command, model metadata, and `vllm.log`. Follow that
log from another terminal while the model loads. A failure preserves these files.

Keep the setup terminal running. Ctrl+C stops the owned vLLM process group. The
script refuses to use port 8000 if already occupied and never terminates a server
it did not start. The server binds locally; run the experiment on the same machine.

To restart an already-installed environment without reinstalling:

```bash
bash setup_and_start.sh --skip-install
```

Version, dependency, CUDA, and model-context checks still run.

In a second terminal at the repo root:

```bash
source .venv/bin/activate
```

Then use the smoke/pilot commands from `experiments/budget-step-pilot.md` once
PR #4 has been merged. This setup PR is independent of that experiment runner;
it prepares the environment and keeps vLLM ready, but does not automatically run
paid GPU experiments or shut down the rented machine.

## CPU validation

```bash
python -m unittest discover -s tests -p test_setup_v1.py
bash -n setup_and_start.sh
```

These tests validate command construction and readiness/error handling. Actual
large-wheel installation, model download and CUDA startup require the rented GPU.
