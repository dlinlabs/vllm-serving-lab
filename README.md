# vLLM Serving Benchmark Lab

**A single-GPU research lab for keeping language-model serving usable under heavy traffic.**

Built with vLLM, Qwen3-4B, and an OpenAI-compatible gateway, this project asks:
**How many requests finish on time, and how many must be rejected to keep the service responsive?**
High GPU utilization alone is not the goal.

## How it works

```text
Test requests -> Admission gateway -> vLLM generates answers -> Measure results
```

The gateway acts like a traffic controller. When overloaded, it rejects some new
requests instead of letting everyone wait too long. This trades rejection for
latency; it does not create more GPU capacity.

| Policy | Plain-language description |
|---|---|
| Unrestricted (`baseline`) | Send every request to the model. |
| Request-count limit (`protected` / count) | Allocate capacity by request count; short and long requests each count as one. |
| Estimated-work limit (`cost_aware` / cost) | Estimate work from prompt length and requested output limit; reject new requests when the total cost budget is full. |

**count16 and cost16 are not equal capacities.** Their budgets use different units.
Cost is neither GPU memory in GB nor measured GPU time. Lowering a cost budget
does not cancel requests already admitted.

## Current status

**Fixed-policy experiments are complete. MPC performance is not yet validated.**

The latest batch completed **80 experiments** on one RTX 4090 backend:
count16, count32, cost16, and cost32 across four arrival seeds and five workloads.
Of 112,652 requests, 77,064 completed and 35,588 were rejected, with no other
terminal failures or output-cap terminations. Four different AI models independently
analyzed the results, then cross-reviewed one another's findings.

There is no universal winner. Cost32 averaged only **0.05 more SLO-qualified
requests per second** than count32 within the observation windows, about 0.7%.
This is descriptive evidence, not statistical significance or an MPC result.
Window-only completions and completions during the subsequent drain are reported
separately. Rejections, timeouts, and failures are not removed from evaluation.

[Full results, metric definitions, and limitations (Traditional Chinese)](https://github.com/dlinlabs/vllm-serving-lab/blob/08b933e65ad2b5054beca1c4651296c9a6684be8/experiments/mpc-research/native-v3/baseline04-four-model-technical-consensus.zh-TW.md)

### Why did cost32 only slightly outperform count32?

Its gains on long and mixed prompts were largely offset by losses on short prompts.
These are SLO-qualified, window-only completion counts summed across all four seeds;
the short and long rows each combine the 128- and 256-token output-limit workloads.

| Prompt workload | Count32 | Cost32 | Difference |
|---|---:|---:|---:|
| Short: 128 input tokens | 15,697 | 14,737 | -960 |
| Long: 4,096 input tokens | 161 | 913 | +752 |
| Mixed | 1,042 | 1,370 | +328 |
| **Total** | **16,900** | **17,020** | **+120** |

For example, a 128-input/128-output-limit request costs about 1.02 units, so
cost32 permits up to 31 such concurrent reservations. A 4,096-input/128-output-limit
request costs about 4.03 units, permitting only seven. Count32 does not distinguish
their lengths. This illustrates different admission pressure, not an identified
GPU-memory or scheduling mechanism.

The small overall gain is **not evidence of an input-token counting bug**.
The audit matched workload tapes and observed token geometry, but did not retain
every outgoing payload byte. The cost estimate also uses requested output limits,
not actual output lengths or exact remaining GPU work.

PR33 was merged into `analysis/cap-miss-causes-v1`, not `main`. The default branch
provides the base service and benchmarks;
[PR33's research code and data](https://github.com/dlinlabs/vllm-serving-lab/tree/08b933e65ad2b5054beca1c4651296c9a6684be8)
are available at a fixed revision for browsing and reproduction.

## What is MPC?

Model Predictive Control forecasts the next few seconds, compares multi-step
budget plans, **applies only the first action, then replans using real feedback**.
It is neither a fixed limit nor simply reacting to a growing queue.

Next comes real state/control/feedback qualification, a separately versioned model,
and fair comparison against fixed, reactive, and strong one-move policies.
Offline forecasts, CPU tests, or one successful control action do not prove MPC gains.

## Getting started

Reading the reports requires no GPU. For CPU tests, use Linux and Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu-tests.txt
python scripts/run_linux_cpu_tests.py
```

These steps do not start vLLM or download a model. Passing CPU tests does not
qualify a GPU experiment. To start inference, follow the [V1 setup guide](experiments/v1-setup.md)
and check hardware and costs first. Recorded V1 core versions are vLLM `0.30.0`
and PyTorch `2.13.0+cu130`; this is not a complete historical lockfile.

## Code and data

| Location | Purpose |
|---|---|
| [gateway.py](gateway.py) | Admission control and streaming proxy. |
| [run_experiments.py](run_experiments.py) | Generate traffic, compare policies, and save results. |
| [experiments/](experiments/) | Methods, configurations, and runbooks. |
| [results/](results/) | Raw data, analysis, and reproduction tools. |
| [Latest research data](https://github.com/dlinlabs/vllm-serving-lab/tree/08b933e65ad2b5054beca1c4651296c9a6684be8/results/mpc-native-v3) | PR33 model research and native verification evidence. |

Detailed historical results, architecture, and reproduction commands remain in the
[earlier full README](https://github.com/dlinlabs/vllm-serving-lab/blob/aea7694a0af2d1dadc883b4b9e33d1a9bdb61098/README.md).

This is a research lab, not a validated production auto-controller.
