# Resume evidence package A: scoped prototype wording

Use **one** pair, not all six bullets. This package is independent of Native
closed-loop validation (deliverable B): the claims below do not require B to
succeed. Current scope is an implemented serving gateway and CPU-tested control
prototype, not a completed native MPC deployment. Evidence IDs refer to
[benchmark results](benchmark_results.md).

## AI Inference / Infrastructure

- Built an instrumented admission-control gateway in front of vLLM and a
  trace-replay pipeline, retaining request outcomes, overload rejections and
  control-timing evidence from real-GPU scheduled-budget identification captures.
- Implemented CPU-tested finite-horizon admission-policy planning with shared
  forecast inputs, model-evaluated Fixed/OneMove references and deadline/version
  guards; corrected a Fixed-policy timing bias and added regression tests.

Evidence: first bullet E1, including trace identity, archived outcomes,
collection configuration, recorded source hashes and evidence publication
commit; second bullet E6/E8. E2 is a separate frozen planner, not the version
carrying E6's timing repair. The historical feature/forecast reconstruction
command is not a command that rebuilds the E1 outcome table.
The historical captures are not MPC performance measurements. E6's
model-ranked Fixed reference is not an empirically selected strong baseline.

## Autonomous Systems / Runtime

- Designed and CPU-tested a finite-horizon admission-control planning prototype
  that evaluates all candidate paths on one shared Train-only scenario panel,
  with source/clock/version/deadline guards and permit-dependency invalidation.
- Built a Train-only forecast and hash-pinned replay-input pipeline, recording
  its nominal model assumptions, tokenizer-validation evidence and explicit
  limitations instead of claiming calibrated native prediction.

Evidence: E6/E7/E8 and the [architecture](architecture.md). These are current
nominal components, not the separate E2 frozen planner or a demonstrated native loop.
The forecast is nominal and uncalibrated; its CPU validation is not an
out-of-sample accuracy result.

## General Software / Distributed Systems

- Implemented a streaming inference gateway with reservation accounting,
  overload handling and lifecycle telemetry for auditable serving studies.
- Developed replay and independent validation workflows linking request
  outcomes, configurations and commands to recorded source hashes and evidence
  commits, including reproducible workload generation and real-tokenizer
  input checks while preserving negative results.

Evidence: E1/E4/E5/E6/E7/E8. Do not substitute "production deployed" for
"implemented", or claim full clean-checkout benchmark reproduction.

## Claims explicitly excluded

Do not claim MPC speedup, superior goodput, production use, guaranteed p99,
bounded end-to-end latency, starvation freedom, native sampler convergence or
fully validated GPU scheduling. Do not combine CPU test totals across versions.

Deliverable A can be used without a native MPC success claim. Only once
deliverable B passes may a prototype clause be replaced with its actual result
and its evidence ID, whether positive, neutral or negative. Do not insert a
percentage until its numerator, denominator, workload, configuration, command,
trace and exact commit are reviewable.
