# Resume bullets: verified prototype wording

Use **one** pair, not all six bullets. Current scope is a research prototype,
not a completed native closed-loop MPC deployment. Evidence IDs refer to
[benchmark results](benchmark_results.md).

## AI Inference / Infrastructure

- Built an instrumented vLLM admission-control gateway and trace-replay
  pipeline; audited native serving captures replaying 7,643 offered requests
  over 300 seconds, retaining overload rejections and control-timing evidence.
- Implemented CPU-validated MPC planning components with versioned-state,
  deadline and failure checks, documenting the remaining native integration
  and strong-baseline comparison gaps.

Evidence: first bullet E1, including raw trace, collection configuration,
reproduction command and publication/runtime commits; second bullet E2/E3/E5.
The request count describes each historical identification capture, not MPC
throughput or a production traffic claim.

## Autonomous Systems / Runtime

- Designed a finite-horizon admission-control prototype with coherent scenario
  evaluation, first-action replanning interfaces and guarded command-state
  handling; validated planning components on CPU.
- Built auditable observation, lifecycle and failure evidence for native LLM
  serving, distinguishing stale state, unknown application and incomplete
  cleanup from successful control.

Evidence: E2/E3/E4/E5 and the [architecture](architecture.md). First-action
interfaces are not a claim that the complete native loop has been demonstrated.

## General Software / Distributed Systems

- Implemented an asynchronous inference gateway with reservation accounting,
  overload handling and lifecycle telemetry for reproducible serving studies.
- Developed source-pinned replay and independent validation workflows linking
  request outcomes, configurations and commands to Git revisions, preserving
  negative results and actionable runtime failure cases.

Evidence: E1/E3/E4/E5. Do not substitute "production deployed" for "implemented".

## Claims explicitly excluded

Do not claim MPC speedup, superior goodput, production use, guaranteed p99,
bounded end-to-end latency, starvation freedom, native sampler convergence or
fully validated GPU scheduling. Do not combine CPU test totals across versions.

Once P0 closes, replace a prototype clause with the actual comparable result
and its evidence ID, whether positive, neutral or negative. Do not insert a
percentage until its numerator, denominator, workload, configuration, command,
trace and exact commit are reviewable.
