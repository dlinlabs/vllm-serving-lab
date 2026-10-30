# PR32 Round7 corrections — implementation revision11

Review: https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6372790392
Parent: `0ea58b9479a3d9d9e6ab330b9c5dc64f5e443623` (revision10).
This is CPU/source review evidence; native acceptance and actual GPU qualification
remain pending. Statistical revision9, workload, five paired blocks, two structural
groups, eligibility threshold, 5ms gain floor and the1800s limit are unchanged.

| Review finding | Correction and evidence |
|---|---|
| Cumulative LOCO crosses trial boundaries | Rows retain trial IDs. Reset residual prefixes per trial, average trial errors within each cell, and retain whole-cell LOCO. Noisy-label regression checks all supported families and verifies duplicating a trial with a new ID does not increase its score. |
| Eligibility uses held release states; freeze hash unchecked | Freeze nominal predictions and comparator eligibility at block3_end using nominal D0/D16 state and completed OFF observers. Archive plan, raw predictions, actual jitter values/definition and source/input identities. Scoring verifies suite model hash, input hashes and training/OFF capture hashes before opening held captures. Actual held release state can condition held predictions but cannot change eligibility. |
| Maximum of approximately790 step CVs causes false stops | Preregister12 cell/phase-total strata: six calibration cells by prefill-bearing/decode-only phase. Verify ordered work/position/mode/padding identity across three repeats before aggregation. Phase CV>5% stops structural claims, >2% flags; per-step maximum is descriptive only. Missing work is inconclusive. This is phase-total repeatability rather than per-step label CV. |
| HTTP bridge bypasses gateway | Amend design and runbook to explicitly scope the exact raw-token bridge to shared client -> OpenAI completions -> EngineCore. The existing gateway is chat-only; bridge does not qualify gateway transport. Azure transfer continues through fixed-cost32 gateway. No baseline gateway or gateway-only restart is claimed. Bridge output records its transport scope. |
| Traffic validation can kill EngineCore; stale epoch dereference | External/duplicate/bad-role checks record invalidity and emit a request-local abort using upstream's existing abort-output path. Do not raise from those ADD validations. Late timer posts are ignored outside their controlled epoch, including during HTTP capture. Regression checks continued valid admission after rejection. |
| GIL may consume the5ms posting allowance | Set EngineCore switch interval to0.5ms once at hook installation and archive the resolved setting. Keep actual posting timestamps and the5ms integrity limit. This mitigation is not evidence of native-host timer fidelity. |
| Recorder observes between trials | Worker finish disables recording immediately, even while deferred CUDA queries remain pending. Detach recorder after all pending events resolve. Regression places unrelated engine work between finish polls and verifies no added events. |
| Decode context/stage terminal states | Use pre-execution decoder position (latest sampled token is unprocessed), and close all unfinished running stages on failure, including interleaved drift references. |
| Source/graph/V2 details | Add five source pins (34 total), guard missing forward contexts, and exclude V2 dummy execute calls. Test absent context and backend metadata. Upstream V2 initializes `attn_groups` in KV setup (`model_runner.py:633`); the claim that V2 has none does not apply to the pinned source. Preserve backend reporting from those initialized groups. |
| Portability and measurement semantics | Normalize generated JSON comparison through text newline decoding; explicitly skip native Linux/Python3.12 gate test on other hosts. Document controlled-request contributions to upstream stats and the pinned private utility API. Jitter is1.4826*MAD per OFF-observer foreground position, assigned by frozen position reference, rather than the held SD assumed by historical noise simulations. |

The source review uses vLLM tag v0.30.0, with the five added file hashes computed
from raw upstream bytes. Hook observations remain unqualified until installed GPU
execution. The previous implementation report and revision9 synthetic noise
artifacts remain historical; they are not rewritten to describe this revision.

## Fixed phase sensitivity

Reproduce:

```bash
python scripts/check_prefill_phase_readiness.py \
  --output results/prefill-round7-fixes-20261023/readiness-sensitivity.json
```

Seed20261023,20,000 draws/profile/CV, three independent lognormal repeat labels,
790 executed step strata from the six synthetic calibration cells. Reported rates
are simulation frequencies, not measured GPU rates or population-CV confidence bounds.

| Input label CV | Historical max-step stop, independent step noise | Fixed phase-total stop, independent step noise | Fixed phase-total stop, shared phase/trial noise |
|---:|---:|---:|---:|
|1%|0%|0%|0%|
|2%|77.955%|0.340%|2.260%|
|5%|100%|61.745%|99.600%|
|8%|100%|94.155%|100%|

Phase aggregation averages independent step noise and therefore also loses some
sensitivity to it. It cannot certify that per-step labels lie within the earlier
1–5% power screen, and it does not accept a physical model. LOCO/model support,
observer fidelity, drift and block dependence remain independent gates; dependence
is still unresolved. No held timing selects strata or thresholds.

## Validation

149 related CPU tests pass, including30 existing collector tests and9 Round7
regressions. Python compilation, generated artifacts and source/manifest checks
pass. A broader working-tree run produced781 passed,13 failed,26 errors (820 cases,
zero skips). Eleven failures and26 setup errors involve unavailable resource
observations/PID-procfs identity. One legacy budget mock timed out; its separate
retry passed. One design metadata-hash failure occurred before final document/
manifest synchronization; it is corrected and checked again in the final related
run. The full result, tracebacks and retry are retained in validation.json and
JUnit evidence rather than rewritten as a green full suite. Native acceptance
remains false. No GPU experiment, measured model acceptance or MPC result is claimed.
