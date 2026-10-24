# PR32 round-2 design checks

CPU-only design revision3. No collector implementation, new GPU measurement, held C use or MPC validation.
All299 public evidence files are verified (161 old-host +138 new-host). Original PR28/30 model sources and learned coefficients remain unchanged.

## ITL mechanism sensitivity

Force historical accepted arrivals and realized output lengths through the serial running-first toy. Each live decoder emits one token per step. Gaps are counted by right-token timestamp in(60,300]; exporter endpoints are the first/last valid scrapes within that range. Different timing and EOS accounting cause different total gap counts; these are not exact per-request GPU measurements.

| Capture | Observed >200ms | B8192 predicted >200ms | B2048 predicted >200ms |
|---|---:|---:|---:|
| cap-range-v1-fixed-32-r1 | 0 | 2255 | 0 |
| cap-range-v1-fixed-32-r2 | 0 | 2334 | 0 |
| cap-range-v1-fixed-16-r1 | 0 | 436 | 0 |
| low-cap-switch-v1-fixed-32-r1 | 0 | 2207 | 0 |
| low-cap-switch-v1-fixed-16-r1 | 0 | 268 | 0 |

The long-gap mechanism supports2048 as a working assumption more directly than pooled TTFT MAE. It does not verify old resolved config or rule out alternative scheduling/observation semantics. Future collection logs the effective budget and long/partial-prefill settings.

At unchanged linear coefficients, k=3.15e-9 overpredicts150–200ms counts (1823/2154/132 versus137/109/0 for old32r1/new32/new16); k=1.5e-9 gives236/165/0. But old16r1 already gets29 predicted versus0 observed at1.5e-9. Thus1.5e-9 is a planning sensitivity, NOT a universal fitted upper bound. Refit linear and nonlinear terms only with new training data.

## Conditional transfer preregistration

Replay[40,120) and[160,240), score[60,120) and[180,240), fixed cap32. The committed counts below freeze the historical new-host fixed32 admission/output path, start empty at the lead-in and score right-token endpoints in(20,80]. They do NOT predict the fresh host's admissions or natural EOS. Forced128-output sensitivities are also archived. The same frozen algorithm must additionally be reported on each future capture's actual path, explicitly retrospective.

| Original score window | B8192 >200ms | B2048 >200ms | B2048 k1.5e-9 150–200ms |
|---|---:|---:|---:|
| [60,120) | 578 | 0 | 26 |
| [180,240) | 503 | 0 | 37 |

Lead-in requests remain in lifecycle audits but outside scored offered denominators. Report queue/running/KV state at the boundary;20s does not prove the historical state was reconstructed. ITL agreement never passes a causal forecasting or controller-benefit gate.

## Design corrections

- Sentinel targets now use sentinel-only TTFT, with every within-D offset difference. Four@800 is a policy tie control; the one-long case now uses7168 tokens, making FIFO/RR differ by about190ms at D16.
- Add the smaller nonlinear sensitivity; single2048 remains validation-only. Named H2/H3 identification contrasts and3x-jitter resolvability are explicit.
- Freeze candidate-versus-baseline eligibility after calibration and before held scoring. Require>=4 eligible cells,>=ceil(.75*n) strict MAE wins, and no-regression on all8. An unresolved comparison is inconclusive, not a pass or a statistical significance claim.
- M1 is selected/scored on decode-only MAE with prefill guards; M2 on both prefill targets with decode guards. Relative MAE reduction is not the same as percentage of total execution time.
- Background256 versus512 preserves the entire foreground execution prefix in all110 cell/scenario combinations (including D0). Matched background survival remains an on-GPU validity check.
- The fixed212 trials plus160s transfer now estimate1317.78s, versus1776.94s previously, under the unchanged1800s hard limit. This is a planning estimate, not an observed runtime guarantee.
- Main, bridge, transfer and deep stages have independent completion/gate records. Full-suite completeness still requires every declared stage; shared observer/config failures invalidate affected stages.

## Verification and reproduction

[Tests](tests.txt), [validation](validation.json), and [all scenario counts/source hashes](itl-scenarios.json). The archived first-round sensitivity result is unchanged; its source hashes identify the earlier revision, not the current helper.

```
python scripts/build_prefill_design.py --output-dir /tmp/new-prefill-plan
python scripts/check_prefill_itl.py --old-evidence OLD --new-evidence NEW --output /tmp/new-itl.json
python -m pytest -q tests/test_prefill_design.py tests/test_cap_miss_causes.py tests/test_greybox_simulator.py tests/test_low_cap_frozen.py
```

Both plan regeneration and independent ITL invocations reproduce their results. Installed collector/hooks, exact prompt manifests, trace slicing, causal rollout and native CPU/GPU acceptance remain future work; this PR is still a draft design.
