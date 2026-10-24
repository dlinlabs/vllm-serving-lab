# PR32 self-review: score boundaries and full-model support

Revision4 reviews commit3409711. There was no new reviewer comment at the time
of this pass. This adds CPU design checks; no new GPU experiment or model fit.

## Findings and corrections

1. **The service-scoring window was underspecified.** Background preparation and
   background-only drain could be included in decode-only MAE and fitting even
   though the preservation claim concerned foreground execution. The toy now
   returns per-request completion times and an explicit release-to-last-foreground-
   completion window. `scored_steps` applies it consistently. Real collector
   wall/GPU labels must use common executed step IDs, not compare GPU timestamps
   with host barriers. Background prep/drain remain in full lifecycle/ITL audits,
   whole-trial observer timing and the time budget.
2. **Rank2 for q/Z did not establish support for M0/M1/M2.** The full feature
   matrix is intercept/decode count/prefill count/decode context/attention pairs.
   `prefill_identifiability.py` checks each family on all calibration rows and
   each of six LOCO training partitions. Row weights balance calibration cells;
   scaling uses that partition's training rows only. Rank, zero columns, column
   norms and normalized condition numbers are archived. The declared condition
   ceiling1e4 is an engineering guard, not parameter confidence. A family without
   support cannot earn a fitted-parameter or held-benefit claim. Real valid
   training observations and each timing-label mask must be checked again.
3. **Invalid toy parameters could avoid normal draining.** Nonfinite nonlinear
   sensitivity and fractional output lengths now fail before simulation.

## CPU results

**99 related offline tests pass.**

- All seven structural partitions (full calibration + six LOCO) have full rank
  for M0/M1/M2. M2 normalized condition numbers range from about7.51 to11.31.
  This is one simulated path for each of six calibration cells, not new timing
  evidence, extra fitting captures, or proof that the model is correct.
- Scored steps/features remain identical under background512 versus256 outputs
  in all22 linear case paths. Existing110 scenario/cell foreground-prefix checks
  and44 frozen-barrier TTFT checks remain covered.
- Historical ITL and transfer scenario results exactly match revision3. Source
  hashes were refreshed, and all299 original evidence files verify. Original
  PR28/30 sources and learned coefficients are unchanged.
- The planned212 trials and160s replay remain unchanged: estimated1317.78s under
  a1800s hard active limit. There is no new request to rent/run a GPU.

[Test log](tests.txt), [validation](validation.json),
[current ITL/source hashes](itl-scenarios.json), and
[complete planning diagnostics](../../experiments/prefill-service-v1.discrimination.json).
Earlier round2 artifacts remain archived at their original source revision.

```bash
python scripts/build_prefill_design.py --output-dir /tmp/fresh-prefill-plan
python scripts/check_prefill_itl.py --old-evidence OLD --new-evidence NEW --output /tmp/fresh-itl.json
python -m pytest -q tests/test_prefill_design.py tests/test_prefill_identifiability.py tests/test_cap_miss_causes.py tests/test_greybox_simulator.py tests/test_low_cap_frozen.py
```

Collector/hooks, exact token manifests, trace slicing, causal rollout and native
collector/GPU acceptance remain pending. No C or MPC benefit is claimed.
