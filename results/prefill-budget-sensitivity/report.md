# Frozen-budget sensitivity before the next GPU experiment

CPU-only development diagnosis responding to PR32 review item1. Original PR28/30
model sources, service coefficients, empirical TTFT arrays and5.457ms reference
delay are unchanged. Each B scenario recomputes only the original causal online
bias from closed historical simulations; no learned parameter is refitted.

The archived seven-capture suite re-audits all six gates; all119 capture-file
hashes match. Six full captures yield the same288 non-overlapping five-second
windows.8192 forecasts match the archived PR30 predictions within1e-10. Two
independent evaluation invocations produce identical core results. Tests and
provenance are in [validation.json](validation.json).

| Scheduled-token assumption | Goodput MAE | Cap32 MAE | Cap32 bias | MAE without online bias |
|---|---:|---:|---:|---:|
|8192|0.72896|0.94670|+0.31628|0.76156|
|2048|0.74231|0.97196|+0.38157|0.78405|

Changing B alone does not repair these forecasts. This does not reject2048 as the
actual historical scheduler setting: pending prompts restart from full input at
forecast origins, decode progress is approximate, and the old counter-based step
fit/empirical correction remain unchanged. Numeric forecast error is not a config
identity test. The new experiment pins the serving2048 setting and records actual
work/config rather than choosing a budget from these outcomes.

## Short-miss association is not unique evidence for2048

Force the actual admitted arrivals through an empty-start raw serial service
scenario from t0, with fixed128-token outputs, no empirical TTFT correction and no
admission decisions. Compare successful short prompts from60–300s in the new-host
fixed32 run. Actual outcomes remain available only for retrospective scoring.

| Preceding-second prompt-load quartile | Actual miss |8192 simulated miss|2048 simulated miss|
|---|---:|---:|---:|
|Q1|0.83%|0.42%|0%|
|Q2|3.75%|6.25%|0%|
|Q3|15.42%|17.50%|12.50%|
|Q4|63.33%|52.50%|59.17%|

Each quartile has240 requests; cuts2662.75/6499/11590 input tokens. Both budgets
show an increasing high-load miss association.2048 is closer in Q4 but worse in
some other quartiles. This forced-admission, full-prompt, empty-start diagnostic
is not an online forecast, exclusive mechanism test, or a causal cap comparison.

## ITL histogram check

All five review rows reproduce exactly from the first/last valid exporter samples
within60–300s. Counts are not independent GPU-step timings.

| Capture | ITLs |100–150ms|150–200ms|Above200ms|
|---|---:|---:|---:|---:|
|Old fixed32 r1|178038|6269|137|0|
|Old fixed32 r2|173333|6616|88|0|
|Old fixed16 r1|129485|888|0|0|
|New fixed32|159572|5594|109|0|
|New fixed16|136146|522|0|0|

Upstream v0.30.0 has distinct API-server2048 versus LLM-class8192 defaults on the
smaller-GPU branch; the reference blob is recorded in validation. Old installed
wheel/config/override identity remains unverified. A new-host short replay may
compare ITL tails, but lack of gaps above0.2s is not an acceptance gate that would
force a desired result. Actual per-step scheduler/execution records are required.

## Reproduce

Restore old/new public bundles using PR31's instructions. Python3.12 and the CPU
analysis dependencies from PR30 suffice:

```
python scripts/check_prefill_budget.py \
  --suite-dir NEW/results/low-cap-switch-v1 \
  --old-suite-dir OLD/results/cap-range-v1 \
  --output-dir /tmp/fresh-budget-sensitivity
python scripts/build_prefill_design.py --output-dir /tmp/fresh-design
python -m pytest -q tests/test_prefill_design.py tests/test_cap_miss_causes.py \
  tests/test_greybox_simulator.py tests/test_low_cap_frozen.py
```

Choose fresh output paths. [Summary](summary.json), compressed full predictions,
source/evidence hashes and [test results](tests.txt) are preserved. This is not new
native Linux collector acceptance, GPU measurement or MPC validation.
