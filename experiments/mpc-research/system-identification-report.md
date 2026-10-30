# Development identification results and unresolved causal questions

Source freeze: `8e92371`. This report contains an executed reanalysis of
historical development captures, not a new native GPU/HTTP experiment.
No reserved final trace C was used for model selection.

## What was executed

Mayor restored the cap-range and low-cap archives with their existing verified
extractors, then independently reran the Opus Round1 diagnostic script under
CPython 3.12.16. Both archive transports and restored bytes passed the extractors'
hash/inventory checks. The low-cap source archive is
`cae6d6934a293980bc688b7397c683c46366efd73bcc50a5245783b5ed0e3205`.
The original extractors and their manifests remain the authoritative inventory.

The preserved script is
`experiments/mpc-research/proposals/opus-round1-analysis.py`;
the complete numerical output is
`results/mpc-research-cpu/development-backlog-reanalysis.json`.
Its CLI takes fresh cap-range and low-cap extraction roots, as documented in the
script. It reads no new model outputs, controller actions or future held workload.

At each archived admission, the script reconstructs earlier admitted requests
not yet terminal. It sums prompt estimates of those without observed first
content, and separately retains streaming count and reserved cost. This is a
gateway-internal information experiment. The existing external `/metrics` does
not expose the complete marked state at that admission instant.

The target is client scheduled-arrival TTFT. Do not reinterpret its fitted
coefficient as a measured GPU kernel service rate or exact engine queue drain.

## Reproduced factual prediction

Twelve leave-one-run-out folds exclude the already load-fidelity-invalid
cap16-r2 capture, using the pre-existing rejection reason. That capture remains
in the output's descriptive summary; it was not erased after seeing the fit.

| Factual predictor | Minimum / median / maximum held-run AUC | Minimum / maximum TTFT MAE |
|---|---|---|
| Reserved cost plus own prompt | 0.5858 / 0.6852 / 0.7750 | 0.0978 / 0.3587 s |
| Awaiting-first prompt backlog plus own prompt | 0.9590 / 0.9763 / 0.9949 | 0.0334 / 0.0932 s |
| Causal backlog/count/age/receipt-lag features | 0.9664 / 0.9814 / 0.9981 | 0.0284 / 0.0891 s |

These are correlated development folds of the same offered trace, not twelve
independent workloads or confidence bounds. Feature exploration occurred on
previously inspected data. Strong classification of factual outcomes neither
establishes a policy's counterfactual action ordering nor licenses a new cap.
No conformal coverage, online adaptation or native controller timing was
validated by this calculation.

A separate direct check of successful `client.json` rows found the unique
successful output length to be exactly `[128]` in every restored capture,
including the archived smoke runs. Thus the available evidence does not exercise
variable natural-EOS lengths, even where requests did not explicitly force EOS.
The proposed residual-decode/survival model remains unidentified outside that
degenerate development distribution.

## Consequences for design

1. Pending prompt backlog deserves an online-observer implementation and a
   fresh-session ablation against reserved cost alone. It does not deserve an
   unconditional FCFS or deterministic latency claim.
2. The regression using prompts admitted after the target request first enters
   the system is outcome-window-selected. Its small coefficient cannot identify
   the causal effect of future admissions. It is a diagnostic, not proof that
   lookahead has no value.
3. Maximum observed fixed-cap admission throughput is not a mathematical upper
   bound on all possible policies. A genuine bound needs a validated service/
   workload argument; the historical 7.4 req/s observation supplies none.
4. A mean number of releases in a 50ms interval is not a pathwise upper bound on
   reservation released during that interval. A slack controller needs exact
   accounting plus measured age/delay uncertainty, not that replacement.
5. Session differences and missing historical scheduler provenance prevent a
   physical transfer claim to the current B2048 native protocol.

## Executed mathematical falsification

Run:

```bash
python3 experiments/mpc-research/mayor_counterexamples.py
```

Eight deterministic checks passed and their exact output is preserved in
`results/mpc-research-cpu/mayor-counterexamples.json`. They establish:
holding a valid cap does not guarantee a hard terminal-drain constraint;
expected release is not a hard release bound; a timed-out command can outlive
a subsequent old-state read; omitting existing cohorts can reverse action
preference; first-token success is not eventual stream success; rolling sums
need expiration memory; identical perfect factual fits permit opposite causal
action effects; and absent KV information cannot certify a 0.1% chance constraint.

These small counterexamples reject particular arguments. They are not a measured
serving-performance advantage for any competing controller.

## Remaining identification gates

The next evidence must come from the causal, clock/version-qualified observer,
randomized supported cap transitions, realistic variable EOS, load-crossing and
burst regimes, and whole-episode-separated calibration/validation. See
`system-identification-plan.md` and `benchmark-protocol.md`.
Native state/action qualification, HTTP qualification and MPC policy acceptance
all remain false.
