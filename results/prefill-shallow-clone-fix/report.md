# Shallow-clone bootstrap repair

The rental log for 391ef1b reports 1324 passed and 2 failed CPU cases. The two
failures are historical-source rejection unit tests trying to resolve 8bd96cb
in a depth-one checkout. Setup stopped before GPU collection; this is not an
accepted native CPU run or a predictor measurement. The provided shallow-clone
bootstrap omitted this test prerequisite.

The source-rejection unit tests now use explicit source-byte fixtures and match
the intended rejection messages. Additional regressions create a real local
depth-one clone with a genuinely absent parent, then verify exact source
success, inventory/hash corruption rejection, missing-parent rejection and an
unavailable-blob failure. Tests do not download history or skip guards.

Actual old-capture reproduction still requires the exact pinned 8bd96cb tree.
The analyzer reports an actionable prerequisite error when Git cannot supply
that source, with an explicit fetch command. It never substitutes HEAD or
fetches silently. Its inventory/hash/scope checks remain unchanged. All 179
files of the original native snapshot were checked again successfully.

115 related local CPU cases pass (11.95s), including the repaired and real Git
tests, queue confirmation/audit/host/entry/receipt regressions and committed
design/hash regeneration. This selection overlaps earlier checks and is not
added to their count. Full native acceptance must run again on the rental host
for the exact patched checkout. No new GPU run or model/cap/MPC acceptance is
claimed.

The rental machine can keep installed packages and pinned model files. Fetch
the patch, then rerun prefill_host.py prepare and queue-confirmation. A failed
acceptance is never reused.
