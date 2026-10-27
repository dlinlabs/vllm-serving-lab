# Executable delayed-arrival pilot v2

The next calibration/confirmation operation is documented in [event-clock v1](prefill-event-clock-v1.md). This original four-term protocol is retained for historical reproduction; its first native candidate was rejected.

Status: the first native v2 collection completed, but the candidate failed the
800ms/D16 queue-residence gate. See [the measured diagnosis](../results/prefill-arrivals-39d0c7d-analysis/report.md).
All primary endpoint, observation, work and anchor gates pass. No unchanged-model
rerun is requested; timing-model development can proceed on retained training data.

The source-bound JSON protocol has16 cases and136 trials, fixed RTX4090/Qwen
revision/vLLM0.30.0/Torch2.13.0+cu130/B2048/seq32/context8192, same-process
V2 UniProcExecutor with two async slots and all existing38 source pins. No new
GPU wrapper, scheduler override, per-step synchronization or cap policy is added.

Order on one loaded engine:

1. 16 unscored LIGHT warmups, covering all workload shapes. Their labels never
   enter fitting or parameter selection.
2. 48 training trials: original six zero-offset cells, four OFF/LIGHT pairs each.
3. Fresh same-host four-parameter fit, unchanged v1 basis/optimizer/quality gates;
   write model/costs and all forecasts before any scored held/anchor trial.
4. Four anchor-before trials: mixed D0/D16, one OFF/LIGHT pair each.
5. 64 delayed trials: eight new mixed-plus-sentinel cases, four pairs each.
6. Four anchor-after trials with EXACTLY the same token content as anchor-before.

The main mixed foreground is [1024,1024,7168,7168], outputs128 each. A512-token,
128-output sentinel joins at50/200/400/800ms with D0/D16 backgrounds. Backgrounds
remain1024-token prompts/256 outputs and must actually outlive foregrounds.
No automatic extension, retry or cross-run stitching. New fifth-request joint
states must pass native work/dispatcher checks AND own-rollout work equality;
matching marginal lengths does not establish support.

## Clocks and predeclared gates

Primary labels are **declared target-to-first and target-to-final output**.
The simulator consumes declared arrival offsets, not measured future enqueue
states or observed work. Extracted diagnostics separately retain actual posted
TTFT, posting lag, and actual queue-post-to-engine-enqueue residence. Immediate
foreground posting must equal the release barrier; sentinel target must equal
release+offset and posting must not precede it or exceed5ms lag. Long queue
residence is an outcome and does not relax the5ms posting limit.

Training retains v1 max(5ms,20%) TTFT / max(10ms,20%) completion fit rules,
local sensitivity rank4/condition<=1000, both optimizer starts agreeing within5%,
no bound hits and no held fitting. Matched observer differences stay within
max(2ms,10% OFF endpoint); OFF repeat ranges within max(5ms TTFT /10ms completion,
20% median). These unchanged rules also apply to held endpoints.

Every delayed foreground/sentinel endpoint is evaluated by its four-OFF-repeat
median. Each anchor phase is scored independently to prevent before/after
averaging from hiding drift. Sentinel queue-residence median predictions must
also be within max(10ms,20% measured residence) in every delayed case. This
additional gate is explicitly secondary: forecasts assume ideal on-time posting,
while measurements use actual posting; up to5ms allowed posting lag remains
visible and the10ms queue floor is fixed before measurement. Signed endpoint
errors, all OFF sentinel clock diagnostics and separate anchor scores are saved.
No tail/p99 guarantee is made from four repeats.

Failed native integrity/chronology, source/runtime, posting or training gates stop
and retain partial evidence. Held prediction/queue/structure discrepancies retain
all completed captures and return a failing candidate after scoring; they cannot
turn into acceptance by refitting. Complete but failed candidates exit2, incomplete
runs exit nonzero. All physical/full-model/complete-suite/cap/MPC flags stay false,
even if candidate_passed is true. D16 bias from v1 remains a known limitation;
no extra feature is introduced or tuned on its held results here.

## Run only after exact-source preparation

Fresh native Linux/Python3.12 RTX4090 host, PR32 checkout:

```bash
nohup python3 scripts/setup_prefill_calibration.py --next arrivals > ~/pr32-prepare.log 2>&1 < /dev/null &
tail -f ~/pr32-prepare.log
```

After its final PREPARED line, exit tail with Ctrl+C and run:

```bash
nohup .venv/bin/python scripts/prefill_host.py arrivals > ~/pr32-arrivals.log 2>&1 < /dev/null &
tail -f ~/pr32-arrivals.log
```

An installed host first runs prefill_host.py prepare for this NEW exact revision.
Archived v1 coefficients are never imported into native v2 collection. A dedicated
arrival-tokens.json and mutually exclusive endpoint mode keep v1 calibrate and
normal collect/diagnose/stages isolated. Every normal launch explicitly resets
both calibration mode environment flags. No gateway/Azure/deep replay runs.

The active bound is1800s including fitting/analysis, after the separate900s engine
startup bound. Installation/download/native CPU acceptance are separate. There is
no measured v2 runtime estimate. Progress prints trial/total, phase, case, observer.
Outputs and original native acceptance/JUnit/log go to ~/pr32-arrivals/<timestamp>/;
a sibling .tar.gz is packed even if collection fails. Preserve it before returning
the machine. No upload or rerun occurs automatically.

Independent audit on the exact v2 source:

```bash
.venv/bin/python scripts/check_prefill_arrivals.py audit /root/pr32-arrivals/TIMESTAMP
```

It re-extracts all raw clocks/endpoints and work, checks token/model/source/native
JUnit/runtime identities, frozen training hash and regenerated forecasts, then
recomputes the stored verdict. Hashes establish artifact consistency, not signed
attestation of collection order. Offline audit does not rerun the native GPU.

CPU smoke/recovery with saved metadata only:

```bash
python scripts/check_prefill_arrivals.py demo --runtime /path/to/runtime.json --output /tmp/new-arrival-demo
```

The demo is synthetic and never earns native acceptance. Next after successful
native delayed qualification: HTTP/state observability, fixed versus adaptive cap,
then MPC. New evidence may reject the reduced model; preserve that outcome.
