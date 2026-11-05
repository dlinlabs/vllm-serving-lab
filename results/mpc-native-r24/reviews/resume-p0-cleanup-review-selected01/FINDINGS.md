# Independent D2-F1 cleanup-repair review: Role7

**Model:** claude-opus-5.5 at max (verified). No child agents. CPU only. No git, SSH, GPU, network, main(), valid approval, installs, actual arrays, private exports or old-suite reruns.

**Subject:** `.runtime/r24-team16-native-cut-consumer-cleanup-successor01`
- SHA256SUMS `f145546d68934af0b0ec78bee8a78bd4817b53ca694edea8ef91df2731d023f5`, 136 entries (137 files including the seal). Checked at start, during relocation and at seal time; unchanged and untouched.
- Author: Role6. The author's model is not recorded in the viewed REPORT/PLAN, and I did not check it.

## Verdict (bounded, conditional)

The D2-F1 repair is effective at the source level for every defined trigger: T1, a stale monitor alarm; T2, a failed anchor publish or `c.domain`; T3, an exhausted or raising `arm`.
- Every exit path from the try at L405, which is where owned processes can first exist, attempts the owned cleanup and writes the identity report before the unlock, except one narrow residual case (R1, LOW).
- The author's two cleanup controls PASS when relocated and run independently, but only on **py312**.
- **py314 was not reproduced independently.** My own harness missed a precondition, and under the no-retry rule I did not run it again.
- Every py312 invariant matches the author's py314 results.

This is a narrow CPU result on a single host. It is not proof of OS-wide cleanup and not G06, Native, real-loop or component completion.

Result: **ELIGIBLE for Mayor integration review as a CPU-validated narrowing of D2-F1**, subject to the conditions in the last section.

## Pins

| Item | sha256 / value |
|---|---|
| Original `launch01.py` | 21eb1939… (21042 B) |
| Candidate `launch01.py` | 9a7dd920… (29870 B) |
| Original `owned01.py` | 8e2e2da2… (10672 B) |
| Candidate `owned01.py` | 5c4e2c06… (11552 B) |
| `EXACT_SOURCE_DIFF.patch` | 8b6d4ef3…; applied by machine to the hash-read originals, it reproduces both candidates byte for byte (`work/`) |
| Author control freeze | 14e87631a078dc8d315eaf5cb7dbc4f74b545460abc63a2b2e764a3687c87262; my py312 run reported the same value |
| Deployment02 seal | 407bc2e8… = `inputs/deployment/SHA256SUMS` |
| d2r01 seal | b13e5b29… = `inputs/review/SHA256SUMS` |

Full hashes are in `findings.json`, `HASHREAD.txt`, `SELECTION.json` and `RELOCATION.json`.

## Checklist (candidate line numbers)

| Requirement | Result | Source |
|---|---|---|
| Every monitor exit disarms | YES | `monitor` body is in try/finally with `setitimer(0)` (L76/L106-107), covering returns L97/L99/L101/L103 and raises |
| First statement of the outer finally disarms | YES | L455, then `cleanup_owned_stage` at L456-457. The stage disarms on entry (L204), after publish (L237) and in its own finally (L250). The inner finally disarms at L511 before `release_after_cleanup` (L512) |
| Anchor, clock or arming errors cannot bypass owned cleanup | YES, except R1 | Anchor validation is in try/except BaseException (L212-226). An invalid anchor or `initial_unset` falls back to the true current time. A valid exhausted anchor is kept and recorded (L229-230), never reset. Publish and `c.domain` are wrapped (L231-235). The stage never calls `arm()`. `owned.cleanup` is in try/except/finally (L238-252). The monitor's `arm` raise is disarmed in its own finally and caught by main's except (L451) |
| A failed root lifecycle write cannot stop the remaining wait, join or fd close | YES | `owned01` L213-218 is wrapped. The remaining roots' wait (L200), reader join (L203-204) and fd close (L207-209) still run, as do leaf fd close (L225-227) and `capture_complete=not errors` (L237) |
| Original bounds kept | YES | cleanup deadline = min(anchor+40, started+243) (L227; owned L168). IO deadline = min(started+260, finished+17) (L278). Allocation 7/30/2/1 unchanged, with `require(reap==30)` (owned L193) |
| No additional grace | YES (my interpretation) | Past the deadline, `reap_owned` makes one SIGKILL + WNOHANG pass with no sleep (L115-118, L148-151). Waits use `max(0,…)` (L200), so post-deadline waiting is 0. The result is classified as out of budget (L275-276) and never counted as success |
| Unknown/incomplete fates when not joined | YES | `incomplete_all_offered` runs under the IO deadline (L279-286): status INCOMPLETE, every planned ID `unobserved`/`fate_known=False`, `missing_receipt_not_proof_of_no_send`, client result not read. DO_NOT_EXPORT is written under the IO deadline (L287-294) |
| Private key kept on non-join | YES | `close_private_config` refuses first, at the join guard (L351), before touching CONFIG. `publish_outputs` returns False. No RETURNED receipt is substituted (`persist_close` runs only in main's joined postflight) |
| Unlock order | YES | `release_after_cleanup` requires cleanup finished/attempted and `remaining_owned` present (L311-317) |

## Findings

### R1 (LOW, source/documented semantics only; not executed)

There is a narrow window at the start of teardown where a late SIGALRM can still skip cleanup.
- The startup timer `arm(started+180)` (L394) stays armed throughout the pre-monitor try body. `alarm` raises TimeoutError (L19-20).
- CPython runs Python signal handlers only at bytecode/eval-breaker points in the main thread.
- Suppose the timer expires after the main thread's last eval-breaker check in main's except body (L452-453) and before the outer finally's `setitimer(0)` (L455) takes effect.
- The handler then runs at the next check. In CPython that is normally right after the L455 call returns. If the signal is delivered through another thread, it may run a few bytecodes later (L456, or L204-L211 before the stage's first try at L212).
- In that case TimeoutError escapes the finally and `cleanup_owned_stage` never runs. This is the original D2-F1 class of effect: orphaned own-session processes, the lock released at exit, and no `all-offered.json`, receipt or export (still fail-closed).
- If the handler runs even later, inside the stage's wrapped regions, the error is caught and recorded instead.

How serious it is:
- It needs a non-alarm exception from the pre-monitor try body to coincide, within a window of a few bytecodes, with the 180 s expiry.
- The lateness refusals at L430/L443 test the same started+180 boundary, so the two are correlated rather than independent. The coincidence still has to fall within microseconds.
- The monitor path is safe: a race there raises inside monitor's finally and is caught at L451.
- It is a strict narrowing of the original, whose outer finally started without any disarm (diff hunk @@ -292).
- None of the controls exercise it. I added no control because none was authorised.

Hardening (for Role6/Mayor; not implemented):
- Make `alarm` non-raising during teardown. Preferred: a call-free module flag assignment as the first statement of the outer finally, checked by `alarm`, re-enabled only around the postflight `arm(io_deadline)` block (L466-492). Weaker: absorb one TimeoutError around the disarm preamble. That does not cover a handler that runs later because the signal was delivered through another thread.
- **Caveat (CPython implementation detail, not verified here):** `signal.signal()` processes pending signals before it changes the handler. So "install SIG_IGN first" on its own may not close the window.

### I1 (INFO, pre-existing, not part of this repair)

These limitations remain:
- A non-joined run still releases the lock at L512 or at process exit; flock cannot outlive the launcher.
- The spent-namespace check (L385-386) blocks this launcher from reusing the host, but not other lock users.
- There is no PDEATHSIG or subreaper, so an uncatchable launcher death (SIGKILL/OOM) orphans the own-session services.
- D-state or unkillable processes stay unproven.

### I2 (INFO)

In `owned01`'s per-root loop, only lifecycle-write failures are isolated. A non-TimeoutExpired raise from `process.wait`, `stdout.close`, `os.close` or `Reader.join` would still skip the remaining roots. It is caught at L247-248, so the run ends not-joined, with DO_NOT_EXPORT and an incomplete all-offered, and is still fail-closed. The SIGKILL first pass has already run by then. Leaf fds may stay open until exit.

### I3 (INFO, pre-existing, fail-closed)

- In a joined run where the postflight fails before L469, there is no `all-offered.json`. Raw and refusal files are kept.
- If `c.domain` fails before the try (L399), the namespace (created at L388-389) is spent without a receipt. No owned processes exist at that point.
- The DO_NOT_EXPORT fallback (L507-509) has no IO-deadline check.
- The IO timer armed at L467 stays armed through L494-L510 and is only disarmed at L511. If it expires there, postflight publication is aborted after the owned processes have already joined. This is outside D2-F1's cleanup scope and I did not analyse it further.

## Controls

### Discrimination (frozen in `CONTROL-DECISION.json` before execution)

- **T01:** The monitor raise/return disarm assertions fail on the old monitor, which did not disarm on exit. They test the same function name, so this is semantic discrimination.
- **T02:** Exhausted anchor with `arm` prohibited, `c.domain` raising EIO, and a held reader. Expected: SIGKILL on the first pass, INCOMPLETE fates, DO_NOT_EXPORT, close refused.
- `cleanup_owned_stage` does not exist in the old code, so S2-S4 are separated from the old code only by reasoning; executing them against the old code would need an adapter, which was not authorised.
- The controls call components directly and never `main()`. I verified the outer-finally wiring by reading the source only.

### Execution

The 37 relevant subject files were copied byte for byte, with every hash verified, into `relocated/` (352,291 B). The command was exactly the author's argv, run with `cwd=relocated` under an outer `timeout -k 5 245`.

| Run | Result |
|---|---|
| py312 | exit 0. T01 PASS (0.989 s), T02 PASS (0.037 s); `harness-safety` `all_owned_gone`, no safety signals. In-run, the 5 dummy identities were gone ≈1 s after start, well before their 20 s self-exit. |
| py312 sequential postcheck | 6 identities (5 dummies + runner) gone; no scope processes. |
| py314 attempt 1 | **FAIL (reviewer harness).** `run_controls01.py:26` `FileNotFoundError`: `relocated/attempts/` was missing because my relocation copied only regular files. `load()` had already passed, so the author's own hash checks accepted every relocated input and in-place dependency. 0 controls ran, 0 processes spawned. Preserved; **not retried** (`AMENDMENT-01.json`). |
| Reviewer harness error | Both "postcheck" steps were launched in the same parallel batch as the controls and finished before them, so they are **not** post-run evidence. They are preserved and disclosed in `AMENDMENT-02.json`; one sequential rerun of the unchanged read-only checker followed. |

The author's py314 T01/T02 results (`control-result.json`, viewed as reserve files R-b/R-c) match my py312 run on every invariant:
- T01: errors are ENOSPC plus 2×EIO; anchor preserved, deadline−anchor = 40; reap window 37; all identities `old_identity_gone`.
- T02: errors are `exhausted_no_reset` plus EIO clock; not joined; 2 identities `same_identity_present` at report time (signalled, after one kill pass past the deadline with no wait); signals [15, 9, 9]; INCOMPLETE with 2×`unobserved`; IO−hard = 17; `arm_calls=[]`. The harness-safety step then found both gone without sending any signal.

## Accounting

| Category | Contents |
|---|---|
| VIEWED | 14 subject content files, 145,545 B (≤512 KiB; within the frozen bound of 148,919): 12 fixed per `SELECTION.json` (`REVIEW.md` only partly viewed but counted in full) plus R-b and R-c. R-a (`prepare01.py`) and R-d were not viewed. |
| METADATA ONLY | Manifest names and sizes; keys/paths extracted from the freeze and INPUTS files; count-only greps. |
| HASHREAD | `HASHREAD.txt` |
| COPIED NOT VIEWED | `RELOCATION.json` |
| OWN OUTPUTS | `runs/`, `relocated/attempts/` (not counted against the budget) |

No full 136-entry or 479-entry rehash. No old 42/9/7 tests rerun.

## Assumptions and limits

- One Linux host and kernel; CPython 3.12.16 only for the independent run.
- Synthetic anchors (-30/-8, -244/-41) and a 0.25 s timer stand in for the real 180/243/260 budgets.
- Injected errno failures are representative, not exhaustive.
- Process identity is pidfd plus /proc start-ticks within the same PID/time namespace.
- Only SIGALRM from the launcher's own ITIMER_REAL is considered. An external same-uid SIGALRM or SIGKILL is outside the threat model.
- No universal OS or ownership proof.

## Handoff: what makes this source eligible

1. The Mayor decides on R1: accept it as a documented residual, or require a surgical successor that makes `alarm` non-raising during teardown.
2. Mayor/Parent byte and inventory acceptance of the 2-file overlay (I did no Parent inventory).
3. Any Native, G06, loop or composition use needs separate Mayor approval, a freeze and a preregistration.
