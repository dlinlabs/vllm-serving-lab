# deployment02: independent source-only disposition (d2r01)

- Reviewer: Role7, claude-opus-5.5, reasoning effort max, no nested or child agents.
- Subject: `.runtime/r24-team16-native-cut-consumer-deployment02`
  - SHA256SUMS sha256 `407bc2e88426f4997064fd7f62d09651af74a2c18c6e4989267f48a601a16ae8`
  - 479 sealed entries, 480 regular files including the seal, plus 20 unsealed absolute symlinks.
- Reported but not read or repeated here: the Mayor engineering acceptance (c00d29d2296bda7e249578f2de7c08650a4d79c9) and the Parent index.
- Machine-readable report: `report.json`, built deterministically by `make_report_d2r01.py`.
- Verifier: `verify_d2r01.py`.

## Disposition: CONDITIONAL_SOURCE_BLOCKER_D2F1_NO_FALSE_PASS_FOUND

I traced the default-OFF route end to end:

1. launcher
2. two services
3. client
4. consumer worker
5. RETURNED receipt
6. owned cleanup
7. safe export
8. private-config close

Within this selection and under premises P01–P08:

- I found no defect of these kinds: false pass, relabelling, budget broadening, or approval bypass.
- One **medium source blocker**, D2-F1, must be resolved before the single live attempt. It is inherited unchanged from the reference composition and is reachable live only now, through the new activation path.

This is not an acceptance and not a proof:

- Ownership and OS-failure safety is not universally established.
- G06 is not closed, G16 is pending, and Native14 is FAIL (unchanged). No gate status changes.
- No Native, GPU, SSH, network, git, approval, live-main, source-patch or nested-agent action was taken or authorized.

## Budget use

| Item | Used | Limit |
|---|---|---|
| Selected subject files | 39 (36 first selection + 3 post-freeze additions) | 40 |
| Selected bytes | 396,016 | 4 MiB |
| Post-freeze additions | 3 | 4 |
| New controls per Python | **0** | 4 |

Each selected file was pinned against the subject manifest before its content was read; see `PINS.json` and `POST-FREEZE-ADDITIONS*.json`.

I ran no controls. Every finding is visible in the source. Demonstrating D2-F1 live would need a valid or fake approval, a monkeypatched gate, or a call to the live `main()`, and all of these are prohibited. A stand-alone alarm witness would only restate Python's signal semantics.

## D2-F1 (medium): a stale monitor alarm or a failed cleanup-anchor publish skips `owned.cleanup`

### Code facts (launch01.py unless noted)

**Alarm and monitor**
- 18–24: the SIGALRM handler raises `TimeoutError`. `arm()` uses a one-shot `ITIMER_REAL`, and a later `setitimer` replaces a pending timer.
- 74–103: `monitor()` re-arms the timer on every ~50 ms loop. Line 102 sets it to `min(started+203, deadline)`, where the deadline at 90 is `active+23` or `started+180`.
- `monitor()` returns at 95, 97, 99 and 101 **without disarming** the timer.

**Teardown in the outer finally**
- Main's outer `finally` spans 294–360. Lines 295–296 (and 309–313) sit outside any inner `try`, so an exception there skips 349–360.
- The inner `try` does three things in order:
  1. Publishes `host-cleanup-anchor02.json` (298–299: exclusive partial file, fsync, rename).
  2. Re-arms the timer (300).
  3. Calls `owned.cleanup` (301–303).
- `except BaseException` (307–308) records `cleanup:<type>` and leaves `cleanup=None`.

**When cleanup is None**
- `joined` is False. Line 316 refuses before `all-offered.json` is written at 317.
- `publish_outputs` writes `supervision.json` plus an export gate with `safe_to_export=false`, and keeps the private config.
- 358–360 disarm the timer and release the flock.

**Process ownership**
- Owned roots use `start_new_session=True` and have no parent-death signal (owned01.py 66–83).
- The subreaper role is only in-process.

### Triggers

- **T1 (race, narrow, but correlated with the busy regime).** The client's final phase is bounded by `min(active+23, end+3)` (client01.py 655). When it finishes within about one 50 ms poll before `active+23`:
  - Monitor returns `normal=True` while the alarm from line 102 is still pending.
  - If the alarm fires during 297–300, cleanup is skipped.
  - If it fires at 295–296, `TimeoutError` escapes `main`, so `supervision.json`, the export gate and `all-offered.json` are all missing.
  - A non-alarm monitor raise just before the pending alarm has the same exposure.
- **T2 (deterministic under an OS fault).** Any OSError from the anchor publish (ENOSPC, EDQUOT, EIO) or from the `c.domain()` procfs reads skips cleanup. `exclusive_marker` is not a realistic cause, because only the launcher publishes this file.
- **T3 (narrow).** The alarm fires inside the outer `except` body (291–293), so the initial `anchor=started` survives. Then `arm(started+40)` raises `phase_deadline_already_exhausted`, cleanup is skipped, and `TimeoutError` propagates.

### Effects

- **Orphaned processes.** Native vLLM, Gateway, the client and the consumer worker survive the launcher in their own sessions.
  - The descendant identities existed only in launcher memory.
  - What remains on disk: `PUBLIC/{native,gateway,client-process}/started.json` and `consumer-registered02.json`.
- **Lock released while the GPU may be busy.** The flock is released at 358–360, or at process exit.
  - A later launch of this design refuses at its own idle and fresh-host checks.
  - Other lock users must rely on their own idle checks.
- **Lost or retained outputs.**
  - `all-offered.json` is missing.
  - The private config holding the admin key is retained.
- **Reporting stays truthful.** `cleanup=null`, `owned_joined=false`, `DO_NOT_EXPORT`, and `component_pass=false`. There is no false pass.

Author coverage: controls C08 and C09 cover only the normal join and the config close. There is no stale-alarm control and no anchor-failure control. LIMITATIONS disclaims universal OS-failure safety, and REMAINING_PREREGISTRATION has no manual reap procedure.

### Resolution: either R-a or R-b before the single live attempt

**R-a: surgical successor**
- **S1.** Disarm the timer on every monitor exit, and as the first statement of the outer `finally`.
- **S2.** Isolate the anchor publish in its own `try`, so its failure is recorded but never skips cleanup.
- **S3.** Whenever `owned` is set, always attempt `owned.cleanup` under a valid bound no later than `started+243`. If the recorded anchor is exhausted, recompute it from now.
- **S4.** When not joined, write an explicitly incomplete `all-offered` snapshot. DO_NOT_EXPORT stays unchanged.
- Add one focused CPU control per Python that injects a pending alarm and an anchor-publish OSError. All owned helpers must still join.
- Budgets 260/180/20/3/40/17 and teardown 7/30/2/1 stay unchanged.

**R-b: explicit Mayor risk acceptance plus a preregistered manual reap procedure**

Apply this whenever `supervision.json` is missing, or shows `cleanup=null` or `owned_joined=false`:
1. Read the PID, start ticks and process group from the `started.json` files and `consumer-registered02.json`.
2. Verify the start ticks.
3. SIGKILL the verified process groups.
4. Confirm there are no GPU compute apps and no listeners on 8000 or 8080 before the lock is reused.
5. Classify the run as a supervision failure and preserve every output.

## Low notes and notes (fail-closed unless stated)

| ID | Summary |
|---|---|
| N01 | `fresh_host` runs after the one-attempt namespace and the lock, so a stale or mismatched receipt burns the attempt. The receipt is re-read by path without a re-hash. |
| N02 | Four acceptance kinds, each with its own subject and hashed receipt bytes. Receipt contents are never parsed and there is no distinctness check, so one file can satisfy both `successor_*` kinds. lstat→read window. |
| N03 | Only the approval path is absolutized. Relative receipt or fresh-host paths fail in the children (`cwd=HERE` or `PREFLIGHT`) after side effects. |
| N04 | A READY or registration failure is fail-closed but unclassified: `protocol-finished` carries reason `client_startup_or_protocol_refusal`, the monitor returns normal, and the launcher records only `postflight:FileNotFoundError`. |
| N05 | An anchor mismatch makes the client skip the consumer's own close. The leaf is reaped by the launcher and no RETURNED receipt is written. |
| N06 | A begin-refusal reason (client01 495) is overwritten by a cut refusal (528). |
| N07 | R-ATTR: `begin` runs before the producer cut. The cut keeps its full .5 s but starts later. |
| N08 | `privacy_scan` decodes only `*_base64` JSON fields, not `.jsonl`/`.log`. The export-gate wording overstates this; capture-time redaction mitigates it. |
| N09 | The 20 absolute symlinks are not sealed. |
| N10 | A default-OFF run without `-B` may write `__pycache__` into the stage. |
| N11 | A SIGALRM between Popen and root/identity recording could leave an untracked root (only near `started+180`). |
| N12 | `remember` walks only main-thread children; `remember_adopted` is never called. The worker is a main-thread child, so it is discovered. |
| N13 | Runtime pins self-verify `consumer/common01` and `seal_inputs01`, and the 479-entry manifest is not re-verified at runtime. The Mayor must verify the stage on the fresh host. |
| N14 | The flock is taken with LOCK_NB, so `existing_Mayor_exclusive_GPU_lock_held=True` cannot mean another process holds `c.LOCK`. The prereg must define its meaning. |

## Premises (the disposition is conditional on all of them)

- **P01.** The Mayor and the filesystem are trusted. Attestations are digests, not cryptography.
- **P02.** The stage equals the sealed deployment02, uses `c.PYTHON`, and runs isolated with `-B`.
- **P03.** Process groups and PID/start-tick/pidfd identity cover the owned processes.
- **P04.** The worker exits on EOF or at its hard deadline (scr01).
- **P05.** No other GPU user ignores the lock.
- **P06.** No undecoded key encodings exist.
- **P07.** The scr01 A01–A12 conditions carry over only to hash-identical consumer files.
- **P08.** The Mayor Parent's 128+1 binding inventory is relied on, not re-derived.

## Answers (details and spans in report.json)

| Q | Status |
|---|---|
| Q01 approval gate before side effects | Verified (N01, N02, N03, N14). The template is refused at authorization02.py 53–57. |
| Q02 bindings / CONSUMER_SEAL / Python | Verified (N13). Consumer runtime files are byte-identical to successor01; the changes are as declared. |
| Q03 denied paths, READY/registration, Popen→registration | Verified (N01, N03, N04, N11, N12, N14). |
| Q04 80 ms / source+.5 / active end; producer deadlines | Composition verified, consumer internals inherited under A01–A12 (N07). A consumer-only refusal neither resets the deadline nor substitutes or retries. |
| Q05 .2 / 1900 / 60 / 20 | Verified unchanged against base acquire01. |
| Q06 all offered/unknown fates | No relabelling. Gap: `all-offered.json` is missing on non-joined paths (D2-F1); N06. |
| Q07 poison / publication failure | Verified. A sticky refusal is enforced, so an ignored return never means an absent refusal; `counted=false` does not mean nothing was written. |
| Q08 cleanup order / real RETURNED receipt | Verified, with blocker D2-F1. The pre-return `component.json` cannot substitute for the receipt. |
| Q09 config / privacy / Base64 export | Verified (N08). The config is retained on non-joined paths. |
| Q10 260/180/20/3/40/17 budgets | Verified. No bound is broadened. |
| Q11 R-SR4 / R-ATTR / R-COUNT | See the next section. |

The two questions the Mayor closed were verified directly in the source:
- `started.json` is written synchronously, and both services are spawned before the client.
- Cut GET and flush calls are not published.
- Caveat: genuine managed-offer transport during a lease does make the lease non-releasable, which is conservative.

## R-rule applicability

| Rule | Model-free diagnostic | Formal paired study |
|---|---|---|
| R-SR4 | Claim limit only. Strictness is Native-only, and no Gateway strictness was added. | Blocking unless both arms use an identical parser, or Gateway strictness is added. |
| R-ATTR | Report the begin, cut, handoff and publication durations separately. No arm attribution. | Blocking until identical-infrastructure and attribution rules exist. |
| R-COUNT | `counted=false` is not proof that nothing was written. Count all attempts and refusals. | Blocking for count and cost metrics. |

## Remaining live conditions

1. Resolve D2-F1 via R-a or R-b.
2. Four distinct receipts with absolute paths, a fresh-host receipt validated immediately before launch, and the prereg commit.
3. Verify the full stage against SHA256SUMS on the fresh host, and run with `-I -B`.
4. Define the lock-field semantics.
5. Analysis rules for N04, N06, N07, R-COUNT and R-SR4, with no paired claim.
6. Mayor acceptance of P01–P08.
7. Separate Mayor acceptance and prereg. This review grants nothing.

## Disclosures

- `add_pins_d2r01.py` was edited once, before its first and only run.
- One `grep` line of `consumer/worker01.py` (line 65, outside this selection) was displayed. The file is hash-identical to the successor01 copy that scr01 reviewed, and no conclusion depends on it.
- The first pin step ran on py314 only. `verify_d2r01.py` re-hashes every pin on both Pythons.
- Failures: none. All pin and addition steps exited 0, and no controls were run.
