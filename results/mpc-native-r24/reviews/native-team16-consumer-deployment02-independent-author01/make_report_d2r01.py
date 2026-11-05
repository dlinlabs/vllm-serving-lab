"""Deterministic builder for the deployment02 independent source-only disposition (d2r01).

build() reads only this review's own pin records (PLAN, PINS, POST-FREEZE-ADDITIONS*) and returns the
report; main() writes report.json exclusively. No subject read, no environment- or time-dependent field,
so py312 and py314 must regenerate byte-identical JSON (verify_d2r01.py checks this).
"""

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

SUBJECT_SEAL = "407bc2e88426f4997064fd7f62d09651af74a2c18c6e4989267f48a601a16ae8"
SUCCESSOR_SEAL = "13bc7f834a9bd0cbc0b8c69765642974e4ba8b1bce5f523666e1f7629f5f1393"
SCR01_SEAL = "1a6c00f513aec3442d954d591ed9a530a982313656ee750e01736cf134f0186f"
DISPOSITION01_SEAL = "c590c8378a59863a103c780e490814227f40f320f53bed55e63a26e95f694328"

DISPOSITION_CODE = "CONDITIONAL_SOURCE_BLOCKER_D2F1_NO_FALSE_PASS_FOUND"

TRACE = [
    dict(step="T01", what="Pure default-OFF activation gate is the first action of all three entry points",
         spans=["launch01.py:212-221", "authorization02.py:43-89",
                "service01.py:55-67 (activation 55, remote_only 63, owned_parent 65, verify_code 67)",
                "client01.py:848-860 (activation 848, remote_only 854, owned_parent 858, verify_code 860)"],
         result="verified; disabled path prints status disabled and returns 0 before remote_only, verify_code, "
                "lock, namespace, config or helper use; only stdlib/stage-module imports precede it (see N10)"),
    dict(step="T02", what="Launcher preconditions",
         spans=["launch01.py:222-237"],
         result="remote_only, verify_code, one-attempt namespace check, flock (LOCK_NB), STATE/PUBLIC mkdir, "
                "SIGALRM handler, startup alarm started+180; SIGTERM/SIGINT/SIGHUP only append to cancelled"),
    dict(step="T03", what="Host prerequisites before any config or service",
         spans=["launch01.py:245-264", "launch01.py:27-42"],
         result="fresh-host receipt (after namespace creation, N01), TMPDIR prelaunch receipt, source snapshot, "
                "import prerequisites, gpu_before + require_idle, boot id equals receipt, immutable model alias"),
    dict(step="T04", what="Private config and ownership root",
         spans=["launch01.py:265-268", "composition_common01.py:165-190", "owned01.py:59-64",
                "source13-entry02.py:71-80"],
         result="config created only after all prerequisites; Owned enables the child-subreaper role in the launcher"),
    dict(step="T05", what="Two services spawned before the client; started.json written synchronously",
         spans=["launch01.py:269-276", "owned01.py:66-83", "service01.py:84-103"],
         result="verified Mayor-closed question 1; each service re-runs the activation gate and execs its entry point"),
    dict(step="T06", what="Absolute-budget invocation file, then the client",
         spans=["launch01.py:277-287", "client01.py:709-717"],
         result="started+180/260/243, clock domain, CONSUMER_SEAL, approval sha, owner pid/start ticks; client cwd=PREFLIGHT"),
    dict(step="T07", what="Client session: consumer READY, registration, Wire, protocol",
         spans=["client01.py:751-764", "consumer/consumer01.py:43-119", "client01.py:688-706",
                "launch01.py:79-86", "owned01.py:91-111"],
         result="READY binds process/clock/boot/run/source-freeze; launcher acknowledges registration only after "
                "parent pid/start ticks/consumer CONTROL_FREEZE checks and register_leaf (pidfd)"),
    dict(step="T08", what="Protocol: single activation, bootstrap, initial cut, two managed offers, final phase",
         spans=["client01.py:607-685", "client01.py:459-604"],
         result="active_at = consumer.activate(); active deadline +20 published as active-start.json; ordinal-2 "
                "begin/cut/handoff at 484-528; final phase bounded by min(active+23, end+3); result.json then "
                "protocol-finished.json"),
    dict(step="T09", what="Launcher monitor",
         spans=["launch01.py:74-103"],
         result="50 ms polling with remember(); re-arms a one-shot alarm each loop (102) and returns without "
                "disarming it (D2-F1)"),
    dict(step="T10", what="Shared teardown",
         spans=["launch01.py:294-309", "owned01.py:158-223", "client01.py:720-745", "client01.py:769-790",
                "consumer/consumer01.py:348-437"],
         result="anchor publication precedes the cleanup arm and owned.cleanup inside one try (D2-F1); natural phase "
                "min(anchor+7, deadline-33), reap 30, join 2, reserve 1; client writes the real post-return staging "
                "receipt after boundary.close() returns"),
    dict(step="T11", what="Postflight",
         spans=["launch01.py:310-348", "launch01.py:111-157"],
         result="joined gate before all-offered.json; source/GPU/code/origin/privacy checks; persist_close binds the "
                "true returned close; component_pass conjunction; phase bounds"),
    dict(step="T12", what="Export and lock release",
         spans=["launch01.py:349-360", "launch01.py:160-209", "composition_common01.py:257-291"],
         result="supervision.json, joined+scan, result-index, rescan, private-config close after safe-export proof, "
                "rescan, export-gate; timer disarmed and flock released"),
]

D2F1 = dict(
    id="D2-F1",
    severity="medium",
    classification="source_blocker_for_the_single_live_attempt",
    title=("A pending monitor alarm or a failed cleanup-anchor publication skips owned.cleanup, leaving Native, "
           "Gateway, client and consumer worker running after the launcher exits and the GPU lock released"),
    inherited=True,
    inherited_from=("reference composition launch01: the monitor alarm logic and the finally ordering are unchanged "
                    "context lines in EXACT_SOURCE_DIFF.patch (launch hunks '@@ -103,11 +74,11 @@' and "
                    "'@@ -291,7 +295,7 @@' change only file names and the schema string)"),
    newly_live_reachable_because=("deployment02 adds the Mayor activation path; the reference composition was "
                                  "never live-runnable (SOURCE_BINDING author_launch_permitted False)"),
    code_facts=[
        "launch01.py:18-24: the SIGALRM handler raises TimeoutError; arm() sets a one-shot ITIMER_REAL and raises "
        "ValueError phase_deadline_already_exhausted (composition_common01.py:41-43) when the deadline has already "
        "passed. A later setitimer replaces a pending timer, so the stale timer matters only until line 300 completes.",
        "launch01.py:74-103: monitor() re-arms the one-shot timer on every ~50 ms loop (102) to "
        "min(started+203, deadline), with deadline = active_started_at+23 or started+180 (90); it returns at 95 "
        "(normal), 97, 99 and 101 without disarming the timer.",
        "launch01.py:294-308: main's outer finally (294-360) starts with lines 295-296 outside any inner try (as are "
        "309-313); an exception there leaves main and skips 349-360. The inner try publishes "
        "host-cleanup-anchor02.json first (298-299; composition_common01.py:54-61 exclusive partial open, dump, "
        "fsync, rename), re-arms for cleanup (300), and only then calls owned.cleanup (301-303); except "
        "BaseException (307-308) records 'cleanup:<type>' and leaves cleanup=None.",
        "launch01.py:310-317: joined=False when cleanup is None; 316 refuses before all-offered.json is written at "
        "317. publish_outputs (349-356) writes supervision.json and an export-gate with safe_to_export False and "
        "keeps the private config (close_private_config requires joined, 191). 358-360 disarm the timer and "
        "release the flock.",
        "owned01.py:66-83: owned roots are spawned with start_new_session=True and without a parent-death signal; "
        "the subreaper role exists only in the launcher process (owned01.py:59-64; source13-entry02.py:71-80); "
        "service01.py:84-103 execs Native/Gateway inside the owned root processes.",
    ],
    triggers=[
        dict(id="T1", kind="race",
             text=("After monitor returns normal=True at 95, the timer armed at 102 is still pending for "
                   "min(started+203, active+23). The client's final phase is bounded by min(active+23, end+3) "
                   "(client01.py:655), so in the slow/stall regime (active phase ending close to its 20 s bound and "
                   "the final drain/capture running to its deadline) protocol-finished.json can be observed only "
                   "tens of milliseconds before active+23. If the alarm is delivered while lines 297-300 run "
                   "(publish/fsync/rename or arm), TimeoutError is caught at 307 and owned.cleanup is skipped; if it "
                   "is delivered at 295-296, TimeoutError escapes main's finally and no supervision.json, "
                   "export-gate.json or all-offered.json is written. The same exposure applies when monitor raises "
                   "a non-alarm error (for example late_or_inconsistent_protocol_completion, 92-94) shortly before "
                   "the pending alarm."),
             window="about one 50 ms poll plus the anchor publish/fsync duration, immediately before active+23; "
                    "narrow but correlated with the busy regime this diagnostic is meant to observe"),
        dict(id="T2", kind="deterministic_under_OS_fault",
             text=("Any OSError from c.publish at 298-299 (open, write, fsync or rename: ENOSPC, EDQUOT, EIO) or "
                   "from the c.domain() procfs reads is caught at 307 and owned.cleanup is skipped. (exclusive_marker "
                   "is not a realistic cause: only the launcher publishes this file, once; client01.py:722 and 813 "
                   "only read it.)")),
        dict(id="T3", kind="narrow_race",
             text=("If the pending alarm is delivered inside the except body (291-293) before supervision is "
                   "reassigned, supervision keeps its initial anchor=started (244); arm(min(started+40, "
                   "started+243)) at 300 then raises phase_deadline_already_exhausted whenever more than 40 s have "
                   "elapsed since started, so owned.cleanup is skipped and the TimeoutError propagates after the "
                   "finally completes.")),
    ],
    effects=[
        "Native vLLM, Gateway, client and consumer worker keep running in their own sessions after the launcher "
        "exits; no owned process group is signalled. Descendant PID/start-tick identities exist only in launcher "
        "memory (owned.identities) and are lost; on disk remain each root's started.json (pid, start_ticks, "
        "process_group) and consumer-registered02.json.",
        "The flock on c.LOCK is released at 358-360 (or at process exit in the 295-296 sub-case) while owned GPU "
        "processes may still run. A later launch of this design would refuse at its own idle/fresh-host checks "
        "(launch01.py:246, 261), but any other user of the lock must rely on its own idle check.",
        "all-offered.json is not written. client/result.json and client/http receipts remain on disk if the client "
        "finished, but the launcher-level all-offered summary is missing.",
        "The private config holding ADMIN_API_KEY stays in STATE because close_private_config requires joined.",
        "Reporting stays truthful: cleanup=null, owned_joined=false, errors contain 'cleanup:<type>' (or the "
        "launcher exits with a TimeoutError traceback), export-gate safe_to_export=false, component_pass=false. "
        "No false-pass or relabelling risk was found.",
    ],
    author_coverage=("CONTROL_PLAN.json C08/C09 exercise the normal owned join and the private-config close; no "
                     "control injects a pending monitor alarm or an anchor-publication failure. LIMITATIONS.json "
                     "disclaims a universal OS-failure/orphan guarantee and asks the review to inspect failure paths; "
                     "REMAINING_PREREGISTRATION.json contains no manual reap procedure."),
    resolution_options=[
        dict(id="R-a", kind="surgical_successor",
             requirements=[
                 "S1: no stale one-shot alarm; disarm ITIMER_REAL on every monitor exit (return or raise) and as the "
                 "first statement of main's outer finally.",
                 "S2: isolate the host-cleanup-anchor publication in its own try; record its failure without "
                 "skipping cleanup.",
                 "S3: whenever owned is not None, always call owned.cleanup under a valid bound no later than "
                 "started+243, recomputing the anchor from the current monotonic time if the recorded anchor would "
                 "make the bound already exhausted; record its failure.",
                 "S4: when not joined, still write an explicitly incomplete all-offered snapshot (DO_NOT_EXPORT "
                 "unchanged) so offered fates are not lost behind the export gate.",
                 "Prove S1-S3 with one focused CPU control per Python: an injected pending alarm and an injected "
                 "anchor-publish OSError must both still join all owned helpers. Budgets 260/180/20/3/40/17 and "
                 "teardown allocation 7/30/2/1 stay unchanged.",
             ]),
        dict(id="R-b", kind="explicit_Mayor_risk_acceptance",
             requirements=[
                 "Preregister a manual post-run procedure. If supervision.json is missing, or shows cleanup null or "
                 "owned_joined false: identify the owned roots from PUBLIC/{native,gateway,client-process}/"
                 "started.json and consumer-registered02.json; verify PID start ticks before signalling; SIGKILL "
                 "the verified process groups; confirm no GPU compute apps and no listeners on 8000/8080 before "
                 "anyone reuses the GPU lock.",
                 "Record the run as a supervision failure, never as a workload outcome; preserve every output.",
             ]),
    ],
)

NOTES = [
    dict(id="N01", severity="low", classification="fail_closed_attempt_burning",
         spans=["launch01.py:225-231", "launch01.py:246", "launch01.py:27-42", "authorization02.py:87-88"],
         text=("fresh_host runs after the one-attempt namespace and the lock (225-231). A stale (>3600 s) or "
               "mismatching receipt refuses only after STATE/PUBLIC exist, which spends the single attempt. "
               "fresh_host reloads the receipt by path after authorization02 hashed it (no re-hash; "
               "trusted-filesystem premise)."),
         recommendation="Mayor validates the fresh-host receipt immediately before launch."),
    dict(id="N02", severity="low", classification="procedural_premise",
         spans=["authorization02.py:12-13", "authorization02.py:72-86", "authorization02.py:21-26"],
         text=("Exactly four acceptance kinds are required, each with accepted True, its own subject (successor_* "
               "-> CONSUMER_SEAL, observational_* -> protocol sha, deployment -> manifest sha) and a regular, "
               "single-link, <=16 MiB receipt whose bytes hash to the declared sha. Receipt contents are never "
               "parsed, and paths/digests are not required to be pairwise distinct: both successor_* kinds share "
               "one subject, so one file could satisfy both. Separation of the four Mayor decisions is therefore a "
               "trusted-Mayor procedural premise. lstat-then-read is a TOCTOU window under the trusted-filesystem "
               "premise."),
         recommendation="Prereg cites four distinct receipt files, one per decision."),
    dict(id="N03", severity="low", classification="fail_closed_attempt_burning",
         spans=["authorization02.py:85-89", "launch01.py:271-276", "launch01.py:284-287"],
         text=("Only the approval path is made absolute (89). Receipt and fresh-host paths are used as given: "
               "services re-validate with cwd=HERE and the client with cwd=PREFLIGHT, so a relative path that "
               "resolves in the launcher's cwd fails closed in a child after namespace, config and service startup."),
         recommendation="Prereg requires absolute receipt and fresh-host paths."),
    dict(id="N04", severity="low", classification="evidence_classification",
         spans=["client01.py:751-767", "client01.py:769", "client01.py:720-735", "launch01.py:91-95",
                "launch01.py:333", "launch01.py:139-157"],
         text=("A READY or registration failure writes client-session-refused.json, and shared_teardown_anchor then "
               "publishes protocol-finished.json with reason client_startup_or_protocol_refusal; the monitor returns "
               "normal=True and the launcher records only 'postflight:FileNotFoundError' (result.json absent). The "
               "outcome is fail-closed and all_offered labels both offers unobserved, but no machine-readable "
               "startup_prerequisite_failure class exists, and the reason string merges startup and protocol "
               "refusal."),
         recommendation=("Prereg analysis rule: client-session-refused.json present and active-start.json absent "
                         "means startup prerequisite/supervision failure, never a workload outcome.")),
    dict(id="N05", severity="low", classification="fail_closed",
         spans=["client01.py:740-745", "client01.py:769-790"],
         text=("If the launcher's cleanup anchor differs from the client's (abnormal launcher return between the "
               "client's two anchor reads), HostBoundConsumer.close refuses one_shared_cleanup_anchor before the "
               "consumer's own close; the leaf is left to the launcher reap and no RETURNED receipt exists.")),
    dict(id="N06", severity="low", classification="evidence_retention",
         spans=["client01.py:490-496", "client01.py:524-528", "consumer/consumer01.py:168-184"],
         text=("A begin refusal stored in attempt['consumer'] (495) is overwritten by a later cut refusal (528). "
               "begin raises without appending to the consumer's errors, so the begin reason can be lost when both "
               "begin and the cut refuse. handoff-intent.json survives unless the refusal precedes "
               "consumer01.py:173 (client01.py:492 or consumer01.py:171). Fail-closed: no client lease, no handoff."),
         recommendation="Retain begin and cut refusals in separate fields in any successor."),
    dict(id="N07", severity="note", classification="R-ATTR_timing",
         spans=["client01.py:490-500", "consumer/consumer01.py:168-184"],
         text=("begin runs synchronously before the producer cut; cut_deadline=min(deadline, now+.5) is computed "
               "after begin, so the producer keeps its full .5 s but starts later by the begin duration. The clocks "
               "to report this exist (collection_boundary_started_at, collector_call_entered_at, cut_returned_at).")),
    dict(id="N08", severity="low", classification="privacy_scope_premise",
         spans=["composition_common01.py:257-291", "launch01.py:185"],
         text=("privacy_scan searches raw bytes of every file and decodes only values under keys ending in "
               "'_base64' inside .json files. .jsonl/.log files and other encodings are not decoded, so the "
               "export-gate wording 'decoded response bodies' holds only for *_base64 JSON fields. Capture-time "
               "redaction (owned01 Reader, transport01.py:31-32, composition_common01.py:294-296) mitigates."),
         recommendation="Premise P06; optionally decode .jsonl lines in a successor."),
    dict(id="N09", severity="note", classification="packaging",
         spans=["attempts/*/cases/*current (20 absolute symlinks)"],
         text="The 20 symlinks are not covered by SHA256SUMS; they have no runtime role.",
         recommendation="Exclude them from the remote stage copy or ignore them in the manifest comparison."),
    dict(id="N10", severity="note", classification="stage_integrity",
         spans=["launch01.py:12-15", "composition_common01.py:141-144"],
         text=("A default-OFF invocation without -B/PYTHONDONTWRITEBYTECODE imports stage modules before the gate "
               "returns and can create __pycache__ inside the stage; verify_code checks only listed files. The live "
               "path enforces dont_write_bytecode through remote_only.")),
    dict(id="N11", severity="note", classification="ownership_premise",
         spans=["owned01.py:68-76", "launch01.py:233-237", "launch01.py:270", "launch01.py:283"],
         text=("SIGALRM is the only signal whose handler raises. Delivered between Popen returning and "
               "roots.append (68-71), or before the identity/pidfd are recorded (72-76), it could leave a spawned "
               "root untracked. Spawns happen only before started+180 while the startup alarm targets started+180, "
               "so this needs a spawn within the last instants of startup.")),
    dict(id="N12", severity="note", classification="ownership_premise",
         spans=["source13-entry02.py:42-63", "source13-entry02.py:82-87", "owned01.py:85-89",
                "consumer/consumer01.py:82-86"],
         text=("remember walks only /proc/<pid>/task/<pid>/children (children of each process's main thread); "
               "remember_adopted exists but owned01 never calls it. The consumer worker is spawned from the client's "
               "event-loop main thread, so the 50 ms launcher remember() can discover it before registration. "
               "Descendants spawned from non-main threads, or orphans adopted between polls, are not discovered by "
               "this walk; cleanup then relies on process-group signalling and on the worker's own EOF/hard-deadline "
               "exit.")),
    dict(id="N13", severity="low", classification="runtime_binding_scope",
         spans=["composition_common01.py:68-82", "composition_common01.py:158-162", "pins02.py:28-36",
                "consumer/common01.py:14", "consumer/common01.py:130-136", "authorization02.py:67-71"],
         text=("Runtime pins cover INPUTS/INPUT_FREEZE (including the owner helper copy), ADDITIONAL_INPUTS, the "
               "SUCCESSOR-SHA256SUMS copy, PLAN/OFFERS copies, SOURCE_BINDING, the consumer CONTROL_FREEZE map and "
               "the top-level CONTROL_FREEZE map. consumer/common01.py and seal_inputs01.py are imported and "
               "executed before their own digests are checked (self-verification), and the top-level map lists "
               "consumer/CONTROL_FREEZE.json but not consumer/*.py. The approval compares the deployment "
               "SHA256SUMS digest as a value; the 479-entry manifest is not re-verified at runtime. This detects "
               "accidental change, not adversarial change."),
         recommendation="Premise P02: the Mayor verifies the staged copy against the full deployment02 SHA256SUMS on the fresh host."),
    dict(id="N14", severity="low", classification="prereg_semantics",
         spans=["launch01.py:45-55", "launch01.py:227", "authorization02.py:63", "launch01.py:358-360"],
         text=("acquire_lock takes flock LOCK_EX|LOCK_NB on c.LOCK through a new open file description, before the "
               "namespace exists and outside the try. If another Mayor process (for example a flock(1) wrapper) "
               "holds that lock, the launcher refuses with BlockingIOError and no side effect. "
               "existing_Mayor_exclusive_GPU_lock_held=True therefore cannot mean that another process holds a flock "
               "on c.LOCK during the run. The lock fd is non-inheritable, so children never hold it."),
         recommendation="Prereg defines the field's meaning and makes the launcher the c.LOCK acquirer."),
]

PREMISES = [
    dict(id="P01", text="Trusted Mayor and filesystem: approval and receipts are filesystem attestations checked by "
                        "byte digest, not cryptographic signatures; receipt contents are not interpreted; no "
                        "adversarial swaps in lstat/read windows."),
    dict(id="P02", text="The remote stage equals sealed deployment02, checked against the full SHA256SUMS on the fresh "
                        "host; the interpreter is c.PYTHON; the launcher runs isolated and without bytecode writing "
                        "(remote_only enforces both)."),
    dict(id="P03", text="OS ownership: start_new_session process groups contain all Native/Gateway descendants, or "
                        "process-group signalling covers descendants the children walk misses; PID plus start-tick "
                        "identity and pidfd semantics hold."),
    dict(id="P04", text="The consumer worker exits on pipe EOF or at its hard deadline when the launcher reap does "
                        "not reach it (scr01 A-series, Mayor-accepted at 93221982741a723c17ecdc3732baec16a42829c6)."),
    dict(id="P05", text="No other GPU user ignores c.LOCK or the Mayor's exclusive-GPU procedure."),
    dict(id="P06", text="No component writes the admin key in an encoding the privacy scan does not decode."),
    dict(id="P07", text="Consumer runtime files byte-identical to successor01 (consumer01, worker01, validate01, "
                        "generation01, ipc01, fixture01) carry the scr01 conclusions only under conditional A01-A12."),
    dict(id="P08", text="Binding completeness relies on the Mayor Parent inventory (128+1 bindings, c00d29d2296bda7e24"
                        "9578f2de7c08650a4d79c9), which this review did not re-derive."),
]

ANSWERS = dict(
    Q01=dict(status="verified_with_notes", notes=["N01", "N02", "N03", "N14"],
             text=("All three entry points call require_activation first; the disabled path returns before any host, "
                   "config or helper side effect. The enabled path requires a bounded regular approval file, strict "
                   "JSON (duplicate keys and non-finite values refused), exact FIELDS, schema, scope and "
                   "execute_once; actuation and model qualification False; exact capture/operation ids, GPU, ROOT, "
                   "PYTHON, the lock flag and a 40-hex preregistration commit; digests of SHA256SUMS, PROTOCOL, "
                   "NATIVE_PLAN and OFFERS; exactly four acceptances with per-kind subjects and hashed receipt "
                   "bytes; and a hashed fresh-host receipt. These are filesystem attestations, not cryptography. "
                   "MAYOR_APPROVAL_TEMPLATE_INVALID.json fails at authorization02.py:53-57 (execute_once false; "
                   "also lock flag false, null commit/manifest/receipts, accepted false)."),
             spans=["authorization02.py:43-89", "launch01.py:217-221", "service01.py:55-59", "client01.py:848"]),
    Q02=dict(status="verified_with_notes", notes=["N13"],
             text=("The approval binds the deployment SHA256SUMS, PROTOCOL, NATIVE_PLAN and OFFERS digests, "
                   "CONSUMER_SEAL (successor01 manifest 13bc7f83...), ids, GPU, ROOT and PYTHON. CONSUMER_SEAL is "
                   "the reviewed-successor identity: it is also bound by the receipts' successor subjects, "
                   "invocation_check, the invocation file and the SUCCESSOR-SHA256SUMS copy check. The runnable "
                   "consumer is the deployment copy, bound by the deployment manifest digest (approval), the "
                   "consumer CONTROL_FREEZE (runtime; READY 104; launcher registration 84; SOURCE_BINDING 77-80) "
                   "and the top-level CONTROL_FREEZE. Hash-only identity: consumer01, worker01, validate01, "
                   "generation01, ipc01 and fixture01 equal successor01; common01 (NAMESPACE/CAPTURE), "
                   "seal_inputs01 (delegates to ../pins02.py), NATIVE_PLAN/OFFERS (ids) and CONTROL_FREEZE changed "
                   "as declared in EXACT_SOURCE_DIFF.patch. Python: remote_only requires HERE==STAGE, "
                   "sys.prefix==PYTHON venv, isolated, dont_write_bytecode, no optimize; children run c.PYTHON -I -B "
                   "and the worker sys.executable -I -B."),
             spans=["authorization02.py:58-71", "composition_common01.py:38", "composition_common01.py:68-82",
                    "composition_common01.py:141-144", "launch01.py:84", "consumer/consumer01.py:104",
                    "client01.py:709-717"]),
    Q03=dict(status="verified_with_notes", notes=["N01", "N03", "N04", "N11", "N12", "N14"],
             text=("Order: gate, remote_only, verify_code, namespace, lock, mkdir, startup alarm, fresh host, "
                   "prerequisites, config, Owned, native, gateway, invocation, client, monitor. Refusals: disabled, "
                   "missing approval, template, any binding mismatch, spent namespace, lock contention (before "
                   "namespace creation), fresh-host mismatch (after namespace creation, N01), relative receipt "
                   "paths in children (N03), "
                   "cancelled or late before a service or the client. A READY/registration failure is fail-closed but "
                   "lacks a machine-readable class (N04). Popen-to-registration: the worker is a client main-thread "
                   "child discovered by the 50 ms remember(), then pinned by register_leaf with pidfd; residual "
                   "windows N11 and N12 remain premises."),
             spans=["launch01.py:212-288", "owned01.py:66-111", "client01.py:688-706", "launch01.py:79-86"]),
    Q04=dict(status="verified_composition_consumer_internals_inherited", notes=["N07"],
             text=("A single activation (client01.py:618-622) sets active_at, the deadline active+20 and "
                   "active-start.json; the consumer lease uses authority.start(active_at+20) (consumer01.py:177), so "
                   "the active end is shared. The pipeline min(begin+.080, source+.500, active end) is computed in "
                   "consumer code byte-identical to successor01 (PIPELINE .080 unchanged at consumer/common01.py:22); "
                   "this review inherits it from scr01 under A01-A12 and does not re-derive it. Producer cuts use "
                   "min(deadline, now+.5) at 500, 599 and 631, as in base acquire01 451/524. A consumer-only "
                   "refusal is caught at 494-496 or 520-521, the producer cut proceeds, and the deadline is not "
                   "reset; consumer01.py:171-175 forbid a second begin or a non-ordinal-2 substitute, and there is "
                   "no retry. Source clock: markers, invocation and anchors are bound to c.domain() (boot id plus "
                   "time namespace)."),
             spans=["client01.py:484-530", "client01.py:607-647", "consumer/consumer01.py:121-127",
                    "consumer/consumer01.py:168-184", "launch01.py:58-71"]),
    Q05=dict(status="verified", notes=[],
             text=("Unchanged against base acquire01: settings .2 s/1900/1 MiB (client01 33 = base 31); warmup "
                   "limit 60 (249 = 226); managed limit 20 (469 = 427); CAS not_after +.5 (292 = 267); final "
                   "deadline (655 = 573). Capture requires maximum_http_exchanges <= 1900 (transport01.py:20-21). "
                   "The .2 s value is the "
                   "per-exchange HTTP timeout, not an inference cap; inference uses limit=60 for warmup and 20 for "
                   "managed, as in the base. bounded_loop waits only tighten toward the cutoff (823-825)."),
             spans=["client01.py:33", "client01.py:249", "client01.py:469", "client01.py:292", "client01.py:655",
                    "transport01.py:20-21", "client01.py:797-838"]),
    Q06=dict(status="verified_with_gap", notes=["N04", "N06"], findings=["D2-F1"],
             text=("Offers start as not_sent and change fate only on observed transport; all_offered uses the "
                   "reported offers, or unobserved/not_sent when no result exists, requires exactly the two planned "
                   "request_ids, lists raw receipts, the warmup-send flag and missing_receipt_not_proof_of_no_send. "
                   "Consumer publication refusals are retained. No relabelling was found. Gap: on any non-joined "
                   "path (including D2-F1) all-offered.json is never written (316 before 317); N06 can lose a "
                   "begin-refusal reason."),
             spans=["client01.py:608-609", "launch01.py:139-157", "launch01.py:316-317", "client01.py:675"]),
    Q07=dict(status="verified", notes=[],
             text=("publish converts any exception into invalidate_publication: sticky authority poison, a refusal "
                   "row (consumer_refused_producer_continues; counted=False unless an attempt row exists, then "
                   "counted=committed), errors, a refusal file and cleanup component_pass False. Capture.observed "
                   "routes publisher exceptions to the same invalidator and requires authority_poisoned on refusal "
                   "receipts. The client wrapper ignores the return value (357-359), but refusals are sticky and "
                   "enforced at client01.py:675-677, consumer01.py:419-420/435 and launch01.py:337, so an ignored "
                   "return never means an absent refusal (R-ATTR). counted=False is not proof that nothing was "
                   "written (R-COUNT). R-SR4: consumer and transport files are byte-identical; no Gateway strictness "
                   "was added; strict duplicate-key refusal remains a Native-only claim."),
             spans=["consumer/consumer01.py:129-166", "transport01.py:34-55", "client01.py:357-359",
                    "client01.py:675-677", "consumer/consumer01.py:419-436", "launch01.py:335-340"]),
    Q08=dict(status="verified_with_blocker", findings=["D2-F1"], notes=["N05"],
             text=("Normal order: protocol-finished, launcher anchor, cleanup arm, owned.cleanup (natural phase until "
                   "min(anchor+7, deadline-33); services signalled after the client exits; scoped reap 30; join 2; "
                   "reserve 1). In parallel the client reads the shared anchor, closes the wire, and awaits "
                   "boundary.close(). The consumer writes a pre-return component.json (status "
                   "cleanup_evidence_pending_final_return_guard, component_pass False) and returns status complete "
                   "with cleanup_return_guard_at. The client writes consumer-close-returned02.json (0600) only after "
                   "the return. persist_close requires joined and a private single-link file, copies the bytes "
                   "exclusively with fsync, and checks schema, domain, anchor, cutoff, deadline, status complete, "
                   "all_joined and chronology. The pre-return component.json cannot substitute. Shared teardown "
                   "ownership is broken only on the D2-F1 paths."),
             spans=["launch01.py:294-309", "owned01.py:158-223", "client01.py:769-790",
                    "consumer/consumer01.py:419-437", "launch01.py:111-136"]),
    Q09=dict(status="verified_with_notes", notes=["N08"], findings=["D2-F1"],
             text=("The config is created only after all prerequisites; the key is redacted at capture "
                   "(Reader, Capture.scrub) and in failures. Export order: supervision.json; if joined, "
                   "scan -> result-index -> rescan -> close_private_config (joined, scan, safe-export proof, key "
                   "identity, unlink, record) -> rescan; then export-gate (safe and removed). Response bodies stored "
                   "as response_body_base64 are decoded by the scan. On a non-joined path (D2-F1) the config is "
                   "retained and export is refused."),
             spans=["launch01.py:160-209", "composition_common01.py:165-190", "composition_common01.py:257-296",
                    "transport01.py:31-32"]),
    Q10=dict(status="verified", notes=[],
             text=("PROTOCOL bounds 260/180/20/3/40/17 equal the base. Startup alarm started+180; monitor deadline "
                   "active+23 or started+180 with cap started+203; cleanup bound min(anchor+40, started+243); IO "
                   "deadline min(started+260, cleanup_finished+17); phase_bounds 344-347; consumer activation requires "
                   "active+20+3+40+17 <= hard deadline; invocation_check requires exact +180/+260/+243. Teardown "
                   "allocation 7/30/2/1 = 40. No bound is broadened."),
             spans=["launch01.py:234", "launch01.py:90", "launch01.py:102", "launch01.py:300", "launch01.py:313",
                    "launch01.py:344-347", "consumer/consumer01.py:125", "client01.py:709-717", "owned01.py:158-161"]),
    Q11=dict(status="answered",
             text="See r_rules_applicability: claim limits for the model-free diagnostic; blocking for a formal paired study."),
)

R_RULES = dict(
    R_SR4=dict(diagnostic="claim_limit_only: strict duplicate-key refusal is Native-only; deployment02 adds no "
                          "Gateway strictness (consumer and transport files byte-identical)",
               formal_paired_study="blocking unless both arms share an identical parser path or Gateway "
                                   "strictness is added and independently reviewed"),
    R_ATTR=dict(diagnostic="report begin, producer cut, handoff and publication durations separately from the "
                           "existing clocks; no arm attribution; begin-before-cut ordering disclosed (N07)",
                formal_paired_study="blocking until prospective identical-infrastructure and attribution rules exist"),
    R_COUNT=dict(diagnostic="counted=False is not proof of no write; publication cost and busy incidence count all "
                            "attempt rows (committed or not) and every sticky refusal",
                 formal_paired_study="blocking for count/cost metrics until write-ahead attempt accounting is defined"),
    scope=("A model-free component run measures publication cost, busy incidence, producer continuation and "
           "whole-boundary timing; it is not a paired study and supports no arm-versus-arm attribution."),
)

MAYOR_CLOSED_QUESTIONS = [
    dict(question="Owned.spawn synchronously writes started.json before returning; the launcher spawns both "
                  "services before the client.",
         result="verified", spans=["owned01.py:82-83", "launch01.py:269-287"]),
    dict(question="Capture.observed publishes only offer/control history; an ordinary cut GET or flush does not "
                  "self-invalidate the lease.",
         result="verified",
         spans=["transport01.py:34-55", "client01.py:65-84", "client01.py:151"],
         caveat=("Cut GET /admin/mpc/state, POST /admin/mpc/flush-prefix and GET /pr33-native/phase-state carry no "
                 "request_id and are not control paths, so they are not published. Genuine managed-offer transport "
                 "progress during a lease is published; per the byte-identical scr01-reviewed generation semantics "
                 "this makes the release non-releasable, which is a conservative measured outcome rather than "
                 "self-invalidation by a GET or flush.")),
]

MAYOR_C00_ITEMS = [
    dict(item="producer collection min(call+.5, original active end)", result="verified",
         spans=["client01.py:500", "client01.py:599", "client01.py:631"]),
    dict(item="consumer pipeline min(begin+.080, source+.500, original active end)",
         result="inherited_from_byte_identical_consumer_code_under_A01-A12", spans=["consumer/consumer01.py:177"]),
    dict(item="consumer-only refusal does not reset producer deadlines, substitute the ordinal or retry",
         result="verified", spans=["client01.py:490-528", "consumer/consumer01.py:171-175"]),
    dict(item="acquisition .2 is not an inference cap (warmup <=60, managed <= remaining 20)", result="verified",
         spans=["client01.py:33", "client01.py:249", "client01.py:469"]),
    dict(item="startup READY/registration failure is a prerequisite/supervision failure",
         result="fail_closed_but_unclassified_N04", spans=["client01.py:751-769", "launch01.py:333"]),
    dict(item="every offered/not-sent/unknown/failed/cleanup outcome retained, no relabelling",
         result="no_relabelling_found; all-offered.json missing on non-joined paths (D2-F1); N06",
         spans=["launch01.py:139-157", "launch01.py:316-317"]),
    dict(item="ignored return must not mean absent refusal", result="verified",
         spans=["client01.py:357-359", "client01.py:675-677", "launch01.py:337"]),
    dict(item="counted=false after pwrite/fsync failure is not proof of no write", result="agreed_R-COUNT",
         spans=["consumer/consumer01.py:143-157"]),
    dict(item="gateway strictness remains a Native-only claim unless the source adds more",
         result="verified_no_new_gateway_strictness", spans=[]),
]

REMAINING_LIVE_CONDITIONS = [
    "D2-F1 resolved by R-a (surgical successor plus a focused control) or R-b (explicit Mayor risk acceptance plus "
    "a preregistered manual reap/GPU-idle procedure) before the single live attempt.",
    "Four distinct Mayor receipt files with absolute paths (N02, N03), fresh-host receipt validated immediately "
    "before launch (N01), and the preregistration commit.",
    "Full staged-copy verification against deployment02 SHA256SUMS on the fresh host; launch isolated with -B (N13, P02).",
    "Lock-field semantics defined; the launcher is the c.LOCK acquirer (N14).",
    "Preregistered analysis rules: startup-prerequisite classification (N04), begin-refusal retention (N06), "
    "R-ATTR durations (N07), R-COUNT all-attempt cost, R-SR4 claim limit, and no paired-study claim.",
    "Premises P01-P08 accepted by the Mayor, or tested separately.",
    "Separate Mayor acceptance of this deployment and the Mayor's freeze/prereg; this review grants nothing.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    plan = json.loads((HERE / "PLAN.json").read_bytes())
    pins = json.loads((HERE / "PINS.json").read_bytes())
    first = json.loads((HERE / "POST-FREEZE-ADDITIONS.json").read_bytes())
    second = json.loads((HERE / "POST-FREEZE-ADDITIONS-02.json").read_bytes())
    selection = [dict(path=name, bytes=row["bytes"], sha256=row["sha256"], origin="first_selection")
                 for name, row in sorted(pins["selection"].items())]
    additions = []
    for sequence, document in ((1, first), (2, second)):
        for row in document["additions"]:
            additions.append(dict(sequence=sequence, path=row["path"], bytes=row["bytes"], sha256=row["sha256"],
                                  reason=row["reason"]))
            selection.append(dict(path=row["path"], bytes=row["bytes"], sha256=row["sha256"],
                                  origin="post_freeze_addition_%d" % sequence))
    identity = pins["hash_only_identity"]
    return dict(
        schema="d2r01-report-v1",
        reviewer=dict(role="Role7 independent consumer/composition source disposition", model_id="claude-opus-5.5",
                      reasoning_effort="max", nested_or_child_agents=0),
        subject=dict(path=pins["subject_path"], SHA256SUMS_sha256=pins["subject_SHA256SUMS_sha256"],
                     sealed_entries=pins["subject_sealed_entries"],
                     regular_files_including_seal=pins["subject_regular_files"],
                     unsealed_absolute_symlinks=len(pins["subject_symlinks_not_sealed"]),
                     mayor_engineering_acceptance_commit_reported="c00d29d2296bda7e249578f2de7c08650a4d79c9 "
                                                                  "(reported; not read or repeated)"),
        related_sealed_inputs=dict(successor01_SHA256SUMS_sha256=pins["successor_SHA256SUMS_sha256"],
                                   deployment_copy_of_successor_SHA256SUMS_sha256=pins[
                                       "deployment_copy_of_successor_SHA256SUMS_sha256"],
                                   scr01_review_SHA256SUMS_sha256=SCR01_SEAL,
                                   scr01_disposition_SHA256SUMS_sha256=DISPOSITION01_SEAL,
                                   scr01_and_disposition_acceptance_commit_reported=
                                   "93221982741a723c17ecdc3732baec16a42829c6"),
        pins=dict(PLAN_sha256=sha(HERE / "PLAN.json"), PINS_sha256=sha(HERE / "PINS.json"),
                  additions_01_sha256=sha(HERE / "POST-FREEZE-ADDITIONS.json"),
                  additions_02_sha256=sha(HERE / "POST-FREEZE-ADDITIONS-02.json"),
                  review_start_marker=(HERE / "MARKER-review-start.txt").read_text().strip(),
                  freeze_marker=(HERE / "MARKER-freeze.txt").read_text().strip(),
                  first_selection_pinned_before_content_read=True, additions_pinned_before_content_read=True,
                  first_pin_step_python="3.14.4 only; verify_d2r01.py re-hashes every pin on both Pythons"),
        budget_use=dict(files=second["total_files"], files_limit=plan["budget"]["selected_files_limit"],
                        bytes=second["total_bytes"], bytes_limit=plan["budget"]["selected_bytes_limit"],
                        post_freeze_additions=second["additions_used_total"],
                        post_freeze_additions_limit=plan["budget"]["post_freeze_additions_limit"],
                        controls_run_per_python=0,
                        controls_limit_per_python=plan["budget"]["new_controls_limit_per_python"],
                        controls_rationale=("Every finding is source-evident. Demonstrating D2-F1 live would need a "
                                            "valid or fake approval, a monkeypatched gate or the live main(), all "
                                            "prohibited; a stand-alone alarm witness would only restate Python "
                                            "signal semantics.")),
        selection=selection,
        span_paths=dict(
            base="Spans cite line numbers of the sealed deployment02 bytes, relative to the subject root.",
            abbreviations={"source13-entry02.py": "consumer/inputs/base/inputs/launch02/inputs/source13-entry02.py",
                           "base acquire01": "consumer/inputs/base/acquire01.py"}),
        post_freeze_additions=additions,
        hash_only_identity=dict(
            identical_to_successor01=sorted(k for k, v in identity.items() if v["identical_to_successor"]),
            changed_vs_successor01=sorted(k for k, v in identity.items() if not v["identical_to_successor"]),
            identical_to_generation01=sorted(k for k, v in identity.items() if v["identical_to_generation01"])),
        disclosures=[
            "add_pins_d2r01.py was edited once to match the PINS schema before its first and only run.",
            "One grep line of consumer/worker01.py (line 65; not in this selection) was displayed while locating "
            "CONTROL_FREEZE references; the file is hash-identical (ba353590...) to successor01 candidate/worker01.py "
            "from the scr01 selection, and no conclusion depends on it.",
            "Line numbers cite the sealed deployment02 bytes pinned in PINS.json and POST-FREEZE-ADDITIONS*.json.",
        ],
        trace=TRACE,
        findings=[D2F1],
        notes=NOTES,
        premises=PREMISES,
        questions=plan["questions"],
        answers=ANSWERS,
        r_rules_applicability=R_RULES,
        mayor_closed_source_questions=MAYOR_CLOSED_QUESTIONS,
        mayor_c00_design_items=MAYOR_C00_ITEMS,
        remaining_live_conditions=REMAINING_LIVE_CONDITIONS,
        disposition=dict(
            code=DISPOSITION_CODE,
            summary=("Within this selection and under premises P01-P08, no false-pass, relabelling, budget-broadening "
                     "or approval-bypass defect was found in the traced default-OFF deployment02 route. One medium "
                     "source blocker (D2-F1, inherited and now live-reachable) must be resolved by R-a or R-b before "
                     "the single live attempt."),
            blockers=["D2-F1"], conditional_on=[p["id"] for p in PREMISES],
            is_acceptance=False, not_a_proof=True,
            ownership_and_OS_failure_safety_universally_established=False),
        gates=dict(G06="not_closed", G16="pending", Native14="FAIL_UNCHANGED", gate_status_changes=[]),
        permissions=dict(native_execution=False, gpu=False, ssh=False, network=False, git=False,
                         approval_created=False, live_main_called=False, source_patched=False,
                         subject_or_sealed_writes=False, nested_agents=False, native_launch_authorized=False),
        not_claimed=plan["not_claimed"],
        failures_preserved=[],
        failures_note="No failure occurred: the pin and both addition steps exited 0, and no controls were run.",
        helper_processes="none spawned; every command ran synchronously to completion",
    )


def render():
    return json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def main():
    text = render()
    with open(HERE / "report.json", "x") as handle:
        handle.write(text)
    report = json.loads(text)
    print(json.dumps(dict(report_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          disposition=report["disposition"]["code"], files=report["budget_use"]["files"],
                          bytes=report["budget_use"]["bytes"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
