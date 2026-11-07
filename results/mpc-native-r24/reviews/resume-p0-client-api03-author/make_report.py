"""Write report.json for the client-api03 scope from recorded files (run once; no import of the edited modules)."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/daniel_lin/gt/pr32/mayor/mpc-local-r21')
SCOPE = ROOT / '.runtime/r24-resume-p0-nominal-client-api03'
SRC = 'experiments/mpc-research/resume-mvp'
OWNED = ('nominal_client.py', 'test_nominal_client.py')
UNCHANGED = ('nominal_metrics.py', 'test_nominal_metrics.py')
FROZEN = ('STARTED.txt', 'ORIGINALS.sha256', 'API-FREEZE.json', 'PLAN.json', 'pytest-empty.ini', 'run_attempt.sh',
          'static-check.txt')
PUBLIC = ('PreparedCapture', 'prepare_capture', 'capture_replay', 'write_capture', 'replay')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return Path(path).read_bytes().count(b'\n')


def top_level(path):
    text = Path(path).read_text()
    tree, out = ast.parse(text), {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
            for target in node.targets:
                out[target.id] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node is tree.body[0]:
            out['<module docstring>'] = node.value.value
    return text, out


def signature(node):
    prefix = 'async def ' if isinstance(node, ast.AsyncFunctionDef) else 'def '
    return f'{prefix}{node.name}({ast.unparse(node.args)})'


def attempt(name):
    base = SCOPE / 'attempts' / name
    read = lambda file: (base / file).read_text().strip()  # noqa: E731
    return {
        'dir': f'attempts/{name}', 'run_sh': read('run.sh').splitlines()[-1], 'run_sh_sha256': sha(base / 'run.sh'),
        'started': read('started.txt'), 'ended': read('ended.txt'), 'exit_code': int(read('exit_code.txt')),
        'result_line': read('log.txt').splitlines()[-1], 'log_sha256': sha(base / 'log.txt'),
        'pins_before_equal_after_and_pycache_unchanged': read('unchanged.txt') == 'unchanged_exit=0',
        'pinned_copies_equal_final_sources': all(sha(base / 'pinned' / file) == sha(ROOT / SRC / file)
                                                 for file in OWNED),
        'interpreter': 'files/r24-mpc-py312-venv/bin/python3.12 (3.12.16; pytest 9.2.0; httpx 0.28.1)',
    }


pins = [line.split(None, 1) for line in (SCOPE / 'ORIGINALS.sha256').read_text().splitlines()]
originals = {name: digest for digest, name in pins[:4]}
sources = {}
for file in OWNED:
    assert sha(SCOPE / 'originals' / file) == originals[file]
    sources[f'{SRC}/{file}'] = {'original_sha256': originals[file], 'final_sha256': sha(ROOT / SRC / file),
                                'original_lines': lines(SCOPE / 'originals' / file),
                                'final_lines': lines(ROOT / SRC / file)}
for file in UNCHANGED:
    assert sha(ROOT / SRC / file) == originals[file], file
    sources[f'{SRC}/{file}'] = {'pinned_sha256': originals[file], 'unchanged': True}

old_text, old_defs = top_level(SCOPE / 'originals/nominal_client.py')
new_text, new_defs = top_level(ROOT / SRC / 'nominal_client.py')
definitions = {
    'changed': sorted(name for name in old_defs if name in new_defs and old_defs[name] != new_defs[name]),
    'added': sorted(name for name in new_defs if name not in old_defs),
    'removed': sorted(name for name in old_defs if name not in new_defs),
    'byte_identical_count': sum(1 for name in old_defs if new_defs.get(name) == old_defs[name]),
}
assert definitions['removed'] == []
IDENTICAL = ('_request', 'StreamParse', 'normalized_row', 'summarize_records', 'build_run', 'check_binding',
             'check_tokenizer', 'check_served_models', '_run_cancellable', '_finalize_main', 'main', '_write_new',
             'prepare_records', 'clock_domain', 'load_tokenizer')
assert all(name in old_defs and new_defs[name] == old_defs[name] for name in IDENTICAL), IDENTICAL
nodes = {node.name: node for node in ast.parse(new_text).body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
constants = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(new_text).body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in ('EXIT_CODES', 'NOT_RUN_ERROR')}
slots = next(ast.literal_eval(item.value) for item in nodes['PreparedCapture'].body
             if isinstance(item, ast.Assign) and item.targets[0].id == '__slots__')

old_tests = (SCOPE / 'originals/test_nominal_client.py').read_bytes()
new_tests = (ROOT / SRC / 'test_nominal_client.py').read_bytes()
old_names = {node.name for node in ast.parse(old_tests).body if isinstance(node, ast.FunctionDef)}
added_tests = [node.name for node in ast.parse(new_tests).body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in old_names]

report = {
    'schema': 'r24-resume-p0-nominal-client-api03-report-v1',
    'scope': '.runtime/r24-resume-p0-nominal-client-api03',
    'role': 'PR32 team16 Role7 fair benchmark metrics and negative cases',
    'model': {'id': 'claude-opus-5.5', 'reasoning_effort': 'max'},
    'status': 'SEALED: additive in-loop raw-v2 capture API in nominal_client.py; CPU literal controls 43/43 on '
              'py3.12 (2 new + 41 pre-existing, one attempt, no retries); NOT independently accepted (Role4 B '
              'review pending); no Native/GPU/HTTP-to-real-service evidence',
    'trigger': {'scope': '.runtime/r24-resume-p0-nominal-episode01',
                'SHA256SUMS_sha256': 'c3f9bbcd1be6d78a3b860a1de355f34682d6328cd370c17bff3a4fb5eb17cec7',
                'entries': 21, 'verified': 'sha256sum -c before reading',
                'read': 'BLOCKER.json and API_EVIDENCE.json only',
                'blocker': 'NO_GO_SAFE_RAW_V2_COMPOSITION_API_MISSING'},
    'scope_control_files_sha256': {name: sha(SCOPE / name) for name in FROZEN},
    'scope_control_note': 'STARTED/ORIGINALS/API-FREEZE/PLAN/pytest-empty.ini written before any source edit; '
                          'run_attempt.sh and static-check.txt after the edits, before attempt01',
    'sources': sources,
    'definitions_vs_original': definitions,
    'definitions_note': 'top-level def/class/assignment source segments compared by ast; every definition not '
                        'listed as changed/added is byte-identical, asserted for ' + ', '.join(IDENTICAL),
    'api': {
        'signatures': {name: signature(nodes[name]) for name in PUBLIC if name != 'PreparedCapture'},
        'PreparedCapture': {'__slots__': list(slots), 'construct': 'only via prepare_capture (single use)'},
        'constants': constants,
        'usage': 'prepared = prepare_capture(...) [sync, before the start exists; no HTTP/loop/signals]; '
                 'if prepared.ready: capture = await capture_replay(prepared, offered_start=S, deadline=D, '
                 'cleanup_reserve_s=R, clock_domain=nominal_client.clock_domain()) inside the caller loop; '
                 'write_capture(path, prepared.capture) in every outcome (also after refusal/cancellation)',
        'prepare_capture': 'same envelope initializer and strict input loading as the CLI (sources, trace, '
                           'binding, loopback URLs, provenance purpose) plus arm in RUN_ARMS, the CLI purpose rule, '
                           'request_timeout finite > 0, check_binding, tokenizer from local_model_path and '
                           'check_tokenizer; refusal/failure returned (ready False, status refused/failed), never '
                           'raised; ready envelope keeps status failed with NOT_RUN_ERROR until capture_replay',
        'capture_replay': 'ClientRefusal raised (envelope untouched) unless ready and unused; caller values recorded '
                          'in config (non-finite -> null); refusals with zero sends: non-finite inputs or reserve < '
                          '0, clock_domain differs, deadline - reserve < offered_start + duration (never truncated), '
                          'offered_start not in (now, now + 600 s] at entry and again after GET /v1/models; then the '
                          'unchanged replay with drain_deadline = deadline - cleanup_reserve_s exactly; '
                          'collected/refused/failed returned; CancelledError/KeyboardInterrupt finalize interrupted '
                          'in place then re-raise; other BaseException finalize failed then re-raise; rows and '
                          'summary cover every allocated record whenever records exist',
        'write_capture': 'the CLI capture.json codec (_write_new: sorted keys, indent 1, allow_nan False, exclusive '
                         'create, fsync); returns sha256 of the written bytes',
        'cli': 'main/replay/finalize argv, _run_cancellable signal semantics around asyncio.run, exit codes '
               '(EXIT_CODES values unchanged), stdout/stderr lines and capture.json keys unchanged; _replay_main and '
               '_capture delegate to _new_capture/_load_inputs/_preflight_local/_collect/_settle',
    },
    'role6_requirements': {
        'awaitable_entry_or_prepare_capture_finalize': 'prepare_capture + await capture_replay (+ write_capture); '
                                                       'metrics finalization stays build_run (unchanged)',
        'reuse_same_replay_sse_rowmapper_codec': 'capture_replay -> _collect -> _replay -> unchanged _request/'
                                                 'StreamParse; _settle -> unchanged normalized_row/'
                                                 'summarize_records; write_capture -> unchanged _write_new; the CLI '
                                                 'uses the same helpers',
        'no_asyncio_run_or_signal_handlers_in_library': 'none in prepare_capture/capture_replay/_collect/_replay; '
                                                        'test 1 runs inside a caller loop with a concurrent heartbeat '
                                                        'sampling signal.getsignal(SIGINT/SIGTERM)',
        'caller_exact_future_start_original_deadline_reserve_never_retimed': 'offered_start used verbatim (records '
                                                                             'scheduled_at = start + arrival); '
                                                                             'drain_deadline = deadline - reserve; '
                                                                             'passed/truncating/foreign-domain/'
                                                                             'non-finite inputs refused, zero sends',
        'preload_binding_tokenizer_before_start': 'prepare_capture runs check_binding + tokenizer + check_tokenizer '
                                                  'before any offered start exists; unbound alias refused there',
        'retain_all_offers_partial_capture_tasks_joined_failure_visible': 'except BaseException -> _settle in place; '
                                                                          'cancellation re-raised after every '
                                                                          'request task is joined; Exceptions '
                                                                          'returned as failed/refused envelopes',
        'caller_lifecycle_owns_interrupts': 'no handler installation; the caller catches CancelledError/'
                                            'KeyboardInterrupt and persists prepared.capture',
    },
    'tests': {
        'file': f'{SRC}/test_nominal_client.py',
        'existing_tests_unchanged': new_tests.startswith(old_tests),
        'existing_note': 'original file bytes are an exact prefix of the final file (additions appended only)',
        'added': added_tests,
        'test_library_capture_inside_running_loop_keeps_caller_clock_and_cli_envelope': [
            'prepared envelope: ready, status failed + NOT_RUN_ERROR, records null, 4 prompts tokenizer-verified',
            'awaited inside a running loop with a concurrent caller heartbeat task: collected, no leftover tasks, '
            'heartbeat before start and after offered_end, signal.getsignal samples constant',
            'offered_start/offered_end/drain_deadline == start / start + 0.4 / (start + 1.2) - 0.5 exactly; config '
            'records entry library + caller start/deadline/reserve/domain',
            'all 4 offers retained: complete, rejected admission_capacity_503, complete (length), cancelled '
            'response_body at the drain deadline (before the caller deadline); scheduled_at exact',
            'rows/summary equal the unchanged mappers; 1 GET /v1/models + 4 POSTs',
            'write_capture sha matches bytes, JSON round-trip equal, second write FileExistsError',
            'unchanged build_run + nominal_metrics.compute: all_offered 4, cancelled 1',
            'CLI capture top-level keys identical; CLI config keys strict subset of library config keys',
            'zero-send refusals: truncating deadline, passed start, foreign clock_domain, NaN reserve, negative '
            'reserve (each persisted by write_capture)',
            'prepare refusals: served_alias mismatch (ready False, refused), heldout purpose',
            'unready and already-used PreparedCapture raise ClientRefusal; first envelope unchanged; nothing sent',
        ],
        'test_library_cancellation_and_failure_keep_every_allocated_offer_after_joining_requests': [
            'three caller cancellations mid-window: CancelledError re-raised; no leftover tasks; envelope '
            'unchanged after sleeping past the third offer schedule; 1 GET + 2 POSTs only',
            'status interrupted, error CancelledError: replay interrupted; fates complete, cancelled response_body, '
            'not_launched cancelled_before_send; rows/summary (offered 3, sent 2); build_run refuses as failure '
            'evidence',
            'transport close failure after all offers: failed envelope returned (not raised) with the complete '
            'record, rows and summary',
        ],
    },
    'static_check': (SCOPE / 'static-check.txt').read_text().splitlines(),
    'attempt_protocol': json.loads((SCOPE / 'PLAN.json').read_text())['attempt_protocol'],
    'attempts': [attempt('attempt01')],
    'failures': [],
    'disclosures': [
        'freeze wording says replay becomes a thin wrapper of _replay(drain_deadline=offered_start + duration + '
        'drain_timeout); the implementation passes drain_timeout and _replay computes offered_end + drain_timeout '
        'as before (identical float, original validation/exception order kept); the CLI passes that same float as '
        'drain_deadline',
        'no join change: in the installed CPython 3.12.16 and 3.14.4 asyncio sources a cancelled gather cancels its '
        'children and completes only after all of them finish, so the unchanged replay finally (cancel all, then '
        'gather) joins every request task under repeated cancellation; executed on 3.12 only (test 2)',
        'the non-cancel failure with allocated offers is triggered by a transport whose aclose raises (per-request '
        'exceptions are recorded by _request and never escape)',
        'not executed: the offered_start-passed refusal inside _replay after records are allocated (needs the clock '
        'to pass the start between the post-/v1/models check and _replay), KeyboardInterrupt and other '
        'BaseException paths of capture_replay',
        'discrimination is reasoned, not mutation-executed: the original lacks the API (AttributeError); a nested '
        'asyncio.run raises RuntimeError (status failed, not collected); unjoined tasks leave tasks/sends; '
        'retiming breaks exact start/scheduled_at/drain_deadline equalities',
        'signal check samples getsignal every 20 ms during the capture; a handler installed and removed between '
        'samples would not be seen (no handler code exists in the library path)',
        'CLI parity checked as envelope key sets plus the 41 unchanged CLI/SSE/SIGINT tests, not value equality',
        'py3.14: ast.parse only (preregistered); the 87x2 suites were not rerun',
        'test timing margins: 0.2-0.3 s start lead, cancellation 0.3 s before the third offer',
    ],
    'claims_not_made': [
        'no Native inference, GPU, SSH, network or real-service HTTP',
        'no full wall at client return (driver-owned), no control/comparability verification',
        'no qualification, admission-boundary, apply/reobserve, performance or winner evidence',
        'no Train selection, Eval access or FIFO/FCFS claim',
        'no independent acceptance',
    ],
    'remaining_blockers': {
        'next_gate': 'independent Role4 recorder+metrics review of this sealed client with unchanged metrics02 '
                     '(nominal_metrics.py 49c76b8b...); no GPU startup before it passes',
        'caller_obligations': [
            'choose offered_start with lead for the GET /v1/models preflight (start still future afterwards, else '
            'refused with zero sends)',
            'persist prepared.capture with write_capture in every outcome, including refusal and after catching '
            'CancelledError/KeyboardInterrupt',
            'own signals/cancellation and the full wall (actual startup to final drain) outside capture_replay',
            'one PreparedCapture per capture (single use)',
        ],
        'unchanged_from_metrics02_report': [
            'driver comparability record + full_wall + static control evidence before Train fixed selection',
            'train_selection_sha256 not recorded by the client: Eval train_selected_fixed alias stays REFUSED '
            '(UNVERIFIED)',
            'gateway owner ledger join by x-request-id, gateway zero-fallback check, FIFO/FCFS backend verification '
            '(driver/Mayor)',
            'A8 adapter public-alias vs local-asset restriction (adapter path only, deferred)',
        ],
    },
}
target = SCOPE / 'report.json'
with open(target, 'x') as handle:
    handle.write(json.dumps(report, indent=1, allow_nan=False) + '\n')
print('report.json', sha(target), 'definitions', json.dumps(definitions, sort_keys=True))
