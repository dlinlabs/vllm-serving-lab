"""Write report.json for the metrics02 scope from recorded attempt files and literal findings (run once)."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/daniel_lin/gt/pr32/mayor/mpc-local-r21')
SCOPE = ROOT / '.runtime/r24-resume-p0-nominal-metrics02'
SRC = 'experiments/mpc-research/resume-mvp'
OWNED = ('nominal_metrics.py', 'nominal_client.py', 'test_nominal_metrics.py', 'test_nominal_client.py')
DEPS = ('trace_workload.py', 'telemetry.py', 'experiments/mpc-research/native-r23/score_http.py', 'gateway.py')
FROZEN = ('STARTED.txt', 'ORIGINALS.sha256', 'CLIENT01-DISPOSITION.json', 'SELECTION.json', 'SELECTION-REPORT-PIN.txt',
          'SELECTION-AMENDMENT-01.json', 'SELECTION-AMENDMENT-02.json', 'PLAN.json', 'SCHEMA-FREEZE.json',
          'PLAN-AMENDMENT-01.json', 'PLAN-ERRATA-01.json', 'pytest-empty.ini', 'run_attempt.sh')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return Path(path).read_bytes().count(b'\n')


def attempt(name):
    base = SCOPE / 'attempts' / name
    read = lambda file: (base / file).read_text().strip()  # noqa: E731
    pinned = {file: sha(base / 'pinned' / file) for file in OWNED}
    return {
        'dir': f'attempts/{name}', 'run_sh': read('run.sh').splitlines()[-1], 'run_sh_sha256': sha(base / 'run.sh'),
        'started': read('started.txt'), 'ended': read('ended.txt'), 'exit_code': int(read('exit_code.txt')),
        'result_line': read('log.txt').splitlines()[-1], 'log_sha256': sha(base / 'log.txt'),
        'pins_before_equal_after_and_pycache_unchanged': read('unchanged.txt') == 'unchanged_exit=0',
        'pinned_copies_equal_final_sources': all(pinned[file] == sha(ROOT / SRC / file) for file in OWNED),
    }


originals = {line.split()[1]: line.split()[0] for line in (SCOPE / 'ORIGINALS.sha256').read_text().splitlines()[:4]}
sources = {}
for file in OWNED:
    assert sha(SCOPE / 'originals' / file) == originals[file]
    sources[f'{SRC}/{file}'] = {'original_sha256': originals[file], 'final_sha256': sha(ROOT / SRC / file),
                                'final_lines': lines(ROOT / SRC / file),
                                'original_lines': lines(SCOPE / 'originals' / file)}
attempts = [attempt('attempt01-py314'), attempt('attempt01-py312')]
attempts[0]['interpreter'] = 'ROOT/.venv/bin/python3.14 (3.14.4; pytest 9.2.0; httpx 0.28.1)'
attempts[1]['interpreter'] = 'files/r24-mpc-py312-venv/bin/python3.12 (3.12.16; pytest 9.2.0; httpx 0.28.1)'

report = {
    'schema': 'r24-resume-p0-nominal-metrics02-report-v1',
    'scope': '.runtime/r24-resume-p0-nominal-metrics02',
    'role': 'PR32 team16 Role7 fair benchmark metrics and negative cases',
    'model': {'id': 'claude-opus-5.5', 'reasoning_effort': 'max'},
    'status': 'SEALED: CPU literal controls 87/87 per Python; NOT accepted until independent Role4 recorder+metrics '
              'review; no Native/GPU/HTTP-to-real-service evidence',
    'objective': 'minimal metrics F1-F5 corrections plus F6/F7 labels from the sealed independent audit, minimal '
                 'client build_run/finalize integration, and verbatim gateway rejection-body retention (Mayor)',
    'audit_input': {'scope': '.runtime/r24-resume-p0-metrics-independent01',
                    'SHA256SUMS_sha256': '43b80aff6fb02be80a3d79b60fe7c56c9f4bbd3a461477344c79cb3acaf093bb',
                    'entries': 39, 'REPORT.json_sha256': 'a3fb13a66a43a0bdf652d82333542e80ad23a22d823247bcb031e8aba40589d3',
                    'read': 'REPORT.json findings only (no audit view logs)'},
    'sources': sources,
    'dependency_pins': {name: sha(ROOT / name) for name in DEPS},
    'dependency_pins_note': 'hash-read only (not viewed) except gateway.py lines 653-692 viewed for the 503 body; '
                            'score_http.py natural rule unchanged (9a173ab9...); pins equal before and after each '
                            'attempt (gateway.py hashed at report time, never imported by these tests)',
    'frozen_scope_files_sha256': {name: sha(SCOPE / name) for name in FROZEN},
    'content_viewed': ['audit REPORT.json', 'ROOT gateway.py lines 653-692', 'the four owned files and their '
                       'originals/', 'client01 attempt directory layout (names only)', 'this scope'],
    'findings': {
        'F1_HIGH_cross_arm_comparability': {
            'fix': 'provenance.comparability (optional; absent = NOT_RECORDED) with exactly hardware, model, backend, '
                   'gateway, resources, code, client_generation, clock; model exactly local_model_path (absolute), '
                   'asset_sha256, served_alias, served_alias_verified (must be true), tokenizer_sha256; '
                   'client_generation.ignore_eos (bool, must equal every row requested_ignore_eos) and '
                   'max_tokens_policy; gateway must not carry per-arm MAX_ADMITTED_COST; canonical JSON. Cross-arm '
                   'identity: trace sha, all-offered count, ordered request-id digest, request-shape digest, '
                   'duration, envelope scope, relative schedule within 1e-6 (absolute starts may differ); every '
                   'measured arm DECLARED with identical per-fact digests, else select_train_fixed REFUSED and '
                   'comparison_table raises',
            'where': ['nominal_metrics._validate_comparability', '_comparability', '_cross_arm_refusal',
                      '_binding_error', 'select_train_fixed', 'comparison_table',
                      'nominal_client.build_run(comparability=)', 'nominal_client._check_comparability',
                      'finalize --comparability-json'],
            'client_cross_check': 'declared model served_alias/local_model_path/tokenizer_sha256 must equal the '
                                  'recorder-verified binding evidence (tokenizer fingerprint) and ignore_eos must be '
                                  'False (what the recorder sends); other facts are driver-declared, compared only',
            'tests': ['test_comparability_facts_validated_and_required_for_selection_and_tables',
                      'test_literal_records_map_to_unchanged_natural_slo_with_all_offered_denominator',
                      'test_cli_replay_then_finalize_fixed_arm_only_with_driver_evidence']},
        'F2_MEDIUM_selection_binding_and_reuse': {
            'fix': 'SELECTED record v2: schema, purpose train, decided_by, rule, candidates with metrics_sha256, '
                   'trace sha, ordered request ids + digest, comparability_sha256, self digest selection_sha256. '
                   'Tables verify exact keys, digest and recomputed decision; Train alias requires candidates to be '
                   'the measured fixed arms; Eval alias refuses any positive reuse signal (same trace sha, same id '
                   'digest, any shared request id), requires the same comparability sha, and every measured Eval arm '
                   'must record the same train_selection_sha256 (missing: alias REFUSED UNVERIFIED; different: '
                   'raise); alias row carries selection sha, decided_by and candidates',
            'where': ['select_train_fixed', '_decide', '_verify_selection', '_alias_row', 'comparison_table'],
            'tests': ['test_selection_record_binding_and_eval_alias_reuse_refusals',
                      'test_train_selection_primary_ties_and_refusals',
                      'test_comparison_table_alias_refusals_and_no_winner'],
            'limits': 'digests bind declared records only; a self-consistent record can still be fabricated; passing '
                      'reuse checks is not Train/Eval disjointness proof'},
        'F3_early_send': {'fix': 'sent_at < scheduled_at refused (EARLY_SEND_TOLERANCE_S = 0.0); send_lag_s adds min, '
                                 'n_negative, early_send_tolerance_s; summary carries min and n_negative',
                          'where': ['_validate_row', 'compute', 'summarize'],
                          'tests': ['test_early_send_is_refused_and_lag_extremes_reported']},
        'F4_offered_cohort_binding': {
            'fix': 'provenance.offered_count (= len(rows)) and offered_request_ids_sha256 (canonical JSON of ids in '
                   'frozen trace order; must equal row-order digest) required; compute exposes offered_cohort; client '
                   'replay records the digest from the frozen trace; adapter derives both from the trace',
            'where': ['request_ids_sha256', '_validate_cohort', '_offered_cohort', 'nominal_client._replay_main',
                      'nominal_client.build_run', 'adapt_run_trace_pilot provenance'],
            'tests': ['test_all_offered_cohort_is_bound_to_offered_count_and_ordered_ids',
                      'test_literal_records_map_to_unchanged_natural_slo_with_all_offered_denominator']},
        'F5_window_and_envelope_binding': {
            'fix': 'every scheduled_at within the offered window; every sent/first/terminal within the declared '
                   'envelope; offered_cohort_observed must equal [offered_start, max(offered_end, last terminal)] '
                   'within 1e-6, so an envelope ending before the last terminal is refused',
            'where': ['_validate_windows'],
            'tests': ['test_offered_window_and_declared_envelope_bind_every_observation']},
        'F6_LOW_group_labels': {'fix': 'per-group n_offered_by_group, k, Jain note; worst rejection group identity, n '
                                       'and tied groups; summary carries them',
                                'where': ['_spread', '_fairness', 'summarize'],
                                'tests': ['test_group_identity_ties_and_rate_scope_labels']},
        'F7_LOW_rate_labels': {'fix': 'scope-dependent wall label and direction (offered_cohort_observed: lower-bound '
                                      'envelope, rates are UPPER bounds on full-wall rates); delivered-only token scope '
                                      'with the endpoint max_tokens premise',
                               'where': ['ENVELOPE_LABELS', 'WALL_RATE_DIRECTION', 'TOKEN_RATE_SCOPE', '_throughput',
                                         '_natural_slo', 'summarize'],
                               'tests': ['test_group_identity_ties_and_rate_scope_labels']},
        'F8_F9': 'not in scope: Mayor dispositions forecaster findings; scenario/planner not modified',
    },
    'rejection_retention': {
        'source_of_truth': 'ROOT gateway.py 0b1a80ad... lines 653-692 (503 body + x-request-id header)',
        'classification_rule': 'unchanged: 503 + exact detail + estimated_request_cost key',
        'retained': 'detail, estimated_request_cost (finite else null, unchanged), verbatim estimated_input_tokens, '
                    'estimated_output_tokens, max_admitted_cost, budget_version (null when absent) and fields_absent; '
                    'record.request_id = x-request-id; every non-success body keeps body_bytes, body_sha256 and '
                    'body_text (strict UTF-8 and <= 65536 bytes) else body_text_null_reason not_utf8 | '
                    'over_65536_bytes | body_read_incomplete (read failure, fate transport_error)',
        'not_recorded': 'no auth header sent or stored; gateway admitted/terminal cost/budget/version telemetry and '
                        'the ordered owner ledger are gateway/driver-owned and joined by x-request-id (Mayor)',
        'tests': ['test_rejection_body_retention_is_verbatim_and_never_invented',
                  'test_replay_retains_every_fate_payload_and_absolute_schedule'],
        'schema': 'client capture r24-resume-p0-nominal-client-v2; v1 captures are not finalized by v2 code'},
    'unchanged': ['score_http natural SLO rule and pin', 'fates, all-offered denominators and quantile estimands '
                  '(missing-first +inf = failure-penalized operational TTFT, not a survival quantile)',
                  'selection rule and tie order', 'request sending, absolute schedule, SSE parsing and row mapping',
                  'raw engineering-purpose captures (never finalized as Train/Eval)', 'FIFO label pending',
                  'A8 adapter public-alias restriction (deferred)'],
    'tests': {'collected_per_python': 87, 'test_functions': {'test_nominal_metrics.py': '12 -> 18 (6 new)',
                                                             'test_nominal_client.py': '14 -> 15 (1 new)'},
              'per_file_split': 'not separately collected', 'removed_tests': [],
              'discrimination': 'NOT run: new regressions were not executed against the original metrics01/client01 '
                                'bytes and no mutation controls were run (bounded delivery); independent review may'},
    'attempt_protocol': 'scripted by run_attempt.sh: one generation per Python, no retries, failures preserved, '
                        'pins before/after, pinned copies, isolated basetemp; command in each attempts/*/run.sh',
    'attempts': attempts,
    'failures': 'none in attempt01; PLAN.json strict-JSON defect disclosed in PLAN-ERRATA-01.json (original kept)',
    'disclosures': ['PLAN-ERRATA-01.json: PLAN.json is not strict JSON; two client edits beyond the named plan scope '
                    '(_replay_main digest, NOT_CLIENT_OWNED lines)',
                    'client01 seal (0ccdbf13...) stays valid only for its own bytes; its outputs are superseded by '
                    'v2 and are not benchmark or selection evidence'],
    'claims_not_made': ['no Native inference, GPU, SSH, network or real-service HTTP', 'no winner, superiority, '
                        'capacity, SLO guarantee or production claim', 'no G06/G16/Native14 qualification',
                        'comparability facts are driver-declared and compared, not verified here',
                        'hashes are integrity pins, not Train/Eval disjointness or leakage proof',
                        'no FIFO/FCFS backend claim', 'no actual admission-boundary or apply/reobserve evidence'],
    'remaining_blockers': {
        'next_gate': 'independent Role4 recorder+metrics acceptance of this seal; no GPU startup before it PASSES',
        'train_fixed_freeze': 'needs a driver-produced comparability record per arm (local path, asset sha256, '
                              'verified served alias, tokenizer sha, backend/gateway/resources/code/clock, '
                              'client_generation ignore_eos=false + max_tokens policy) plus driver full_wall and '
                              'static control evidence; without it select_train_fixed returns REFUSED (UNVERIFIED)',
        'eval_alias': 'the recorder does not record train_selection_sha256, so the Eval train_selected_fixed alias '
                      'stays REFUSED (UNVERIFIED) until a driver/client successor records it after the Train freeze',
        'driver_or_mayor_owned': ['gateway telemetry/owner ledger join by x-request-id',
                                  'gateway zero-fallback check', 'FIFO/FCFS backend verification',
                                  'F8/F9 forecaster dispositions'],
        'role7_deferred': ['A8 adapter public-alias vs local-asset restriction (adapter path only)']},
}
with open(SCOPE / 'report.json', 'x') as handle:
    json.dump(report, handle, indent=1, sort_keys=False, allow_nan=False)
    handle.write('\n')
print('report.json', sha(SCOPE / 'report.json'))
