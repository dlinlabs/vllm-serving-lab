"""Writes report.json for .runtime/r24-resume-p0-nominal-client01 once ('x' mode); constants come from the tested module."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
SCOPE = ROOT / '.runtime' / 'r24-resume-p0-nominal-client01'
RESUME = ROOT / 'experiments' / 'mpc-research' / 'resume-mvp'
sys.path[:0] = [str(RESUME), str(ROOT)]
import nominal_client as nc  # noqa: E402


def pin(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


attempts = {}
for tag in ('py314', 'py312'):
    attempt = SCOPE / 'attempts' / f'attempt01-{tag}'
    attempts[f'attempt01-{tag}'] = {
        'run_sh': f'attempts/attempt01-{tag}/run.sh', 'command': (attempt / 'run.sh').read_text(),
        'started_utc': (attempt / 'started.txt').read_text().strip(),
        'ended_utc': (attempt / 'ended.txt').read_text().strip(),
        'exit_code': int((attempt / 'exit_code.txt').read_text()),
        'result_line': (attempt / 'log.txt').read_text().strip().splitlines()[-1],
        'source_pins_equal_before_after': (attempt / 'pins-before.sha256').read_bytes()
        == (attempt / 'pins-after.sha256').read_bytes(),
        'tested_sha256': {name: digest for digest, name in
                          (line.split() for line in (attempt / 'pins-before.sha256').read_text().splitlines())},
    }
dry = {tag: json.loads((SCOPE / 'dry' / f'dry-{tag}.json').read_text()) for tag in ('py314', 'py312')}
dry_summary = {name: {key: entry.get(key) for key in ('sha256', 'sha256_match', 'validate_trace', 'model', 'n_requests',
                                                      'duration_seconds', 'output_cap', 'provenance_purpose_equals_declared',
                                                      'finalizable_core_purpose', 'provenance_keys')}
               | {'tape_kind': entry['tape']['kind'], 'binding': 'accepted' if isinstance(entry['check_binding'], dict)
                  else entry['check_binding']}
               for name, entry in dry['py314']['inputs'].items()}

report = {
    'schema': 'r24-resume-p0-nominal-client01-report-v1',
    'status': 'SEALED_FIRST_CHECKPOINT_CPU_ONLY',
    'claim_scope': 'CPU-only literal controls of an isolated native replay REQUEST recorder. NO native inference, live '
                   'server/socket HTTP, GPU, permit, budget mutation, qualification, winner, capacity or SLO guarantee.',
    'model': {'id': 'claude-opus-5.5', 'reasoning_effort': 'max', 'role': 'Role7 fair benchmark metrics / negative cases',
              'child_agents': 0},
    'scope': '.runtime/r24-resume-p0-nominal-client01', 'started_utc': (SCOPE / 'STARTED.txt').read_text().strip(),
    'owned_files': [pin(RESUME / 'nominal_client.py'), pin(RESUME / 'test_nominal_client.py')],
    'read_only_runtime_dependencies': [pin(path) for name, path in nc.SOURCES.items() if name != 'nominal_client.py'],
    'frozen_before_tests': ['PLAN.json', 'SELECTION.json', 'SELECTION-AMENDMENT-01.json', 'SELECTION-AMENDMENT-02.json',
                            'SELECTION-AMENDMENT-03.json', 'SCHEMA-FREEZE.json', 'SCHEMA-FREEZE-AMENDMENT-01.json'],
    'frozen_before_dry_check': ['SELECTION-AMENDMENT-04.json'],
    'source_reads': {'viewed': '7 SELECTION files + telemetry.has_sse_content span (A01/A02) + nominal_protocol.json (A03)',
                     'viewed_bytes_upper_bound': 141416, 'cap_bytes': 262144,
                     'reserved_not_read': ['.runtime/r24-resume-p0-nominal-runtime01/REPORT.json',
                                           '.runtime/r24-resume-p0-nominal-runtime01/INTERFACE.json'],
                     'parse_only_never_viewed': ['Train replay 5e418b89...', 'Mayor qualification.json fe480796...',
                                                 'Mayor warmup.json 3a893f48...'],
                     'all_selection_pins_rechecked_unchanged_at_report_time': True,
                     'not_read': ['any Eval tape/replay/outcomes', 'gateway.py', 'trace_control.py',
                                  'peer unsealed runtime/planner/permit/scenario sources']},
    'api': {
        'python': [
            'check_binding(trace, binding) -> argv evidence | ClientRefusal (pure, no I/O)',
            'check_served_models(binding, models_json, max_model_len) -> {ids, alias_root, alias_max_model_len} | ClientRefusal',
            'check_tokenizer(trace, binding, tokenizer) -> {fingerprint, prompts_verified} | ClientRefusal',
            'StreamParse(): feed(line, monotonic_at) per received SSE line; result() -> per-request evidence (pure)',
            'prepare_records(trace, offered_start, run_id) -> one not_launched placeholder per offered request, before any send',
            'await replay(client, trace, records, *, gateway_url, offered_start, drain_timeout) -> drain deadline; records '
            'filled in place; passed offered_start refused before any send',
            'normalized_row(record) -> strict nominal_metrics row (row_mapping)',
            'summarize_records(records) -> fate / finish-evidence / usage-evidence / prompt-token-match counts, max send lag',
            'clock_domain() -> identity string; drivers should call it so full_wall/control_evidence strings match exactly',
            'build_run(capture, *, control, control_missing, control_evidence, full_wall, capture_sha256) -> '
            'nominal_metrics run (validate_run passed) | ClientRefusal',
            'main(argv=None, *, transport=None, tokenizer_loader=load_tokenizer) -> exit code (injection for CPU controls only)',
        ],
        'cli_replay': 'nominal_client.py replay --trace T --binding B --gateway-url http://127.0.0.1:P --backend-url '
                      'http://127.0.0.1:P --arm {%s} --purpose {train,eval,<raw engineering id>} --output-dir NEW '
                      '[--offered-start ABS_MONOTONIC | --start-delay S(0.2)] [--drain-timeout S(120)] '
                      '[--request-timeout S(120)]' % ','.join(nc.nm.RUN_ARMS),
        'cli_finalize': 'nominal_client.py finalize --capture NEW/capture.json --output-dir NEW2 [--control-json C '
                        '[--control-evidence-json E]] [--control-missing REASON] [--full-wall-json W]',
        'outputs': {'replay': 'NEW/capture.json (new 0o700 dir, exclusive create, fsync) + stdout {status, error, '
                              'capture_sha256, summary}',
                    'finalize': 'NEW2/run.json + NEW2/metrics.json (nominal_metrics.compute) + stdout {FINALIZED, sha256, '
                                'summary}; refusal -> stderr {REFUSED, error}, exit 2, no output dir created'},
        'exit_codes': {'0': 'replay collected / finalize written',
                       '2': 'refused: input, binding, served models, tokenizer, purpose, START, output dir, finalize '
                            '(argparse errors also exit 2 before any output)',
                       '3': 'interrupted (SIGINT/SIGTERM/cancel): capture written with every offered record',
                       '4': 'failed (internal error): capture written, records kept when allocated'},
        'capture_schema': nc.SCHEMA, 'capture_statuses': list(nc.STATUSES),
    },
    'request_payload': 'model=trace.model (frozen public alias, never changed), one user message=trace prompt, '
                       'temperature 0, max_tokens=trace max_tokens, stream true, stream_options.include_usage true, '
                       'ignore_eos false; header x-benchmark-request-id=uuid5(run_id, trace_id); response x-request-id kept',
    'row_mapping': nc.ROW_MAPPING,
    'fates': {
        'not_launched': 'never sent (fate_reason not_sent | cancelled_before_send | implementation_exception_before_send)',
        'transport_error': 'httpx.HTTPError before or after headers (incl. request-timeout)',
        'rejected': 'HTTP 503 whose strict JSON is the gateway admission body (detail + estimated_request_cost)',
        'http_error': 'any other non-2xx (fate_reason http_<status>)',
        'stream_error': '2xx stream with any anomaly (fate_reason = first by precedence)',
        'complete': '2xx stream with [DONE], content and no anomaly (finish_reason may be stop/length/other)',
        'cancelled': 'still in flight at offered_end + drain_timeout, or interrupted after send',
        'unknown': 'any other exception after send (fate_reason implementation_exception)',
        'late_errors': 'errors after an outcome was decided keep the outcome; message prefixed after_outcome:',
    },
    'anomaly_precedence': list(nc.ANOMALIES),
    'finish_reason_evidence': {'observed': 'exactly one distinct finish_reason and no content after it (same-chunk '
                                           'content+finish is NOT a contradiction)',
                               'absent': 'no finish_reason (complete fate possible; natural SLO not good)',
                               'contradiction': 'two distinct values or content after finish -> stream_error',
                               'not_applicable': 'no 2xx stream'},
    'natural_slo': 'unchanged native-r23 score_http through sealed nominal_metrics (status complete, finish stop, '
                   'ignore_eos false, [DONE], no stream error, tokens>=1, TTFT<=0.5 s from scheduled, TPOT<=0.05 s); '
                   'no TTFT-only substitution; finish_reason only from actual SSE JSON',
    'model_binding': {
        'schema': nc.BINDING_SCHEMA, 'keys': sorted(nc.BINDING_KEYS),
        'sample': {'schema': nc.BINDING_SCHEMA, 'served_alias': 'Qwen/Qwen3-4B-Instruct-2507',
                   'local_model_path': '/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507',
                   'tokenizer_sha256': '4daabcf7fab3bce457d30dbd14e7eb9c5e79d77c15916f6abdb85b7b8bf0d2c5',
                   'served_model_names': ['Qwen/Qwen3-4B-Instruct-2507'],
                   'backend_argv': ['<actual captured argv, e.g. from /proc/<pid>/cmdline>'],
                   'backend_argv_source': '<driver capture method>'},
        'verified_by_recorder': 'alias == frozen trace.model; absolute normalized local path; tokenizer_sha256 == trace '
                                'provenance; unique served names incl. alias; argv launches the local path (serve <p> | '
                                '--model <p>) with exactly one space-form --served-model-name == names; backend '
                                '/v1/models ids == names, alias root == local path, alias max_model_len == 8192; '
                                'tokenizer loaded from local path matches fingerprint and every prompt token count',
        'default': 'no binding file -> refused (unbound alias), zero sends',
        'recorded_not_claimed': '--scheduling-policy value, prefix-caching flags, --max-model-len value',
        'assumption': 'vLLM /v1/models ModelCard.root equals the launch model argument (fail-closed: mismatch refuses)',
        'not_verified_by_recorder': 'gateway MODEL_NAME, gateway tokenizer_available, zero character fallback, CostSpec '
                                    'identity, other required backend settings (driver-owned)',
    },
    'driver_owned': list(nc.NOT_CLIENT_OWNED),
    'driver_records': {
        'offered_start': '--offered-start = shared absolute time.monotonic() START (same host); must be in (now, now+600 s] '
                         'when preflight ends, else refused with zero sends (never retimed)',
        'control_json_fixed': '{"kind": "static_fixed", "budget": 16.0|32.0 (must equal arm), "basis": "<install method>"}',
        'control_evidence_json': '{"acknowledged_at" < offered_start, "held_verified_at" >= max(offered_end, last '
                                 'terminal, collection_end), "clock_domain" == capture, "source"}',
        'full_wall_json': '{"start" <= offered_start (actual arm setup start), "end" >= max(offered_end, last terminal, '
                          'collection_end) (after final drain/cleanup), "clock_domain" == capture, "source"}; absent -> '
                          'offered_cohort_observed envelope and full_wall_s null',
        'control_missing': 'reason for non-fixed arms without a Parent-accepted measured_intervals control (overhead '
                           'NULL); fixed arms always refuse without static_fixed control',
    },
    'qualifiers': list(nc.QUALIFIERS),
    'raw_engineering_purpose': {
        'rule': nc.RAW_PURPOSE_RULE,
        'behaviour': 'replay records every offered fate like a core run; build_run/finalize refuse: never Train '
                     'selection, Eval performance or a metrics run; no metrics fabricated',
        'cli_refused_examples': ['heldout', 'eval_probe', 'Train'],
        'mayor_probe_inputs_dry_check': 'qualification.json and warmup.json: strict load, validate_trace, purpose '
                                        'equality and literal binding accepted on both Pythons (parse-only)',
        'limit': 'capture.trace_provenance keeps TAPE_KEYS only; probe keys derivation, '
                 'not_a_held_out_or_performance_comparison, probe, source_request_ids, source_train_replay_sha256 are '
                 'recoverable through trace_sha256 + config.trace_path, not copied',
    },
    'tests': {
        'file': 'experiments/mpc-research/resume-mvp/test_nominal_client.py', 'per_python': 40,
        'kind': 'literal CPU controls: in-process httpx.MockTransport, fake tokenizer; no server/socket/GPU/network',
        'controls': [
            'SSE: stop stream with non-data lines and same-chunk finish; 15 literal edge cases (length, finish absent, '
            'two finish values, content after finish, error event, invalid JSON, NaN constant, null choices, missing '
            '[DONE], empty content, data after [DONE], multiple usage, bool usage, choice index 1, precedence)',
            'replay: 8 offered fates retained with exact payload/header/URL and absolute schedule; metrics count all 8',
            'blocked event loop -> late sends, lag recorded, never early or retimed',
            'passed START -> refused with zero sends; external cancel -> unsent kept not_launched/cancelled_before_send',
            'literal timed records -> unchanged natural SLO: good 1 of 3 all-offered, attainment 1/3, groups 1/1/1',
            'binding: alias+local path accepted (serve and --model forms); 13 refusals; /v1/models and tokenizer refusals',
            'CLI fixed16: replay + finalize with driver control evidence and full wall; 5 finalize refusals create no '
            'output; repeated finalize refused without overwrite',
            'CLI refusals: missing binding (0 sends), HF-cache root (models GET only), purpose relabel (0 sends), '
            'existing output dir untouched, argparse heldout/eval_probe/Train/nan/inf',
            'raw engineering capture collected, finalize refused',
            'SIGINT mid-stream -> exit 3, cancelled/response_body + not_launched/cancelled_before_send kept, handler '
            'restored, build_run refuses',
        ],
        'attempts': attempts, 'repairs': 'none: attempt01 passed on both Pythons, no retries',
        'not_run': 'no mutation testing, no old suites, no live server',
    },
    'dry_check': {'script': 'dry/dry_check.py', 'commands': ['dry/command-py314.txt', 'dry/command-py312.txt'],
                  'exit_codes': {tag: int((SCOPE / 'dry' / f'exit-{tag}.txt').read_text()) for tag in ('py314', 'py312')},
                  'py314_equals_py312': dry['py314']['inputs'] == dry['py312']['inputs'], 'inputs': dry_summary},
    'deviations_from_frozen_schema': [
        'finalize output option is --output-dir (frozen text said --output)',
        'trace loaded with strict JSON + trace_workload.validate_trace (not load_trace); SSE/admission JSON strict: '
        'duplicate keys and NaN/Infinity -> invalid_sse (fail-closed, stricter)',
        'connection pool = offered count + 1 (the extra for the backend /v1/models GET)',
        'pre-existing --output-dir refused (exit 2, nothing written)',
    ],
    'gaps_and_limits': [
        'no live vLLM/gateway SSE validated: chunk/usage/header shapes are literal fakes until a Mayor native run',
        'prompt_tokens_match recorded, not enforced (evidence, not a fate)',
        'backend settings beyond the binding (fcfs, prefix caching off, max_num_seqs, max_num_batched_tokens, '
        'gpu_memory_utilization, enforce_eager, TP) are recorded or driver-owned, NOT enforced; FIFO label pending',
        'gateway MODEL_NAME/tokenizer_available/zero fallback/CostSpec/final drain/warmup/budget install: driver-owned',
        'coordinator journal -> measured_intervals mapping NOT defined here (runtime01 reserved, not read): non-fixed '
        'arms finalize only with control_missing (overhead NULL) until a Parent-accepted mapping',
        'client clocks are HTTP observations: sent_at before connection acquisition, terminal_at after close; '
        'event-loop lag recorded per request (max_send_lag_s), no acceptance bound predeclared here',
        'A8: sealed nominal_metrics01 _model_fidelity (run_trace_pilot adapter path) requires absolute trace.model; '
        'WRONG per Mayor for alias launches; needs Parent-reviewed amendment; does not affect this recorder',
        'interrupted/refused/failed captures are preserved evidence and never scored',
    ],
    'mayor_decisions_retained': [
        'FIFO label pending; static32 FCFS only an alias after actual config verification; no 4th algorithm',
        '+inf missing-first TTFT is a failure-penalized operational estimand, not a survival quantile',
        'SLO attainment fraction (good/all offered) and goodput (good/s) stay distinct',
        'hashes are integrity pins, not Train/Eval disjointness proof; Eval never read; Train is in-sample only',
        'Train fixed 16 vs 32 chosen empirically on Train by natural-SLO goodput with predeclared ties',
        'fixed caps static before offers; no solver loop or fake MPC overhead; full wall from the driver',
        'public alias + local path binding is legitimate; unbound alias refused; gateway fallback 0 checked by driver',
        'no G06/G16 pass, Native14 qualification, winner, capacity or SLO guarantee',
    ],
}
with open(SCOPE / 'report.json', 'x') as handle:
    json.dump(report, handle, indent=1, sort_keys=True, allow_nan=False)
    handle.write('\n')
print(json.dumps({'status': report['status'], 'owned_files': report['owned_files'],
                  'attempts': {key: (value['exit_code'], value['result_line']) for key, value in attempts.items()}}))
