"""W01 literal CPU witness for candidate finding N1 (Role4 independent02, preregistered in PLAN.json).

Claim under test: nominal_client._strict_json refuses NaN/Infinity TOKENS but an overflow literal such as 1e999
parses to +/-inf; _admission_rejection copies max_admitted_cost verbatim (only estimated_request_cost is
sanitized), and _write_new (allow_nan=False) then raises ValueError before creating the file.

Executes ONLY the named functions/constants AST-extracted from the pinned snapshot inputs/nominal_client.py; the
module itself is never imported (no httpx, telemetry, tokenizer, socket use or network).  Literal inputs only; no
Native, Train or Eval data.  Writes only inside witness/out-<tag>/ (created fresh, exclusive).
"""
import ast
import hashlib
import json
import math
import os
import sys

SNAPSHOT = 'inputs/nominal_client.py'
SNAPSHOT_SHA256 = '933e4161d4eadd7b9730f4d122387aad15932343bcc18de7580e9ea92b86fe95'
FUNCTIONS = ('_finite', '_reject_constant', '_strict_json', '_admission_rejection', '_write_new')
CONSTANTS = ('ADMISSION_DETAIL', 'GATEWAY_REJECTION_FIELDS')
TAGS = ('py314', 'py312')
DETAIL = 'request rejected: admission capacity exceeded'


def extract():
    data = open(SNAPSHOT, 'rb').read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SNAPSHOT_SHA256:
        raise SystemExit(f'snapshot sha256 mismatch: {digest}')
    tree = ast.parse(data.decode('utf-8'), filename=SNAPSHOT)
    picked, spans = [], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            name = node.name
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in CONSTANTS):
            name = node.targets[0].id
        else:
            continue
        picked.append(node)
        spans.append([name, node.lineno, node.end_lineno])
    names = [span[0] for span in spans]
    if sorted(names) != sorted(FUNCTIONS + CONSTANTS):
        raise SystemExit(f'extraction mismatch (missing or duplicated top-level names): {names}')
    namespace = {'json': json, 'math': math, 'hashlib': hashlib, 'os': os}
    exec(compile(ast.Module(body=picked, type_ignores=[]), SNAPSHOT, 'exec'), namespace)
    return namespace, digest, spans


def body(max_admitted_cost='32.0', estimated_request_cost='40.0'):
    return ('{"detail": "' + DETAIL + '", "estimated_request_cost": ' + estimated_request_cost
            + ', "estimated_input_tokens": 100, "estimated_output_tokens": 64, "max_admitted_cost": '
            + max_admitted_cost + ', "budget_version": 3}').encode()


def attempt(function):
    try:
        return {'raised': None, 'value': function()}
    except Exception as exc:
        return {'raised': type(exc).__name__, 'message': str(exc)[:200], 'value': None}


def main():
    tag = sys.argv[1] if len(sys.argv) == 2 else None
    if tag not in TAGS:
        raise SystemExit(f'usage: w01_strict_json_overflow.py {{{"|".join(TAGS)}}}')
    out_dir = os.path.join('witness', f'out-{tag}')
    os.mkdir(out_dir, 0o755)
    ns, digest, spans = extract()
    strict, rejection_of, write_new = ns['_strict_json'], ns['_admission_rejection'], ns['_write_new']
    finite_path = os.path.join(out_dir, 'finite_control.json')
    overflow_path = os.path.join(out_dir, 'overflow.json')
    cases = []

    def case(case_id, function, expectation, check):
        observed = attempt(function)
        try:
            met = bool(check(observed))
        except Exception as exc:
            met = False
            observed['check_error'] = f'{type(exc).__name__}: {exc}'
        shown = dict(observed)
        shown['value'] = repr(observed['value'])
        cases.append({'id': case_id, 'expected': expectation, 'observed': shown, 'expectation_met': met})
        return observed['value']

    case('C1_strict_overflow_positive', lambda: strict('{"a": 1e999}', 'W01'), 'no exception; a == +inf',
         lambda o: o['raised'] is None and math.isinf(o['value']['a']) and o['value']['a'] > 0)
    case('C2_strict_overflow_negative', lambda: strict('{"a": -1e999}', 'W01'), 'no exception; a == -inf',
         lambda o: o['raised'] is None and math.isinf(o['value']['a']) and o['value']['a'] < 0)
    case('C3_strict_infinity_token_control', lambda: strict('{"a": Infinity}', 'W01'), 'ValueError',
         lambda o: o['raised'] == 'ValueError')
    case('C4_strict_nan_token_control', lambda: strict('{"a": NaN}', 'W01'), 'ValueError',
         lambda o: o['raised'] == 'ValueError')
    finite = case('C5_rejection_finite_control', lambda: rejection_of(503, body()),
                  'dict; max_admitted_cost == 32.0; estimated_request_cost == 40.0; fields_absent == []',
                  lambda o: o['raised'] is None and o['value']['max_admitted_cost'] == 32.0
                  and o['value']['estimated_request_cost'] == 40.0 and o['value']['fields_absent'] == [])
    overflow = case('C6_rejection_overflow_max_admitted_cost', lambda: rejection_of(503, body(max_admitted_cost='1e999')),
                    'dict (classified as admission rejection); max_admitted_cost == +inf copied verbatim',
                    lambda o: o['raised'] is None and math.isinf(o['value']['max_admitted_cost'])
                    and o['value']['estimated_request_cost'] == 40.0)
    case('C7_rejection_overflow_cost_sanitized_control',
         lambda: rejection_of(503, body(estimated_request_cost='1e999')),
         'dict; estimated_request_cost is None (sanitized); max_admitted_cost == 32.0',
         lambda o: o['raised'] is None and o['value']['estimated_request_cost'] is None
         and o['value']['max_admitted_cost'] == 32.0)

    def write_finite():
        returned = write_new(finite_path, {'rejection': finite})
        with open(finite_path, 'rb') as handle:
            return {'returned': returned, 'file_sha256': hashlib.sha256(handle.read()).hexdigest()}

    case('C8_write_new_finite_control', write_finite, 'file created; returned sha256 == sha256(file bytes)',
         lambda o: o['raised'] is None and o['value']['returned'] == o['value']['file_sha256'])
    case('C9_write_new_overflow', lambda: write_new(overflow_path, {'rejection': overflow}),
         'ValueError raised before open; no file created',
         lambda o: o['raised'] == 'ValueError' and not os.path.lexists(overflow_path))
    result = {
        'witness': 'W01', 'tag': tag, 'python': sys.version, 'executable': sys.executable,
        'snapshot': SNAPSHOT, 'snapshot_sha256': digest, 'extracted_spans': spans, 'cases': cases,
        'all_expectations_met': all(item['expectation_met'] for item in cases),
    }
    print(json.dumps(result, sort_keys=True, indent=1, allow_nan=False))
    return 0 if result['all_expectations_met'] else 1


if __name__ == '__main__':
    sys.exit(main())
