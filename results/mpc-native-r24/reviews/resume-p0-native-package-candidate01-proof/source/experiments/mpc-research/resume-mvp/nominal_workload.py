"""Predeclared SYNTHETIC offered-workload tapes (Train/Eval) for future real-GPU measurements.

Protocol v1 values are Mayor protocol choices: they are not verified capacity or model accuracy
and were not designed for an MPC win.  A tape is a synthetic *offered* workload drawn from a
public deterministic generator; it carries no outcomes and is distinct from any Native/GPU
execution of it.  Requested prompt-length classes are labels: backend tokenization and gateway
cost are measured at replay time (see ``to_replay_trace``), never assumed equal to the label.

CLI (no protocol override flags exist):
    python -B -I nominal_workload.py generate --out-dir NEW_DIR
    python -B -I nominal_workload.py verify --out-dir DIR
"""
import argparse
import decimal
import hashlib
import json
from pathlib import Path
import sys

GENERATOR_ID = 'r24-resume-p0-nominal-workload-generator'
GENERATOR_VERSION = 1
MODULE_PATH = 'experiments/mpc-research/resume-mvp/nominal_workload.py'
TAPE_SCHEMA = 'r24-resume-p0-offered-tape-v1'
TAPE_KIND = 'SYNTHETIC_OFFERED_WORKLOAD_TAPE'
MANIFEST_SCHEMA = 'r24-resume-p0-workload-manifest-v1'
MANIFEST_NAME = 'manifest.json'
PURPOSES = ('train', 'eval')
STREAMS = ('arrival', 'length_class', 'payload')
COLUMNS = ('request_id', 'offset_ns', 'requested_prompt_tokens', 'max_output_tokens', 'payload_key')
REPLAY_MAX_MODEL_LEN = 8192  # trace_workload.MAX_MODEL_LEN for replay schema_version 1
NS_PER_SECOND = 1_000_000_000
TWO_64 = 1 << 64
DECIMAL_PRECISION = 50
MAX_SUPPORTED_OFFERS = 10_000  # request IDs use a 4-digit index

_PROTOCOL_V1 = {
    'protocol_id': 'r24-resume-p0-nominal-workload-v1',
    'protocol_version': 1,
    'kind': 'SYNTHETIC_OFFERED_WORKLOAD',
    'duration_ns': 5 * NS_PER_SECOND,
    'arrival_process': 'homogeneous_poisson',
    'arrival_rate_per_second': 20,
    'max_offers': 1024,
    'prompt_length_classes': [256, 1024, 3072],
    'prompt_length_class_distribution': 'uniform',
    'max_output_tokens': 128,
    'later_gpu_config': {'prefix_cache': 'disabled: required for later GPU measurement; not verified by this CPU component'},
    'splits': {
        'train': {'seed': 4101, 'source_id': 'synthetic:r24-resume-p0-nominal-workload:v1:train',
                  'request_id_prefix': 'p0w1-train-'},
        'eval': {'seed': 4102, 'source_id': 'synthetic:r24-resume-p0-nominal-workload:v1:eval',
                 'request_id_prefix': 'p0w1-eval-'},
    },
}

DISCLAIMERS = (
    'SYNTHETIC offered workload from a public deterministic generator: not a real or Azure trace and not a '
    'held-out real-trace sample.',
    'Offered workload only: no execution outcomes; any Native/GPU execution of this tape is a separate measurement.',
    'requested_prompt_tokens are class labels: backend tokenization and gateway cost must be measured, not assumed equal.',
    'Protocol values are protocol choices, not verified capacity or model accuracy, and were not designed for an MPC win.',
)
POLICY = {
    'train': 'Train split: controllers/forecasters may take scenario information only from Train and past observations, '
             'using a forecast RNG distinct from every domain of this generator; Train selection method PENDING Mayor/Role7.',
    'eval': 'Eval split (held out): controllers, forecasters and tuning must never read or regenerate this tape or use its '
            'outcomes; only the measurement harness replays it.',
}
RNG_SPEC = {
    'uniform': 'X(purpose, seed, stream, k) = big-endian uint64 of the first 8 bytes of sha256(domain || k as 8-byte '
               'big-endian); domain = ASCII JSON (separators "," ":") of [generator_id, generator_version, protocol_id, '
               'purpose, seed, stream] followed by one 0x00 byte',
    'streams': list(STREAMS),
    'arrival': 'gap_seconds = (ln(2^64) - ln(2^64 - X)) / rate in decimal.Context(prec=50, ROUND_HALF_EVEN); arrival time '
               't accumulates from 0; offset_ns = floor(t * 10^9); stop at the first offset_ns >= duration_ns; arrival '
               'k = request index; more than max_offers in-window arrivals is refused (no truncation or resampling)',
    'length_class': 'class = prompt_length_classes[(X * n_classes) >> 64] with k = request index (bias <= 2^-64)',
    'payload': 'nonce = 16 lowercase hex digits of X with k = request index',
    'decimal_precision': DECIMAL_PRECISION,
}
PAYLOAD_GENERATION = {
    'key_format': 'protocol_id/purpose/request_id/nonce',
    'template': 'SHARED by both splits at replay-adapter time: trace_workload.synthetic_prompt(tokenizer, '
                'requested_prompt_tokens, payload_key); tapes carry no prompt text',
}
PROMPT_LENGTH_SEMANTICS = (
    'requested_prompt_tokens is a protocol class label (requested chat-template tokens). It is not assumed equal to '
    'backend tokenization or gateway cost: replay input_tokens are measured with the replay tokenizer and gateway cost '
    'is whatever the gateway computes at run time.'
)
HOLDOUT_BASIS = (
    'Generator independence: each tape is a pure function of the shared protocol fields and its own split spec; RNG '
    'domains include purpose and seed, so Train and Eval use disjoint SHA-256 counter domains (tested: Eval bytes are '
    'invariant to Train seed/source/prefix changes and vice versa; equal seeds with different purposes still differ).',
    'Membership: request-ID namespaces and payload keys (which embed purpose) are disjoint; source IDs differ; the '
    'Train tape contains no Eval seed, source or prefix.',
    'Provenance: this manifest pins the protocol, module and both tape SHA-256 values; verify regenerates byte-identically.',
    'Unequal hashes alone are NOT claimed as holdout evidence. Both seeds are public, so this is a procedural holdout of '
    'synthetic draws from one public generator: not an OOD/generalization split and not an Azure or real-trace holdout.',
)
CLAIMS_NOT_MADE = (
    'no CPU or GPU performance result', 'no capacity or model-accuracy claim', 'no OOD/generalization claim',
    'no Azure or real-trace held-out claim', 'no MPC/controller superiority claim',
    'requested prompt length is not claimed equal to backend tokenization or gateway cost',
    'prefix cache disabled is required but not verified here', 'native-client compatibility NOT_VERIFIED',
    'Train selection method not chosen here (PENDING Mayor/Role7)',
)


class OfferCapExceeded(ValueError):
    """More in-window arrivals than max_offers: refused, never truncated or resampled."""


class TapeIntegrityError(ValueError):
    """Tape bytes differ from the frozen generator output."""


class TapePurposeError(ValueError):
    """Tape split purpose differs from the purpose the caller declared."""


class VerificationError(ValueError):
    """Output directory differs from the regenerated outputs."""


def protocol_v1():
    """Fresh copy of the frozen protocol v1 (callers cannot mutate the frozen constant)."""
    return json.loads(json.dumps(_PROTOCOL_V1))


def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii') + b'\n'


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def decimal_context():
    return decimal.Context(prec=DECIMAL_PRECISION, rounding=decimal.ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                           capitals=1, clamp=0, flags=[],
                           traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])


def rng_domain(protocol_id, purpose, seed, stream):
    material = json.dumps([GENERATOR_ID, GENERATOR_VERSION, protocol_id, purpose, seed, stream],
                          separators=(',', ':'), ensure_ascii=True)
    return material.encode('ascii') + b'\x00'


def uniform64(domain, k):
    return int.from_bytes(hashlib.sha256(domain + k.to_bytes(8, 'big')).digest()[:8], 'big')


def _positive_int(value, name, upper=None):
    if type(value) is not int or value <= 0 or (upper is not None and value > upper):
        raise ValueError(f'{name} must be a positive int' + (f' <= {upper}' if upper is not None else ''))
    return value


def exponential_gap_seconds(x, rate, context=None):
    """Exact-decimal exponential(rate) inverse-CDF draw on the uint64 grid."""
    if type(x) is not int or not 0 <= x < TWO_64:
        raise ValueError('x must be an int in [0, 2^64)')
    _positive_int(rate, 'rate')
    ctx = decimal_context() if context is None else context
    numerator = ctx.subtract(ctx.ln(decimal.Decimal(TWO_64)), ctx.ln(decimal.Decimal(TWO_64 - x)))
    return ctx.divide(numerator, decimal.Decimal(rate))


def length_class_index(x, n_classes):
    if type(x) is not int or not 0 <= x < TWO_64:
        raise ValueError('x must be an int in [0, 2^64)')
    _positive_int(n_classes, 'n_classes')
    return (x * n_classes) >> 64


def _nonempty_str(value, name):
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError(f'{name} must be a non-empty ASCII string')
    return value


def _validated(protocol, purpose):
    if purpose not in PURPOSES:
        raise ValueError(f'purpose must be one of {PURPOSES}')
    if not isinstance(protocol, dict):
        raise ValueError('protocol must be a dict')
    _nonempty_str(protocol.get('protocol_id'), 'protocol_id')
    _positive_int(protocol.get('protocol_version'), 'protocol_version')
    if protocol.get('kind') != 'SYNTHETIC_OFFERED_WORKLOAD':
        raise ValueError('protocol kind must be SYNTHETIC_OFFERED_WORKLOAD')
    if protocol.get('arrival_process') != 'homogeneous_poisson':
        raise ValueError('only a homogeneous Poisson arrival process is implemented')
    if protocol.get('prompt_length_class_distribution') != 'uniform':
        raise ValueError('only a uniform prompt-length class distribution is implemented')
    _positive_int(protocol.get('duration_ns'), 'duration_ns')
    _positive_int(protocol.get('arrival_rate_per_second'), 'arrival_rate_per_second')
    _positive_int(protocol.get('max_offers'), 'max_offers', MAX_SUPPORTED_OFFERS)
    cap = protocol.get('max_output_tokens')
    if type(cap) is not int or not 0 < cap < REPLAY_MAX_MODEL_LEN:
        raise ValueError('max_output_tokens must be an int in (0, 8192)')
    classes = protocol.get('prompt_length_classes')
    if (not isinstance(classes, list) or not classes
            or any(type(c) is not int or c <= 0 or c + cap > REPLAY_MAX_MODEL_LEN for c in classes)
            or any(a >= b for a, b in zip(classes, classes[1:]))):
        raise ValueError('prompt_length_classes must be strictly increasing positive ints with class + cap <= 8192')
    splits = protocol.get('splits')
    split = splits.get(purpose) if isinstance(splits, dict) else None
    if not isinstance(split, dict) or set(split) != {'seed', 'source_id', 'request_id_prefix'}:
        raise ValueError(f'protocol needs a {purpose} split with exactly seed, source_id and request_id_prefix')
    if type(split['seed']) is not int or split['seed'] < 0:
        raise ValueError('seed must be a non-negative int')
    _nonempty_str(split['source_id'], 'source_id')
    _nonempty_str(split['request_id_prefix'], 'request_id_prefix')
    shared = {key: value for key, value in protocol.items() if key != 'splits'}
    return json.loads(json.dumps(shared)), dict(split, purpose=purpose)


def generate_split(protocol, purpose):
    """Pure function of the shared protocol fields and this purpose's own split spec."""
    shared, split = _validated(protocol, purpose)
    ctx = decimal_context()
    domains = {stream: rng_domain(shared['protocol_id'], purpose, split['seed'], stream) for stream in STREAMS}
    classes = shared['prompt_length_classes']
    cap = shared['max_output_tokens']
    rows = []
    t = decimal.Decimal(0)
    k = 0
    while True:
        t = ctx.add(t, exponential_gap_seconds(uniform64(domains['arrival'], k), shared['arrival_rate_per_second'], ctx))
        k += 1
        offset_ns = int(ctx.scaleb(t, 9).to_integral_value(rounding=decimal.ROUND_FLOOR, context=ctx))
        if offset_ns >= shared['duration_ns']:
            break
        index = len(rows)
        if index >= shared['max_offers']:
            raise OfferCapExceeded(f'{purpose}: more than max_offers={shared["max_offers"]} arrivals inside the window; '
                                   'refused without truncation or resampling')
        request_id = f'{split["request_id_prefix"]}{index:04d}'
        requested = classes[length_class_index(uniform64(domains['length_class'], index), len(classes))]
        nonce = format(uniform64(domains['payload'], index), '016x')
        rows.append([request_id, offset_ns, requested, cap, f'{shared["protocol_id"]}/{purpose}/{request_id}/{nonce}'])
    return {
        'schema': TAPE_SCHEMA,
        'kind': TAPE_KIND,
        'disclaimers': [*DISCLAIMERS, POLICY[purpose]],
        'generator': {'id': GENERATOR_ID, 'version': GENERATOR_VERSION},
        'protocol': shared,
        'split': split,
        'rng': json.loads(json.dumps(RNG_SPEC)),
        'payload_generation': json.loads(json.dumps(PAYLOAD_GENERATION)),
        'columns': list(COLUMNS),
        'rows': rows,
        'count': len(rows),
    }


def encode_tape(tape):
    return canonical_bytes(tape)


def _module_sha256():
    return sha256_hex(Path(__file__).resolve().read_bytes())


def _replay_adapter_mapping():
    return {
        'consumer': 'run_trace_pilot.py via trace_workload.load_trace/validate_trace/verify_prompts (schema_version 1)',
        'parser_compatible_directly': False,
        'reason': 'schema_version 1 requests require prompt text and tokenizer-measured chat-template input_tokens',
        'function': 'nominal_workload.to_replay_trace(tape, model=..., prompt_for=..., count_tokens=..., '
                    'tokenizer_sha256=...)',
        'field_mapping': {
            'schema_version': 1, 'max_model_len': REPLAY_MAX_MODEL_LEN,
            'model': 'caller supplied (trace_workload.MODEL)',
            'duration_seconds': 'duration_ns / 1e9', 'output_cap': 'max_output_tokens',
            'trace_id': 'request_id', 'arrival_seconds': 'offset_ns / 1e9 (correctly rounded binary64)',
            'prompt': 'prompt_for(requested_prompt_tokens, payload_key)',
            'input_tokens': 'MEASURED count_tokens(prompt); requested label kept in provenance',
            'max_tokens': 'max_output_tokens',
        },
        'recipe_not_executed_here': [
            'import json, sys',
            "sys.path[:0] = ['<ROOT>', '<ROOT>/experiments/mpc-research/resume-mvp']",
            'from transformers import AutoTokenizer',
            'import trace_workload as tw, nominal_workload as nw',
            'tok = AutoTokenizer.from_pretrained(tw.MODEL, local_files_only=True)',
            "tape = nw.load_tape('<tapes-v1>/<purpose>.tape.json', purpose='<purpose>')",
            'trace = nw.to_replay_trace(tape, model=tw.MODEL, prompt_for=lambda n, key: tw.synthetic_prompt(tok, n, key), '
            "count_tokens=lambda text: tw.input_tokens(tok, [{'role': 'user', 'content': text}]), "
            'tokenizer_sha256=tw.tokenizer_fingerprint(tok))',
            'tw.validate_trace(trace); tw.verify_prompts(trace, tok)',
            "open('<new replay trace path>', 'x').write(json.dumps(trace))",
        ],
        'pilot_command_template': 'python run_trace_pilot.py --trace <replay trace json> --output-dir <new dir> '
                                  '--ttft-slo <PENDING Mayor> (--budget <PENDING Mayor> | --budget-schedule '
                                  '<PENDING Mayor>) --cache-policy disabled',
        'pilot_command_notes': [
            '--cache-policy disabled is self-reported by the pilot; the vLLM server must actually be launched with prefix '
            'caching disabled and that launch verified separately.',
            'The Eval replay trace may be built and replayed only by the measurement harness, never by '
            'controllers/forecasters/tuning.',
        ],
        'native_client_mapping': 'NOT_VERIFIED: the peer-owned native-client offered-trace schema was not read by this '
                                 'component (Runtime6 owns it)',
    }


def _check_pair(protocol):
    specs = [_validated(protocol, purpose)[1] for purpose in PURPOSES]
    train, evaluation = specs
    if train['source_id'] == evaluation['source_id']:
        raise ValueError('Train and Eval source IDs must differ')
    if (train['request_id_prefix'].startswith(evaluation['request_id_prefix'])
            or evaluation['request_id_prefix'].startswith(train['request_id_prefix'])):
        raise ValueError('Train and Eval request-ID namespaces must be disjoint (neither prefix may extend the other)')


def build_outputs(protocol=None):
    """All output bytes (both tapes and ONE manifest), built entirely in memory before any write."""
    protocol = protocol_v1() if protocol is None else protocol
    _check_pair(protocol)
    outputs = {}
    splits = {}
    for purpose in PURPOSES:
        tape = generate_split(protocol, purpose)
        data = encode_tape(tape)
        name = f'{purpose}.tape.json'
        outputs[name] = data
        splits[purpose] = {
            'file': name, 'sha256': sha256_hex(data), 'bytes': len(data), 'count': tape['count'],
            'seed': tape['split']['seed'], 'source_id': tape['split']['source_id'],
            'request_id_prefix': tape['split']['request_id_prefix'], 'policy': POLICY[purpose],
        }
    manifest = {
        'schema': MANIFEST_SCHEMA,
        'kind': 'SYNTHETIC_OFFERED_WORKLOAD_MANIFEST',
        'generator': {'id': GENERATOR_ID, 'version': GENERATOR_VERSION, 'module': MODULE_PATH,
                      'module_sha256': _module_sha256()},
        'protocol': protocol,
        'protocol_sha256': sha256_hex(canonical_bytes(protocol)),
        'frozen_cli': {
            'generate': f'python -B -I {MODULE_PATH} generate --out-dir <new dir>',
            'verify': f'python -B -I {MODULE_PATH} verify --out-dir <dir>',
            'protocol_overrides': 'none: the CLI has no protocol override flags',
        },
        'splits': splits,
        'holdout_basis': list(HOLDOUT_BASIS),
        'prompt_length_semantics': PROMPT_LENGTH_SEMANTICS,
        'execution_distinction': 'This synthetic offered workload and any Native/GPU execution measurement are distinct '
                                 'artifacts; this component executed nothing on a backend.',
        'replay_adapter': _replay_adapter_mapping(),
        'claims_not_made': list(CLAIMS_NOT_MADE),
    }
    outputs[MANIFEST_NAME] = canonical_bytes(manifest)
    return outputs


def _summary(outputs):
    manifest = json.loads(outputs[MANIFEST_NAME])
    return {
        'files': {name: {'sha256': sha256_hex(data), 'bytes': len(data)} for name, data in sorted(outputs.items())},
        'counts': {purpose: manifest['splits'][purpose]['count'] for purpose in PURPOSES},
        'module_sha256': manifest['generator']['module_sha256'],
    }


def write_outputs(out_dir):
    """Generate the frozen v1 outputs into a NEW directory; never overwrites."""
    outputs = build_outputs()
    out = Path(out_dir)
    out.mkdir(exist_ok=False)
    for name in sorted(outputs):
        with (out / name).open('xb') as handle:
            handle.write(outputs[name])
    return _summary(outputs)


def verify_outputs(out_dir):
    """Regenerate in memory and require the directory to hold exactly the same files and bytes."""
    outputs = build_outputs()
    out = Path(out_dir)
    present = sorted(path.name for path in out.iterdir())
    if present != sorted(outputs):
        raise VerificationError(f'{out}: expected files {sorted(outputs)}, found {present}')
    for name, data in sorted(outputs.items()):
        if (out / name).read_bytes() != data:
            raise VerificationError(f'{out / name}: bytes differ from the regenerated v1 output')
    return _summary(outputs)


def load_tape(path, *, purpose):
    """Load one frozen v1 tape after checking its declared purpose and exact bytes."""
    if purpose not in PURPOSES:
        raise ValueError(f'purpose must be one of {PURPOSES}')
    data = Path(path).read_bytes()
    try:
        parsed = json.loads(data)
    except ValueError as exc:
        raise TapeIntegrityError(f'{path}: not JSON') from exc
    split = parsed.get('split') if isinstance(parsed, dict) else None
    declared = split.get('purpose') if isinstance(split, dict) else None
    if declared != purpose:
        raise TapePurposeError(f'{path}: tape purpose {declared!r} != requested {purpose!r}')
    if data != encode_tape(generate_split(protocol_v1(), purpose)):
        raise TapeIntegrityError(f'{path}: bytes differ from the frozen v1 {purpose} tape')
    return parsed


def _check_tape_shape(tape):
    if not isinstance(tape, dict) or tape.get('schema') != TAPE_SCHEMA or tape.get('kind') != TAPE_KIND:
        raise ValueError('not an offered tape of schema ' + TAPE_SCHEMA)
    if tape.get('columns') != list(COLUMNS) or not isinstance(tape.get('rows'), list):
        raise ValueError('offered tape columns/rows malformed')
    if tape.get('count') != len(tape['rows']):
        raise ValueError('offered tape count does not match its rows')
    cap = tape['protocol']['max_output_tokens']
    for row in tape['rows']:
        if not isinstance(row, list) or len(row) != len(COLUMNS) or row[3] != cap:
            raise ValueError('offered tape row malformed')


def to_replay_trace(tape, *, model, prompt_for, count_tokens, tokenizer_sha256=None):
    """Map an offered tape onto the run_trace_pilot replay schema (trace_workload schema_version 1).

    prompt_for(requested_prompt_tokens, payload_key) -> prompt text (shared template);
    count_tokens(prompt) -> MEASURED chat-template token count used as input_tokens.
    Synthesis or counting errors propagate: no retry, no new key, no fallback to the label.
    """
    _check_tape_shape(tape)
    if not isinstance(model, str) or not model:
        raise ValueError('model must be a non-empty string')
    if not tape['rows']:
        raise ValueError('an empty offered tape cannot form a replay trace (schema_version 1 needs >= 1 request)')
    cap = tape['protocol']['max_output_tokens']
    requests = []
    requested = []
    for request_id, offset_ns, requested_tokens, max_output_tokens, payload_key in tape['rows']:
        prompt = prompt_for(requested_tokens, payload_key)
        if not isinstance(prompt, str):
            raise TypeError(f'{request_id}: prompt_for must return text')
        measured = count_tokens(prompt)
        if type(measured) is not int or measured <= 0 or measured + cap > REPLAY_MAX_MODEL_LEN:
            raise ValueError(f'{request_id}: measured input token count {measured!r} is not an int in '
                             f'[1, {REPLAY_MAX_MODEL_LEN - cap}]')
        requests.append({'trace_id': request_id, 'arrival_seconds': offset_ns / NS_PER_SECOND,
                         'input_tokens': measured, 'max_tokens': max_output_tokens, 'prompt': prompt})
        requested.append(requested_tokens)
    provenance = {
        'kind': TAPE_KIND,
        'offered_tape_schema': TAPE_SCHEMA,
        'offered_tape_sha256': sha256_hex(encode_tape(tape)),
        'generator': tape['generator'],
        'protocol_id': tape['protocol']['protocol_id'],
        'purpose': tape['split']['purpose'],
        'source_id': tape['split']['source_id'],
        'requested_prompt_tokens': requested,
        'measured_equals_requested_count': sum(q['input_tokens'] == n for q, n in zip(requests, requested)),
        'prompt_length_semantics': PROMPT_LENGTH_SEMANTICS,
    }
    if tokenizer_sha256 is not None:
        if not isinstance(tokenizer_sha256, str) or not tokenizer_sha256:
            raise ValueError('tokenizer_sha256 must be a non-empty string when given')
        provenance['tokenizer_sha256'] = tokenizer_sha256
    return {
        'schema_version': 1,
        'max_model_len': REPLAY_MAX_MODEL_LEN,
        'model': model,
        'duration_seconds': tape['protocol']['duration_ns'] / NS_PER_SECOND,
        'output_cap': cap,
        'requests': requests,
        'provenance': provenance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Frozen protocol v1 synthetic offered-workload tapes '
                                                 '(no protocol override flags).', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('generate', 'verify'):
        commands.add_parser(name, allow_abbrev=False).add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    try:
        summary = write_outputs(args.out_dir) if args.command == 'generate' else verify_outputs(args.out_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'REFUSED', 'command': args.command, 'error': f'{type(exc).__name__}: {exc}'},
                         sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({'status': 'OK', 'command': args.command, **summary}, sort_keys=True, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
