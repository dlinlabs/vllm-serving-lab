"""Focused CPU-only controls for nominal_workload.py (stdlib unittest; no network, GPU or tokenizer).

Run with an absolute capsule scratch directory (never /tmp):
    NOMINAL_WORKLOAD_SCRATCH=<dir> python -B -I test_nominal_workload.py -v
Modules are imported by exact file path, so peer modules in this directory are never imported.
"""
import copy
import decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODULE = HERE / 'nominal_workload.py'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nw = _load('nominal_workload', MODULE)
tw = _load('trace_workload', ROOT / 'trace_workload.py')


def scratch(name):
    base = os.environ.get('NOMINAL_WORKLOAD_SCRATCH')
    if not base:
        raise RuntimeError('NOMINAL_WORKLOAD_SCRATCH must name an absolute capsule scratch directory')
    base = Path(base)
    if not base.is_absolute() or base.parts[:2] in (('/', 'tmp'),) or base.parts[:3] == ('/', 'var', 'tmp'):
        raise RuntimeError(f'refusing scratch directory {base}')
    base.mkdir(exist_ok=True)
    path = base / name
    path.mkdir(exist_ok=False)
    return path


def fresh_tapes():
    protocol = nw.protocol_v1()
    return {purpose: nw.generate_split(protocol, purpose) for purpose in nw.PURPOSES}


class RngSpecTest(unittest.TestCase):
    def test_uniform_source_matches_written_spec(self):
        for purpose, seed, stream, k in (('train', 4101, 'arrival', 0), ('eval', 4102, 'length_class', 7),
                                         ('train', 4101, 'payload', 1023), ('eval', 4102, 'arrival', 2**40)):
            literal = ('["r24-resume-p0-nominal-workload-generator",1,"r24-resume-p0-nominal-workload-v1","%s",%d,"%s"]'
                       % (purpose, seed, stream)).encode('ascii') + b'\x00'
            expected = int.from_bytes(hashlib.sha256(literal + k.to_bytes(8, 'big')).digest()[:8], 'big')
            domain = nw.rng_domain('r24-resume-p0-nominal-workload-v1', purpose, seed, stream)
            self.assertEqual(domain, literal)
            self.assertEqual(nw.uniform64(domain, k), expected)

    def test_known_answer_transforms(self):
        ctx = nw.decimal_context()
        self.assertEqual(nw.exponential_gap_seconds(0, 20), 0)
        top = nw.exponential_gap_seconds(2**64 - 1, 20)
        self.assertEqual(top, ctx.divide(ctx.ln(decimal.Decimal(2**64)), decimal.Decimal(20)))
        self.assertAlmostEqual(float(top), 64 * math.log(2) / 20, places=13)
        half = nw.exponential_gap_seconds(2**63, 20)
        self.assertAlmostEqual(float(half), math.log(2) / 20, places=15)
        self.assertLess(half, top)
        first, second = -(-2**64 // 3), -(-2**65 // 3)
        self.assertEqual((first, second), (6148914691236517206, 12297829382473034411))
        self.assertEqual([nw.length_class_index(x, 3) for x in (0, first - 1, first, second - 1, second, 2**64 - 1)],
                         [0, 0, 1, 1, 2, 2])
        for bad in (-1, 2**64, True, 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                nw.exponential_gap_seconds(bad, 20)
        with self.assertRaises(ValueError):
            nw.exponential_gap_seconds(5, 0)


class GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = nw.protocol_v1()
        cls.tapes = fresh_tapes()

    def test_repeatability_and_order_independence(self):
        eval_alone = nw.encode_tape(nw.generate_split(nw.protocol_v1(), 'eval'))
        train_again = nw.encode_tape(nw.generate_split(nw.protocol_v1(), 'train'))
        self.assertEqual(eval_alone, nw.encode_tape(nw.generate_split(nw.protocol_v1(), 'eval')))
        self.assertEqual(eval_alone, nw.encode_tape(self.tapes['eval']))
        self.assertEqual(train_again, nw.encode_tape(self.tapes['train']))
        self.assertEqual(self.protocol, nw.protocol_v1())

    def test_time_range_counts_values_and_window_rule(self):
        p = self.protocol
        for purpose, tape in self.tapes.items():
            with self.subTest(purpose=purpose):
                rows = tape['rows']
                offsets = [row[1] for row in rows]
                self.assertEqual(tape['count'], len(rows))
                self.assertTrue(1 <= len(rows) <= p['max_offers'])
                self.assertTrue(all(type(o) is int and 0 <= o < p['duration_ns'] for o in offsets))
                self.assertEqual(offsets, sorted(offsets))
                self.assertTrue({row[2] for row in rows} <= set(p['prompt_length_classes']))
                self.assertTrue(all(row[3] == p['max_output_tokens'] for row in rows))
                prefix = p['splits'][purpose]['request_id_prefix']
                self.assertEqual([row[0] for row in rows], [f'{prefix}{i:04d}' for i in range(len(rows))])
                shorter = nw.protocol_v1()
                shorter['duration_ns'] = offsets[-1]
                self.assertEqual(nw.generate_split(shorter, purpose)['rows'], [r for r in rows if r[1] < offsets[-1]])
                longer = nw.protocol_v1()
                longer['duration_ns'] = offsets[-1] + 1
                self.assertEqual(nw.generate_split(longer, purpose)['rows'], rows)

    def test_split_separation_and_invariance(self):
        train, evaluation = self.tapes['train'], self.tapes['eval']
        self.assertFalse({r[0] for r in train['rows']} & {r[0] for r in evaluation['rows']})
        self.assertFalse({r[4] for r in train['rows']} & {r[4] for r in evaluation['rows']})
        self.assertNotEqual(train['split']['source_id'], evaluation['split']['source_id'])
        self.assertNotEqual([r[1:3] for r in train['rows']], [r[1:3] for r in evaluation['rows']])
        train_text = nw.encode_tape(train).decode('ascii')
        for leaked in ('"seed":4102', 'p0w1-eval-', ':v1:eval', '/eval/'):
            self.assertNotIn(leaked, train_text)
        eval_bytes, train_bytes = nw.encode_tape(evaluation), nw.encode_tape(train)
        changed = nw.protocol_v1()
        changed['splits']['train'] = {'seed': 999, 'source_id': 'synthetic:other', 'request_id_prefix': 'x-'}
        self.assertEqual(nw.encode_tape(nw.generate_split(changed, 'eval')), eval_bytes)
        changed = nw.protocol_v1()
        changed['splits']['eval'] = {'seed': 4101, 'source_id': 'synthetic:other', 'request_id_prefix': 'y-'}
        self.assertEqual(nw.encode_tape(nw.generate_split(changed, 'train')), train_bytes)
        same_seed = nw.protocol_v1()
        same_seed['splits']['train']['seed'] = 4102
        pid = same_seed['protocol_id']
        for stream in nw.STREAMS:
            self.assertNotEqual([nw.uniform64(nw.rng_domain(pid, 'train', 4102, stream), k) for k in range(8)],
                                [nw.uniform64(nw.rng_domain(pid, 'eval', 4102, stream), k) for k in range(8)])
        self.assertNotEqual([r[1] for r in nw.generate_split(same_seed, 'train')['rows']],
                            [r[1] for r in evaluation['rows']])

    def test_max_offers_refusal_boundary(self):
        for purpose, tape in self.tapes.items():
            with self.subTest(purpose=purpose):
                at = nw.protocol_v1()
                at['max_offers'] = tape['count']
                self.assertEqual(nw.generate_split(at, purpose)['rows'], tape['rows'])
                below = nw.protocol_v1()
                below['max_offers'] = tape['count'] - 1
                with self.assertRaises(nw.OfferCapExceeded):
                    nw.generate_split(below, purpose)
        heavy = nw.protocol_v1()
        heavy['arrival_rate_per_second'] = 1000
        with self.assertRaises(nw.OfferCapExceeded):
            nw.generate_split(heavy, 'train')

    def test_tape_schema_and_canonical_bytes(self):
        for purpose, tape in self.tapes.items():
            with self.subTest(purpose=purpose):
                data = nw.encode_tape(tape)
                self.assertTrue(data.endswith(b'\n'))
                self.assertEqual(data.count(b'\n'), 1)
                data.decode('ascii')
                parsed = json.loads(data)
                self.assertEqual(nw.encode_tape(parsed), data)
                self.assertEqual(set(parsed), {'schema', 'kind', 'disclaimers', 'generator', 'protocol', 'split', 'rng',
                                               'payload_generation', 'columns', 'rows', 'count'})
                self.assertEqual((parsed['schema'], parsed['kind']), (nw.TAPE_SCHEMA, nw.TAPE_KIND))
                self.assertEqual(parsed['columns'], ['request_id', 'offset_ns', 'requested_prompt_tokens',
                                                     'max_output_tokens', 'payload_key'])
                self.assertNotIn('splits', parsed['protocol'])
                self.assertEqual(parsed['split']['purpose'], purpose)
                self.assertIn(nw.POLICY[purpose], parsed['disclaimers'])
                stack = [parsed]
                while stack:
                    item = stack.pop()
                    if isinstance(item, dict):
                        stack.extend(item.values())
                    elif isinstance(item, list):
                        stack.extend(item)
                    else:
                        self.assertIn(type(item), (int, str))


class OutputsTest(unittest.TestCase):
    def test_write_verify_and_single_manifest(self):
        out = scratch('outputs') / 'tapes'
        summary = nw.write_outputs(out)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['eval.tape.json', 'manifest.json', 'train.tape.json'])
        manifest = json.loads((out / 'manifest.json').read_bytes())
        self.assertEqual(manifest['schema'], nw.MANIFEST_SCHEMA)
        self.assertEqual(manifest['protocol'], nw.protocol_v1())
        self.assertEqual(manifest['protocol_sha256'], hashlib.sha256(nw.canonical_bytes(nw.protocol_v1())).hexdigest())
        self.assertEqual(manifest['generator']['module_sha256'], hashlib.sha256(MODULE.read_bytes()).hexdigest())
        for purpose in nw.PURPOSES:
            entry = manifest['splits'][purpose]
            data = (out / entry['file']).read_bytes()
            self.assertEqual((entry['sha256'], entry['bytes']), (hashlib.sha256(data).hexdigest(), len(data)))
            self.assertEqual(data, nw.encode_tape(nw.generate_split(nw.protocol_v1(), purpose)))
            self.assertEqual(entry['count'], json.loads(data)['count'])
            self.assertEqual(summary['counts'][purpose], entry['count'])
        self.assertIn('PENDING Mayor/Role7', manifest['splits']['train']['policy'])
        self.assertIn('never read', manifest['splits']['eval']['policy'])
        self.assertTrue(any('NOT claimed as holdout evidence' in text for text in manifest['holdout_basis']))
        self.assertFalse(manifest['replay_adapter']['parser_compatible_directly'])
        self.assertTrue(manifest['replay_adapter']['native_client_mapping'].startswith('NOT_VERIFIED'))
        self.assertIn('<PENDING Mayor>', manifest['replay_adapter']['pilot_command_template'])
        self.assertEqual(nw.verify_outputs(out), summary)
        with self.assertRaises(FileExistsError):
            nw.write_outputs(out)

    def test_load_tape_purpose_and_integrity_guards(self):
        base = scratch('load')
        out = base / 'tapes'
        nw.write_outputs(out)
        train_bytes = (out / 'train.tape.json').read_bytes()
        self.assertEqual(nw.load_tape(out / 'train.tape.json', purpose='train'), json.loads(train_bytes))
        self.assertEqual(nw.load_tape(out / 'eval.tape.json', purpose='eval')['split']['purpose'], 'eval')
        with self.assertRaises(nw.TapePurposeError):
            nw.load_tape(out / 'eval.tape.json', purpose='train')
        with self.assertRaises(nw.TapePurposeError):
            nw.load_tape(out / 'train.tape.json', purpose='eval')
        with self.assertRaises(ValueError):
            nw.load_tape(out / 'train.tape.json', purpose='test')
        parsed = json.loads(train_bytes)
        parsed['rows'][0][1] += 1
        (base / 'tampered.json').write_bytes(nw.encode_tape(parsed))
        (base / 'pretty.json').write_text(json.dumps(json.loads(train_bytes), indent=1))
        (base / 'garbage.json').write_bytes(b'\xff not json')
        for name in ('tampered.json', 'pretty.json', 'garbage.json'):
            with self.subTest(name=name), self.assertRaises(nw.TapeIntegrityError):
                nw.load_tape(base / name, purpose='train')


class ReplayAdapterTest(unittest.TestCase):
    def test_adapter_validates_and_records_measured_counts(self):
        tape = fresh_tapes()['train']
        calls = []

        def prompt_for(requested, key):
            calls.append((requested, key))
            return f'{key} {requested}'

        def count_tokens(prompt):
            return 100 + len(prompt) % 7

        trace = nw.to_replay_trace(tape, model=tw.MODEL, prompt_for=prompt_for, count_tokens=count_tokens,
                                   tokenizer_sha256='ab' * 32)
        self.assertIs(tw.validate_trace(trace), trace)
        rows = tape['rows']
        self.assertEqual(calls, [(row[2], row[4]) for row in rows])
        self.assertEqual([q['input_tokens'] for q in trace['requests']],
                         [count_tokens(f'{row[4]} {row[2]}') for row in rows])
        self.assertEqual(trace['provenance']['requested_prompt_tokens'], [row[2] for row in rows])
        self.assertEqual(trace['provenance']['measured_equals_requested_count'], 0)
        self.assertEqual([q['arrival_seconds'] for q in trace['requests']], [row[1] / 1e9 for row in rows])
        self.assertEqual([q['trace_id'] for q in trace['requests']], [row[0] for row in rows])
        self.assertEqual(trace['provenance']['offered_tape_sha256'], hashlib.sha256(nw.encode_tape(tape)).hexdigest())
        self.assertEqual(trace['provenance']['tokenizer_sha256'], 'ab' * 32)
        self.assertEqual((trace['duration_seconds'], trace['output_cap']), (5.0, 128))
        exact = nw.to_replay_trace(tape, model=tw.MODEL, prompt_for=prompt_for, count_tokens=lambda text: int(text.split()[-1]))
        self.assertIs(tw.validate_trace(exact), exact)
        self.assertEqual(exact['provenance']['measured_equals_requested_count'], len(rows))
        self.assertNotIn('tokenizer_sha256', exact['provenance'])
        for bad in (True, 0, -1, 2.0, '7', 8192 - 128 + 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                nw.to_replay_trace(tape, model=tw.MODEL, prompt_for=prompt_for, count_tokens=lambda text, bad=bad: bad)
        with self.assertRaises(TypeError):
            nw.to_replay_trace(tape, model=tw.MODEL, prompt_for=lambda n, key: None, count_tokens=count_tokens)

        def cannot_synthesize(n, key):
            raise ValueError(f'Cannot synthesize exactly {n} chat-template tokens')
        with self.assertRaisesRegex(ValueError, 'Cannot synthesize'):
            nw.to_replay_trace(tape, model=tw.MODEL, prompt_for=cannot_synthesize, count_tokens=count_tokens)
        empty = copy.deepcopy(tape)
        empty['rows'], empty['count'] = [], 0
        with self.assertRaises(ValueError):
            nw.to_replay_trace(empty, model=tw.MODEL, prompt_for=prompt_for, count_tokens=count_tokens)


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
        return subprocess.run([sys.executable, '-B', '-I', str(MODULE), *args], capture_output=True, text=True,
                              env=env, timeout=120, check=False)

    def test_frozen_cli_no_overrides_no_clobber_and_tamper_detection(self):
        base = scratch('cli')
        out = base / 'tapes'
        first = self.run_cli('generate', '--out-dir', str(out))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout)['status'], 'OK')
        before = {path.name: path.read_bytes() for path in out.iterdir()}
        again = self.run_cli('generate', '--out-dir', str(out))
        self.assertEqual(again.returncode, 1)
        self.assertIn('REFUSED', again.stderr)
        self.assertEqual({path.name: path.read_bytes() for path in out.iterdir()}, before)
        verified = self.run_cli('verify', '--out-dir', str(out))
        self.assertEqual(verified.returncode, 0, verified.stderr)
        for flags in (['--rate', '30'], ['--seed', '1'], ['--duration', '9'], ['--max-offers', '5'],
                      ['--protocol', 'p.json'], ['--out', str(base / 'abbrev')]):
            with self.subTest(flags=flags):
                result = self.run_cli('generate', '--out-dir', str(base / 'override'), *flags)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((base / 'override').exists())
        tampered = base / 'tampered'
        nw.write_outputs(tampered)
        path = tampered / 'eval.tape.json'
        path.write_bytes(path.read_bytes().replace(b'"count":', b'"count": ', 1))
        self.assertEqual(self.run_cli('verify', '--out-dir', str(tampered)).returncode, 1)
        extra = base / 'extra'
        nw.write_outputs(extra)
        (extra / 'notes.txt').write_bytes(b'x')
        self.assertEqual(self.run_cli('verify', '--out-dir', str(extra)).returncode, 1)


class ProtocolValidationTest(unittest.TestCase):
    def test_invalid_protocols_and_namespace_collisions_refused(self):
        mutations = [('arrival_rate_per_second', 0), ('arrival_rate_per_second', True), ('arrival_rate_per_second', 20.0),
                     ('duration_ns', 0), ('max_offers', 0), ('max_offers', 10_001), ('prompt_length_classes', []),
                     ('prompt_length_classes', [1024, 256]), ('prompt_length_classes', [256, 256]),
                     ('prompt_length_classes', [256, 8192]), ('max_output_tokens', 0), ('kind', 'REAL_TRACE'),
                     ('protocol_id', ''), ('arrival_process', 'bursty')]
        for key, value in mutations:
            protocol = nw.protocol_v1()
            protocol[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                nw.generate_split(protocol, 'train')
        for key, value in (('seed', -1), ('seed', True), ('source_id', ''), ('request_id_prefix', 7)):
            protocol = nw.protocol_v1()
            protocol['splits']['train'][key] = value
            with self.subTest(split_key=key, value=value), self.assertRaises(ValueError):
                nw.generate_split(protocol, 'train')
        for purpose in ('test', 'TRAIN', ''):
            with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                nw.generate_split(nw.protocol_v1(), purpose)
        for key, value in (('request_id_prefix', 'p0w1-train-'), ('request_id_prefix', 'p0w1-'),
                           ('source_id', 'synthetic:r24-resume-p0-nominal-workload:v1:train')):
            protocol = nw.protocol_v1()
            protocol['splits']['eval'][key] = value
            with self.subTest(collision=key, value=value), self.assertRaises(ValueError):
                nw.build_outputs(protocol)


if __name__ == '__main__':
    unittest.main()
