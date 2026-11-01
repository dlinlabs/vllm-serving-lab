import argparse
import gzip
import hashlib
import json
import math

import numpy as np
import pytest

import diagnose_cap_misses as dc
from scripts import reproduce_cap_miss_causes as rep

START = 1000.
METRICS = {'steps': 'iteration_tokens_total_count', 'prompt': 'prompt_tokens_total',
           'gen': 'generation_tokens_total', 'running': 'num_requests_running',
           'queue_sum': 'request_queue_time_seconds_sum', 'queue_n': 'request_queue_time_seconds_count',
           'prefill_sum': 'request_prefill_time_seconds_sum', 'prefill_n': 'request_prefill_time_seconds_count'}


def write_capture(root, schedule, seed=0, duration=300., failed_scrape=150, transport_failure=True):
    """Synthetic capture: a request every 0.5 s, one scrape per second, one failed scrape."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps(dict(measurement_start=START, created_at=1.)))
    (root / 'replay_plan.json').write_text(json.dumps(dict(
        duration_seconds=duration, budget_schedule=[dict(at_seconds=a, budget=b) for a, b in schedule])))
    events, clients, trace, spans = [], [], [], []
    for i, sched in enumerate(np.arange(1., duration - 1., .5)):
        rid, tid = f'r{i}', f't{i}'
        tokens = (500, 1500, 3000, 6000)[i % 4] + int(rng.integers(0, 100))
        trace.append(dict(trace_id=tid, input_tokens=tokens))
        t = START + float(sched)
        events.append(dict(request_id=rid, event='received', monotonic=t + .01))
        if i % 7 == 6:
            events.append(dict(request_id=rid, event='rejected', monotonic=t + .02, request_cost=1.,
                               estimated_input_tokens=tokens))
            clients.append(dict(request_id=rid, workload='trace', scheduled_at=t, success=False, trace_id=tid))
            continue
        acc = t + .02
        events.append(dict(request_id=rid, event='accepted', monotonic=acc, request_cost=1.,
                           estimated_input_tokens=tokens))
        if transport_failure and i == 11:
            clients.append(dict(client_request_id=rid, workload='trace', scheduled_at=t, success=False,
                                trace_id=tid))
            spans.append((acc - START, math.nan, tokens))
            continue
        first = acc + .05 + tokens * 8e-5 + float(rng.uniform(0., .1))
        events += [dict(request_id=rid, event='first_content', monotonic=first),
                   dict(request_id=rid, event='completed', monotonic=first + 1.)]
        clients.append(dict(request_id=rid, workload='trace', scheduled_at=t, first_content_at=first + .005,
                            success=True, stream_complete=True, trace_id=tid))
        spans.append((acc - START, first - START, tokens))
    clients.append(dict(request_id='w', workload='warmup', scheduled_at=START - 1., trace_id='warm'))
    (root / 'gateway.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n')
    (root / 'client.json').write_text(json.dumps(dict(requests=clients)))
    (root / 'trace.json').write_text(json.dumps(dict(requests=trace)))
    acc, first, tokens = (np.array(x, float) for x in zip(*spans))
    scrapes, steps, gen = [], 0., 0.
    for s in range(int(duration) + 1):
        steps += 20 + s % 5
        gen += 40 + 7 * (s % 3)
        done = first <= s
        values = dict(steps=steps, prompt=tokens[done].sum(), gen=gen, running=2 if s % 10 == 0 else 5,
                      queue_sum=.01 * (acc <= s).sum(), queue_n=(acc <= s).sum(),
                      prefill_sum=(first - acc)[done].sum(), prefill_n=done.sum())
        raw = '' if s == failed_scrape else ''.join(
            f'vllm:{METRICS[k]}{{engine="0"}} {float(v)}\n' for k, v in values.items())
        scrapes.append(dict(sent_at=START + s, received_at=START + s + .05, raw_metrics=raw))
    (root / 'engine_observations.jsonl').write_text('\n'.join(map(json.dumps, scrapes)) + '\n')
    return root


def write_suite(root, seed):
    write_capture(root / 'run-fixed-32', [(0., 32.)], seed=seed)
    write_capture(root / 'run-switch', [(0., 32.), (180., 16.)], seed=seed + 1)
    write_capture(root / 'run-smoke-32', [(0., 32.)], seed=seed + 2, duration=300.)
    write_capture(root / 'run-short', [(0., 32.)], seed=seed + 3, duration=60.)
    (root / 'notes').mkdir()
    return root


def write_report(path):
    predictions = [
        dict(run='r', t=60, cap=32, admission=10., actual_admission=8., simulator=7., actual_goodput=4.,
             no_bias=6., online_bias_s=.01),
        dict(run='r', t=120, cap=32, admission=9., actual_admission=9., simulator=8., actual_goodput=6.,
             no_bias=7., online_bias_s=.03),
        dict(run='s', t=60, cap=16, admission=5., actual_admission=4., simulator=5., actual_goodput=4.,
             no_bias=5., online_bias_s=0.)]
    oracle = [dict(run=p['run'], t=p['t'], goodput=p['actual_goodput'] + 1., admission=p['actual_admission'],
                   ttft_matched_n=2, ttft_signed_error_sum=-.1) for p in predictions]
    path.write_bytes(gzip.compress(json.dumps(dict(predictions=predictions, oracle_diagnostics=oracle)).encode()))
    return path


def test_recent_tokens_excludes_self_and_simultaneous_acceptances():
    out = dc.recent_tokens([0., .5, .5, 1.2, math.nan], [100., 200., 300., 400., 500.])
    assert list(out[:4]) == [0., 100., 100., 500.] and math.isnan(out[4])


def test_window_and_spread_sums():
    times, values = np.array([.2, 1., math.nan, 2.5]), np.array([1., 2., 4., 8.])
    assert list(dc.window_sum(times, values, np.array([0., 1., 0.]), np.array([1., 3., 3.]))) == [1., 10., 11.]
    start, end = np.array([0., 5., math.nan]), np.array([2., 5., 1.])
    out = dc.spread_sum(start, end, np.array([10., 3., 100.]), np.array([0., 1., 1.5, 5.]),
                        np.array([1., 2., 3., 6.]))
    assert out == pytest.approx([5., 5., 2.5, 3.])


def test_budget_and_segment_lookup():
    schedule = [dict(at_seconds=0., budget=32.), dict(at_seconds=180., budget=16.)]
    out = dc.budget_at(schedule, [-1., 0., 179.9, 180., 250.])
    assert math.isnan(out[0]) and list(out[1:]) == [32., 32., 16., 16.]
    assert list(dc.segment_of([59.9, 60., 119.99, 120., 299.9, 300.])) == [-1, 0, 0, 1, 3, -1]


def test_step_models_recover_synthetic_coefficients_and_score_out_of_capture():
    rng = np.random.default_rng(3)
    n = 400
    rows = {k: rng.uniform(1., 10., n) for k in ('steps', 'gen', 'prompt', 'tok_accepted', 'tok_spread')}
    rows.update(sq_accepted=rng.uniform(0., 50., n), sq_spread=rng.uniform(0., 50., n))
    truth = np.array([.01, .002, .003, .004, .0005])
    rows['dt'] = dc.design(rows, 'tok_sq_spread') @ truth
    rows['capture'] = np.array(['a', 'b', 'c', 'd'] * (n // 4), dtype=object)
    assert dc.fit(rows, 'tok_sq_spread') == pytest.approx(truth)
    report = dc.fit_report(rows, 'tok_sq_spread')
    assert report['rmse_s'] < 1e-12 and report['nonnegative']
    assert report['squared_term_s_for_one_8000_token_prompt'] == pytest.approx(.0005 * 8000 ** 2)
    assert 'squared_term_s_for_one_8000_token_prompt' not in dc.fit_report(rows, 'tok_spread')
    held = dc.loco(rows, 'tok_sq_spread')
    assert held['captures'] == 4 and held['captures_better_than_linear'] == 4 and held['rmse_s'] < 1e-12
    assert dc.loco(rows, 'linear')['rmse_s'] > 1e-3
    split = {name: {k: v[rows['capture'] == name] for k, v in rows.items()} for name in ('a', 'b')}
    cross = dc.cross_host(split)
    assert len(cross) == 2 * len(dc.STEP_MODELS)
    assert [x['rmse_s'] < 1e-12 for x in cross if x['model'] == 'tok_sq_spread'] == [True, True]


def test_loco_and_cross_host_never_fit_on_scored_intervals():
    rng = np.random.default_rng(7)
    parts = []
    for i, name in enumerate('abcd'):
        x = {k: rng.uniform(1., 10., 60) for k in ('steps', 'gen', 'prompt', 'tok_accepted', 'tok_spread')}
        x.update(sq_accepted=rng.uniform(0., 50., 60), sq_spread=rng.uniform(0., 50., 60))
        # Each capture has its own true cost, so a fit that saw it would score better.
        x['dt'] = np.column_stack([x[k] for k in dc.BASE]) @ (np.array([.01, .002, .003]) * (1. + .3 * i))
        x['dt'] = x['dt'] + rng.normal(0., .01, 60)
        x['capture'] = np.full(60, name, dtype=object)
        parts.append(x)
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def manual(train, test, model):
        design = lambda r: np.column_stack([r[k] for k in dc.STEP_MODELS[model]])
        coef = np.linalg.lstsq(design(train), train['dt'], rcond=None)[0]
        return test['dt'] - design(test) @ coef

    for model in ('linear', 'tok_sq_spread'):
        errors = []
        for name in 'abcd':
            held = rows['capture'] == name
            errors.append(manual({k: v[~held] for k, v in rows.items()}, {k: v[held] for k, v in rows.items()}, model))
        out = dc.loco(rows, model)
        assert out['rmse_s'] == pytest.approx(float(np.sqrt(np.mean(np.concatenate(errors) ** 2))), rel=1e-9)
        assert out['rmse_s'] > 1.2 * dc.fit_report(rows, model)['rmse_s']
    split = {name: {k: v[rows['capture'] == name] for k, v in rows.items()} for name in ('a', 'd')}
    cross = dc.cross_host(split)
    assert [(x['train'], x['test']) for x in cross] == [('a', 'd')] * 6 + [('d', 'a')] * 6
    for x in cross:
        train, test = split[x['train']], split[x['test']]
        assert x['rmse_s'] == pytest.approx(float(np.sqrt(np.mean(manual(train, test, x['model']) ** 2))), rel=1e-9)
        assert x['rmse_s'] > 3 * dc.fit_report(test, x['model'])['rmse_s']


def test_misses_filters_cap_and_cuts_quartiles_on_short_prompts_only():
    # (capture, segment, cap, ok, tokens, recent, met, accept -> first content)
    table = [
        ('a', 0, 32., True, 500, 0., True, .1), ('a', 0, 32., True, 500, 5., True, .2),
        ('a', 0, 32., True, 1500, 10., True, .3), ('a', 1, 32., True, 1500, 15., False, .4),
        ('a', 1, 32., True, 500, 20., True, .3), ('a', 1, 32., True, 3000, 1e6, False, .6),
        ('b', 2, 32., True, 500, 25., False, .5), ('b', 2, 32., True, 1500, 30., False, .5),
        ('b', 3, 32., True, 500, 35., False, .7), ('b', 3, 32., True, 1500, 40., True, .2),
        ('b', 3, 32., True, 5000, 2e6, False, .9),
        # Excluded: another cap, outside the segments, not admitted and successful.
        ('b', 2, 16., True, 500, 1e5, False, 9.), ('b', -1, 32., True, 500, 1e5, False, 9.),
        ('b', 0, 32., False, 500, 1e5, False, 9.)]
    captures = []
    for name in 'ab':
        r = [x for x in table if x[0] == name]
        acc = 10. + np.arange(len(r))
        captures.append(dict(
            name=name, seg=np.array([x[1] for x in r]), cap=np.array([x[2] for x in r]),
            ok=np.array([x[3] for x in r]), tokens=np.array([x[4] for x in r], float),
            recent=np.array([x[5] for x in r]), met=np.array([x[6] for x in r]),
            acc=acc, first=acc + np.array([x[7] for x in r])))
    out = dc.misses(captures, 32)
    assert out['captures'] == ['a', 'b'] and out['n'] == 11
    # Short-prompt recent loads are 0, 5, ..., 40: cuts 10 / 20 / 30, bins (lo, hi].
    quartiles = out['short_miss_rate_by_recent_quartile']
    assert [(q['lo'], q['hi']) for q in quartiles] == [(-math.inf, 10.), (10., 20.), (20., 30.), (30., math.inf)]
    assert [q['n'] for q in quartiles] == [3, 2, 2, 2]
    assert [q['miss_rate'] for q in quartiles] == [0., .5, 1., .5]
    lengths = out['accept_to_first_by_own_tokens']
    assert [b['n'] for b in lengths] == [5, 4, 1, 1]
    assert [b['accept_to_first_median_s'] for b in lengths] == pytest.approx([.3, .35, .6, .9])
    segments = out['segments']
    assert [(s['segment'], s['n'], s['misses']) for s in segments] == [(1, 3, 0), (2, 3, 2), (3, 2, 2), (4, 3, 2)]
    assert math.isnan(segments[0]['short_share_of_misses'])
    assert [s['short_share_of_misses'] for s in segments[1:]] == [.5, 1., .5]
    assert [b['n'] for b in segments[0]['miss_rate_by_own_tokens']] == [2, 1, 0, 0]
    assert [b['miss_rate'] for b in segments[1]['miss_rate_by_own_tokens']] == pytest.approx(
        [0., 1., 1., math.nan], nan_ok=True)
    assert segments[0]['recent_tokens_mean'] == pytest.approx(5.)


def test_rounded_is_json_safe_and_platform_stable():
    out = dc.rounded(dict(a=np.float64(1.23456789), b=[np.nan, -np.inf, np.int64(3), np.bool_(True)], c=(1, 2.)))
    assert out == dict(a=1.23457, b=[None, None, 3, True], c=[1, 2.])
    assert type(out['b'][2]) is int and type(out['b'][3]) is bool
    json.dumps(out, allow_nan=False)


def test_load_patches_only_identified_transport_failures(tmp_path):
    root = write_capture(tmp_path / 'run', [(0., 32.)])
    meta, table, patched = dc.load(root)
    assert patched == 1 and 'r11' in set(table['request_id'])
    i = list(table['request_id']).index('r11')
    assert table['admitted'][i] and not table['success'][i]
    client = json.loads((root / 'client.json').read_text())
    for row in client['requests']:
        if row['request_id' if 'request_id' in row else 'client_request_id'] == 'r12':
            del row['request_id']
            row['client_request_id'] = 'r12'
    (root / 'client.json').write_text(json.dumps(client))
    with pytest.raises(ValueError, match='identified transport failure'):
        dc.load(root)


def test_load_without_patches_matches_greybox_reader(tmp_path):
    root = write_capture(tmp_path / 'run', [(0., 32.)], transport_failure=False)
    meta, table, patched = dc.load(root)
    assert patched == 0 and meta['start'] == START and len(table['sched']) == 596


def test_discover_skips_smoke_short_and_incomplete_directories(tmp_path):
    suite = write_suite(tmp_path / 'suite', seed=0)
    assert [p.name for p in dc.discover(suite)] == ['run-fixed-32', 'run-switch']
    with pytest.raises(ValueError, match='No full-length captures'):
        dc.discover(suite / 'notes')


def test_intervals_posted_bridges_failed_scrape_and_strict_does_not(tmp_path):
    c = dc.capture(write_capture(tmp_path / 'run', [(0., 32.)]))
    coef = (.01, .001, .0001)
    posted, strict = dc.intervals(c, coef), dc.intervals(c, coef, strict=True)
    assert len(posted['dt']) == 239 and len(strict['dt']) == 238
    assert list(posted['dt'][posted['dt'] != 1.]) == [2.]
    assert set(strict['dt'].tolist()) == {1.}
    # The synthetic prompt counter increments when first content arrives.
    assert posted['prompt'] == pytest.approx(posted['tok_first'])
    assert posted['linear'] == pytest.approx(.01 * posted['steps'] + .001 * posted['gen'] + .0001 * posted['prompt'])
    assert set(posted['cap'].tolist()) == {32.} and set(posted['seg'].tolist()) == {0, 1, 2, 3}


def test_decomposition_splits_goodput_error_exactly(tmp_path):
    path = write_report(tmp_path / 'report.json.gz')
    out = dc.decomposition(path)
    assert out['source'] == 'report.json.gz'
    assert out['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [(g['cap'], g['segment']) for g in out['groups']] == [(16, None), (16, 1), (32, None), (32, 1), (32, 2)]
    for g in out['groups']:
        assert g['goodput_error'] == pytest.approx(g['admission_term'] + g['hit_rate_term'])
        assert g['oracle_goodput_error'] == pytest.approx(1.)
        assert g['oracle_matched_ttft_error_ms'] == pytest.approx(-50.)
    first = out['groups'][3]
    assert (first['goodput_error'], first['admission_term'], first['hit_rate_term']) == pytest.approx((3., 1., 2.))
    assert out['groups'][2]['online_bias_ms'] == pytest.approx(20.)


def test_end_to_end_two_host_suite_is_json_safe(tmp_path):
    hosts = {'old': write_suite(tmp_path / 'old', seed=0), 'new': write_suite(tmp_path / 'new', seed=10)}
    coef = (.01, .001, .0001)
    result = dc.analyze(hosts, coef, write_report(tmp_path / 'report.json.gz'))
    json.dumps(result, allow_nan=False)
    assert result['frozen_step_coefficients'] == dict(step_s=.01, per_decode_token_s=.001, per_prompt_token_s=.0001)
    assert len(result['cross_host_posted_busy']) == 2 * len(dc.STEP_MODELS)
    assert result['frozen_decomposition']['source'] == 'report.json.gz'
    for label, host in result['hosts'].items():
        assert host['suite'] == label
        assert [c['name'] for c in host['captures']] == ['run-fixed-32', 'run-switch']
        assert host['captures'][0]['transport_identity_patches'] == 1
        assert set(host['captures'][0]['sha256']) == set(dc.HASHED)
        assert [s['cap'] for s in host['stages'][1]['segments']] == [32., 32., 16., 16.]
        assert [m['cap'] for m in host['misses_fixed']] == [32.]
        assert host['misses_fixed'][0]['captures'] == ['run-fixed-32']
        assert host['misses_all_cap32_segments']['captures'] == ['run-fixed-32', 'run-switch']
        assert [s['segment'] for s in host['misses_all_cap32_segments']['segments']] == [1, 2, 3, 4]
        # Only long prompts miss in the synthetic engine.
        for segment in host['misses_fixed'][0]['segments']:
            assert [b['miss_rate'] for b in segment['miss_rate_by_own_tokens']] == [0., 0., 0., 1.]
            assert segment['short_share_of_misses'] == 0.
        posted, strict = host['intervals']['posted'], host['intervals']['strict']
        assert [(x['cap'], x['segment']) for x in posted['exporter_fixed']] == [(32., k) for k in (1, 2, 3, 4)]
        # Per capture, posted keeps the busy bridged interval 149-151 s that strict drops.
        assert posted['busy']['intervals'] == strict['busy']['intervals'] + 2
        assert posted['busy']['prompt_counter_correlation']['tok_first'] == pytest.approx(1.)
        assert [x['model'] for x in posted['busy']['leave_one_capture_out']] == list(dc.STEP_MODELS)
        assert all(x['captures'] == 2 for x in posted['busy']['leave_one_capture_out'])


def test_cli_writes_lf_json_and_rejects_bad_hosts(tmp_path, capsys):
    suite = write_suite(tmp_path / 'suite', seed=0)
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps(dict(frozen_step_coefficients=dict(
        step_s=.01, per_decode_token_s=.001, per_prompt_token_s=.0001))))
    out = tmp_path / 'out' / 'result.json'
    dc.main(['--host', f'a={suite}', '--frozen-summary', str(summary), '--no-decomposition', '--output', str(out)])
    data = out.read_bytes()
    assert data.endswith(b'}\n') and b'\r\n' not in data
    result = json.loads(data)
    assert list(result['hosts']) == ['a'] and 'frozen_decomposition' not in result
    assert result['cross_host_posted_busy'] == []
    with pytest.raises(SystemExit):
        dc.main(['--host', f'a={suite}', '--host', f'a={suite}', '--output', str(out)])
    with pytest.raises(argparse.ArgumentTypeError):
        dc.host_arg('no-label')


def test_reproduction_compares_floats_with_tolerance_and_everything_else_exactly():
    expected = dict(a=1.0, b=[1, None, 'x'], c=dict(d=3e-9))
    assert rep.differences(json.loads(json.dumps(expected)), expected) == []
    assert rep.differences(dict(a=1.00001, b=[1, None, 'x'], c=dict(d=3.0001e-9)), expected) == []
    assert rep.differences(dict(a=1.001, b=[1., None, 'y'], c=dict(d=3e-9)), expected) == ['$.a', '$.b[0]', '$.b[2]']
    assert rep.differences(dict(a=1., b=[1], c={}), expected) == ['$.b', '$.c']


def test_reproduction_verifies_every_restored_file(tmp_path):
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / 'f.txt').write_bytes(b'abc')
    files = {'x/f.txt': (3, hashlib.sha256(b'abc').hexdigest())}
    assert rep.verify(tmp_path, files) == 1
    (tmp_path / 'x' / 'f.txt').write_bytes(b'abd')
    with pytest.raises(ValueError, match='x/f.txt'):
        rep.verify(tmp_path, files)
    with pytest.raises(ValueError, match='y.txt'):
        rep.verify(tmp_path, {'y.txt': (0, '0' * 64)})
