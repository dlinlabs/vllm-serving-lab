"""CPU causal two-slot prototype for the reviewed vLLM 0.30 V2 scope.

This is an UNFITTED simulator, not an accepted latency model. It generates work
from requests, not captured future schedules/timestamps. No GPU imports/hooks.
See experiments/prefill-async-predictor-v1.md for its approximations and limits.
"""
from collections import deque
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from .causal import mode_row
from .contracts import digest, features, fifo_reference


FEATURES = frozenset(('bias', 'prefill', 'decode', 'decode_context',
                      'attention_pairs', 'padding_excess', 'num_requests'))
PARTS = ('cpu_to_first', 'cpu_between', 'cpu_after', 'gpu_prefix', 'gpu_tail')
LAUNCHES = ('forward', 'logits_sample', 'post_update')
SCALARS = ('schedule_cpu', 'empty_submit_cpu', 'copy_enqueue_cpu', 'copy_service',
           'output_ready_cpu', 'commit_cpu')


def nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Costs and arrival offsets must be finite and nonnegative')
    return float(value)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class CostLaw:
    """Explicit nonnegative affine coefficients; no implicit timing defaults."""
    coefficients: dict = field(default_factory=dict)

    def __post_init__(self):
        if set(self.coefficients) - FEATURES:
            raise ValueError('Unsupported stage-cost feature')
        object.__setattr__(self, 'coefficients', MappingProxyType({
            k: nonnegative(v) for k, v in self.coefficients.items()}))

    def __call__(self, row):
        result = sum(v * (1. if k == 'bias' else row[k])
                     for k, v in self.coefficients.items())
        return nonnegative(result)


@dataclass(frozen=True)
class PipelineCosts:
    launches: dict
    scalar: dict
    provenance: str

    def __post_init__(self):
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError('Explicit cost provenance is required')
        if set(self.launches) != set(LAUNCHES) or set(self.scalar) != set(SCALARS):
            raise ValueError('Exactly the declared pipeline stages are required')
        for parts in self.launches.values():
            if set(parts) != set(PARTS) or any(not isinstance(v, CostLaw) for v in parts.values()):
                raise ValueError('Each launch requires five explicit CostLaw parts')
        if any(not isinstance(v, CostLaw) for v in self.scalar.values()):
            raise ValueError('Scalar stages require CostLaw values')
        object.__setattr__(self, 'launches', MappingProxyType({
            k: MappingProxyType(dict(parts)) for k, parts in self.launches.items()}))
        object.__setattr__(self, 'scalar', MappingProxyType(dict(self.scalar)))

    @classmethod
    def from_dict(cls, data):
        if set(data) != {'schema', 'provenance', 'launches', 'scalar'} or data['schema'] != 'async-stage-costs-v1':
            raise ValueError('Unsupported stage-cost document')
        return cls({k: {p: CostLaw(v) for p, v in parts.items()}
                    for k, parts in data['launches'].items()},
                   {k: CostLaw(v) for k, v in data['scalar'].items()}, data['provenance'])

    def to_dict(self):
        return dict(schema='async-stage-costs-v1', provenance=self.provenance,
                    launches={k: {p: dict(v.coefficients) for p, v in parts.items()}
                              for k, parts in self.launches.items()},
                    scalar={k: dict(v.coefficients) for k, v in self.scalar.items()})


class EventGraph:
    """Max-plus CPU/main-stream/copy-stream ordering with explicit dependencies."""
    def __init__(self):
        self.nodes, self.resources = [], {}

    def add(self, name, duration, *, resource, dependencies=(), ready=0., batch=None):
        duration, ready = nonnegative(duration), nonnegative(ready)
        deps = list(dict.fromkeys(dependencies))
        if resource in self.resources:
            deps.append(self.resources[resource])
        deps = list(dict.fromkeys(deps))
        if any(type(d) is not int or not 0 <= d < len(self.nodes) for d in deps):
            raise ValueError('Dependency must name an earlier node')
        start = max([ready] + [self.nodes[d]['end_s'] for d in deps])
        end = nonnegative(start + duration)
        node = dict(id=len(self.nodes), name=name, batch=batch, resource=resource,
                    dependencies=deps, ready_s=ready, start_s=start, end_s=end,
                    service_s=duration)
        self.nodes.append(node)
        self.resources[resource] = node['id']
        return node['id']

    def end(self, node):
        return self.nodes[node]['end_s']

    def launch(self, name, costs, row, *, ready=0., batch=None):
        # The GPU prefix can run before CPU submission returns. The GPU tail
        # additionally waits for the last enqueue; later main-stream work waits
        # for this tail. These are explicit model parameters, not measured spans.
        first = self.add(name + '.cpu_to_first', costs['cpu_to_first'](row),
                         resource='cpu', ready=ready, batch=batch)
        prefix = self.add(name + '.gpu_prefix', costs['gpu_prefix'](row),
                          resource='main', dependencies=[first], batch=batch)
        last = self.add(name + '.cpu_between', costs['cpu_between'](row),
                        resource='cpu', dependencies=[first], batch=batch)
        tail = self.add(name + '.gpu_tail', costs['gpu_tail'](row),
                        resource='main', dependencies=[prefix, last], batch=batch)
        returned = self.add(name + '.cpu_after', costs['cpu_after'](row),
                            resource='cpu', dependencies=[last], batch=batch)
        return returned, tail


@dataclass(frozen=True)
class Request:
    id: str
    prompt_tokens: int
    max_tokens: int
    at_s: float = 0.
    role: str = 'foreground'

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('Nonempty request ID required')
        positive_int(self.prompt_tokens, 'prompt_tokens')
        positive_int(self.max_tokens, 'max_tokens')
        nonnegative(self.at_s)
        if self.role not in ('foreground', 'sentinel', 'background'):
            raise ValueError('Unsupported request role')
        if self.role == 'background' and self.at_s != 0:
            raise ValueError('Backgrounds must enter the empty engine at time zero')


@dataclass(frozen=True)
class Scope:
    token_budget: int = 2048
    max_seqs: int = 32
    max_model_len: int = 8192
    batch_slots: int = 2
    block_size: int = 16
    kv_blocks: int = 5594

    def __post_init__(self):
        for k, v in vars(self).items():
            positive_int(v, k)
        if self.batch_slots != 2:
            raise ValueError('Only the reviewed native two-slot scope is implemented')


def simulate(requests, costs, dispatch_table, *, scope=Scope(), max_iterations=20000):
    """Simulate from empty; foreground offsets are relative to predicted barrier.

    V2/UniProc/TP=PP=DP=1, FCFS, forced outputs, no prefix/spec/preemption.
    KV pressure fails closed instead of inventing eviction or acceptance.
    No captured release state, future modes, schedules or timing labels accepted.
    """
    requests = list(requests)
    positive_int(max_iterations, 'max_iterations')
    if not requests or any(not isinstance(r, Request) for r in requests):
        raise ValueError('Explicit Request objects are required')
    if not isinstance(costs, PipelineCosts) or not isinstance(scope, Scope):
        raise ValueError('Explicit PipelineCosts and Scope required')
    if dispatch_table.get('schema') != 'v2-dispatch-candidates-v1':
        raise ValueError('A frozen V2 dispatcher is required')
    if len({r.id for r in requests}) != len(requests):
        raise ValueError('Duplicate request ID')
    if any(r.prompt_tokens + r.max_tokens > scope.max_model_len for r in requests):
        raise ValueError('Prompt plus output exceeds supported context')
    backgrounds = {r.id for r in requests if r.role == 'background'}
    foregrounds = {r.id for r in requests} - backgrounds
    if not foregrounds:
        raise ValueError('At least one foreground request is required')
    specifications = {r.id: r for r in requests}
    state = dict(running=[], waiting=[], requests={})
    outcomes = {r.id: dict(role=r.role, arrival_s=None, enqueue_s=None,
                           first_s=None, finished_s=None, output_tokens=0)
                for r in requests}
    graph, queue, steps, actions = EventGraph(), deque(), [], []
    release, now, peak_slots, peak_blocks, pending = None, 0., 0, 0, []

    def inject(ids, at):
        for rid in sorted(ids):
            spec = specifications[rid]
            state['requests'][rid] = dict(prompt_tokens=spec.prompt_tokens,
                max_tokens=spec.max_tokens, num_tokens=spec.prompt_tokens,
                computed=0, placeholders=0, in_flight_tokens=0)
            state['waiting'].append(rid)
            outcomes[rid]['enqueue_s'] = at
            actions.append(dict(operation='arrival', request_id=rid, at_s=at))

    def release_foreground(at):
        nonlocal release, pending
        release = at
        pending = sorted((at + specifications[rid].at_s, rid) for rid in foregrounds)
        for target, rid in pending:
            outcomes[rid]['arrival_s'] = target
        immediate = [rid for target, rid in pending if target == at]
        inject(immediate, at)
        pending = [(target, rid) for target, rid in pending if target != at]
        actions.append(dict(operation='barrier', at_s=at))

    def finish_iteration(executed, at, batch):
        # EngineCore._process_engine_step yields 1ms when an empty scheduling
        # iteration leaves scheduler requests. This is thread unavailability,
        # not CPU busy time. No sleep or wall clock is used by this simulator.
        if not executed and state['requests']:
            return graph.end(graph.add('native_idle_yield', .001, resource='cpu',
                                       ready=at, batch=batch))
        return at

    if backgrounds:
        for rid in backgrounds:
            outcomes[rid]['arrival_s'] = 0.
        inject(backgrounds, 0.)
    else:
        release_foreground(0.)

    for iteration in range(max_iterations):
        if not state['requests'] and not queue and not pending:
            break
        # Native input queue is polled once per engine iteration, not while the
        # CPU is blocked on an old output future. Timers have ideal posting here.
        if not state['requests'] and not queue and pending:
            now = max(now, pending[0][0])
        due = [(at, rid) for at, rid in pending if at <= now]
        for _, rid in due:
            inject([rid], now)
        pending = [(at, rid) for at, rid in pending if at > now]

        executed = False
        if state['requests']:
            counts = fifo_reference(state, scope.token_budget, scope.max_seqs)
            work = [dict(request_id=rid, position=state['requests'][rid]['computed'], tokens=n)
                    for rid, n in counts.items()]
            row = features(work, {rid: r.prompt_tokens for rid, r in specifications.items()})
            if work:
                executed = True
                row = mode_row(row, dispatch_table)
                if row['mode'] not in ('NONE', 'FULL', 'PIECEWISE'):
                    raise ValueError('Unsupported runtime mode')
            else:
                row.update(mode='EMPTY', padding_excess=0)
            row['num_requests'] = len(work)
            bid = iteration
            schedule = graph.add('schedule', costs.scalar['schedule_cpu'](row),
                                 resource='cpu', ready=now, batch=bid)
            before = {rid: dict(state['requests'][rid]) for rid in counts}
            emits = []
            for rid, n in counts.items():
                r = state['requests'][rid]
                if rid in state['waiting']:
                    state['waiting'].remove(rid)
                    state['running'].append(rid)
                r['computed'] += n
                r['in_flight_tokens'] += n
                if r['computed'] >= r['num_tokens'] + r['placeholders']:
                    r['placeholders'] += 1
                    emits.append(rid)
            # With no sharing/speculation, scheduled positions bound allocated
            # KV. Reject insufficient headroom; no preemption approximation.
            blocks = sum((r['computed'] + scope.block_size - 1) // scope.block_size
                         for rid, r in state['requests'].items() if rid in state['running'])
            if blocks > scope.kv_blocks:
                raise ValueError('KV capacity exceeded; preemption is outside model scope')
            peak_blocks = max(peak_blocks, blocks)
            if work:
                _, forward = graph.launch('forward', costs.launches['forward'], row, batch=bid)
                _, logits = graph.launch('logits_sample', costs.launches['logits_sample'], row, batch=bid)
                enqueue = graph.add('copy_enqueue', costs.scalar['copy_enqueue_cpu'](row),
                                    resource='cpu', batch=bid)
                copied = graph.add('copy_service', costs.scalar['copy_service'](row),
                                   resource='copy', dependencies=[enqueue, logits], batch=bid)
                _, update = graph.launch('post_update', costs.launches['post_update'], row, batch=bid)
                step = dict(batch=bid, step_id=digest(sorted((r['request_id'], r['position'], r['tokens']) for r in work)),
                            work=work, state_before=before, features=row, emits=emits,
                            scheduled_s=graph.nodes[schedule]['start_s'],
                            forward_node=forward, logits_node=logits,
                            copy_node=copied, update_node=update)
                steps.append(step)
                ready = graph.end(copied)
            else:
                empty = graph.add('empty_submit', costs.scalar['empty_submit_cpu'](row),
                                  resource='cpu', batch=bid)
                step, ready = None, graph.end(empty)
            queue.append(dict(batch=bid, step=step, counts=counts, emits=emits, ready_s=ready, row=row))
            now = graph.end(graph.resources['cpu'])
            peak_slots = max(peak_slots, len(queue))
            actions.append(dict(operation='submit', batch=bid, empty=not bool(work),
                                at_s=now, occupied_slots=len(queue)))
            if len(queue) < scope.batch_slots:
                now = finish_iteration(executed, now, bid)
                continue  # Submit first, even if the oldest result is already ready.

        oldest = queue.popleft()
        if oldest['step'] is not None:
            materialize = graph.add('output_ready', costs.scalar['output_ready_cpu'](oldest['row']),
                                    resource='cpu', dependencies=[oldest['step']['copy_node']],
                                    ready=now, batch=oldest['batch'])
            commit = graph.add('commit', costs.scalar['commit_cpu'](oldest['row']),
                               resource='cpu', dependencies=[materialize], batch=oldest['batch'])
            now = graph.end(commit)
            oldest['step'].update(commit_node=commit, committed_s=now)
        else:
            now = max(now, oldest['ready_s'])
        for rid, n in oldest['counts'].items():
            state['requests'][rid]['in_flight_tokens'] -= n
        for rid in oldest['emits']:
            r, result = state['requests'][rid], outcomes[rid]
            r['placeholders'] -= 1
            r['num_tokens'] += 1
            result['output_tokens'] += 1
            if result['first_s'] is None:
                result['first_s'] = now
            if result['output_tokens'] == r['max_tokens']:
                if r['placeholders'] or r['in_flight_tokens']:
                    raise RuntimeError('Terminal request still has predicted work in flight')
                result['finished_s'] = now
                state['running'].remove(rid)
                del state['requests'][rid]
        actions.append(dict(operation='reap', batch=oldest['batch'], at_s=now,
                            occupied_slots=len(queue)))
        if release is None and all(outcomes[rid]['first_s'] is not None for rid in backgrounds):
            release_foreground(now)
        now = finish_iteration(executed, now, iteration)
    else:
        raise RuntimeError('Async rollout exceeded iteration limit')

    for result in outcomes.values():
        result['ttft_s'] = result['first_s'] - result['arrival_s']
        result['engine_ttft_s'] = result['first_s'] - result['enqueue_s']
        result['input_queue_wait_s'] = result['enqueue_s'] - result['arrival_s']
    foreground_done = max(outcomes[rid]['finished_s'] for rid in foregrounds)
    background_outlives = all(outcomes[rid]['finished_s'] >= foreground_done for rid in backgrounds)
    return dict(schema='causal-async-prediction-v1', status='unfitted_prototype',
                simulation_complete=True, model_accepted=False, cuda_scope_accepted=False,
                complete_experiment_passed=False, adaptive_cap_accepted=False, mpc_accepted=False,
                cost_sha256=digest(costs.to_dict()), dispatcher_sha256=digest(dispatch_table),
                scope=dict(vars(scope)), release_s=release, cpu_drained_s=now,
                resources_drained_s=max(now, max(n['end_s'] for n in graph.nodes)),
                peak_slots=peak_slots, peak_kv_blocks=peak_blocks,
                background_outlives_foreground=background_outlives,
                requests=outcomes, steps=steps, actions=actions, nodes=graph.nodes)


def rollout(case, costs, dispatch_table, **kwargs):
    """Adapter for declared controlled cases; background barrier is simulated."""
    if len(case['foreground_tokens']) != len(case['foreground_offsets_ms']):
        raise ValueError('Foreground lengths and arrival offsets must align')
    count = case['background_decoders']
    if type(count) is not int or count not in (0, 16):
        raise ValueError('Only declared D0/D16 cases are supported')
    requests = [Request(f'background-{i}', 1024, 256, role='background') for i in range(count)]
    requests += [Request(f'foreground-{i}', n, 128, nonnegative(at) / 1000.)
                 for i, (n, at) in enumerate(zip(case['foreground_tokens'], case['foreground_offsets_ms']))]
    if case['sentinel']:
        sentinel = case['sentinel']
        requests.append(Request('sentinel', sentinel['input_tokens'], 128,
                                nonnegative(sentinel['arrival_ms']) / 1000., 'sentinel'))
    return simulate(requests, costs, dispatch_table, **kwargs)


def compare_structure(prediction, capture):
    """Independent post-prediction comparison; never read captured timing labels.

    This is development scheduling evidence, not calibration/held accuracy or
    qualification of modes, source identity, measurement scope or cap policies.
    Capture integrity/source audits must be established separately.
    """
    def short(rid):
        return rid.rsplit('/', 1)[-1]

    def work(step):
        return [(short(r['request_id']), r['position'], r['tokens']) for r in step['work']]

    expected, observed = prediction['steps'], capture['steps']
    mismatch = dict(work=[], state=[], features=[], dispatcher=[], output_progression=[])
    workers = [r for w in capture['workers'] for r in w['rows']]
    by_id = {r['step_id']: r for r in workers}
    ids = [s['step_id'] for s in observed]
    identities_valid = (len(by_id) == len(workers) == len(observed)
                        and len(set(ids)) == len(ids) and set(by_id) == set(ids))
    pipeline = [r for w in capture['workers'] for r in w.get('pipeline', {}).get('rows', [])]
    pipeline_by_id = {r['step_id']: r for r in pipeline}
    progression_available = (len(pipeline_by_id) == len(pipeline) == len(observed)
                             and set(pipeline_by_id) == set(ids))
    fields = ('computed', 'num_tokens', 'placeholders', 'prompt_tokens', 'max_tokens')
    for index, (a, b) in enumerate(zip(expected, observed)):
        if work(a) != work(b):
            mismatch['work'].append(index)
        actual_state = {short(k): {f: v[f] for f in fields}
                        for k,v in b['state_before']['requests'].items()}
        if any({f: v[f] for f in fields} != actual_state.get(rid)
               for rid,v in a['state_before'].items()):
            mismatch['state'].append(index)
        if any(a['features'][f] != b[f] for f in ('prefill','decode','decode_context','attention_pairs')):
            mismatch['features'].append(index)
        worker = by_id.get(b['step_id'])
        if (worker is None or a['features']['mode'] != worker['runtime_mode']
                or a['features']['padding_excess'] != worker['padded_tokens'] - b['prefill'] - b['decode']):
            mismatch['dispatcher'].append(index)
        output = pipeline_by_id.get(b['step_id'])
        if output is None or {short(k): v for k,v in output['committed_outputs'].items()} != {rid:1 for rid in a['emits']}:
            mismatch['output_progression'].append(index)
    actual_requests = {short(k): v['output_tokens'] for k,v in capture['requests'].items()}
    output_counts_match = (len(actual_requests) == len(capture['requests'])
                           and actual_requests == {k:v['output_tokens'] for k,v in prediction['requests'].items()})
    matched = (capture['status'] == 'complete' and identities_valid and progression_available
               and len(expected) == len(observed) and output_counts_match
               and not any(mismatch.values()))
    return dict(scope='development scheduling comparison only; no timing accuracy',
                matches=matched, predicted_steps=len(expected), observed_steps=len(observed),
                step_identities_valid=identities_valid, output_counts_match=output_counts_match,
                output_progression_checked=progression_available,
                mismatches=mismatch, captured_timing_labels_used=False,
                model_accepted=False, complete_experiment_passed=False)
