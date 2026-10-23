"""CPU-only capture contracts shared by the engine, worker and offline audit."""
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical(value) + b'\n')
    temporary.replace(path)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Buffer:
    """Overflow invalidates the capture; no dropping the oldest observation."""
    def __init__(self, capacity=8192):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Invalid buffer capacity')
        self.capacity, self.rows, self.overflow = capacity, [], 0

    def append(self, value):
        if len(self.rows) == self.capacity:
            self.overflow += 1
            raise RuntimeError('Capture buffer overflow')
        self.rows.append(value)


def scheduled_work(output):
    """Read PRE-execution positions from SchedulerOutput, including async batches."""
    positions = {r.req_id: (r.num_computed_tokens, len(r.prompt_token_ids))
                 for r in output.scheduled_new_reqs}
    cached = output.scheduled_cached_reqs
    if len(cached.req_ids) != len(cached.num_computed_tokens):
        raise ValueError('Malformed cached request positions')
    for rid, pos in zip(cached.req_ids, cached.num_computed_tokens):
        if rid in positions:
            raise ValueError('Duplicate new/cached request')
        positions[rid] = (pos, None)
    counts = output.num_scheduled_tokens
    if sum(counts.values()) != output.total_num_scheduled_tokens:
        raise ValueError('Scheduled token total mismatch')
    if set(counts) - set(positions):
        raise ValueError('Scheduled request missing its pre-step position')
    rows = []
    for rid, count in counts.items():
        pos, length = positions[rid]
        if type(pos) is not int or type(count) is not int or pos < 0 or count <= 0:
            raise ValueError('Invalid scheduled work')
        rows.append(dict(request_id=rid, position=pos, tokens=count, prompt_length=length))
    return rows


def step_key(output):
    return digest(sorted((r['request_id'], r['position'], r['tokens']) for r in scheduled_work(output)))


def features(work, prompt_lengths):
    chunks, decode, context = [], 0, 0
    for r in work:
        length = prompt_lengths[r['request_id']]
        q = min(r['tokens'], max(0, length-r['position']))
        d = r['tokens']-q
        if d not in (0, 1) or (q and d):
            raise ValueError('Speculative/mixed prompt-decode work is outside this protocol')
        if q:
            chunks.append((r['request_id'], q, r['position']))
        if d:
            decode += d
            context += r['position']
    return dict(chunks=chunks, prefill=sum(c[1] for c in chunks), decode=decode,
                decode_context=context,
                attention_pairs=sum(q*l+q*(q+1)/2 for _, q, l in chunks))


def fifo_reference(state, budget, max_seqs, long_threshold=0):
    """Restricted running-first FCFS reference, checked against installed scheduling.

    No prefix cache, speculation, preemption, encoder, LoRA or KV transfer. Any
    unsupported installed event invalidates parity rather than being approximated.
    State is a BEFORE-schedule snapshot, not later observed chunk decisions.
    """
    remaining, answer = budget, {}
    running = list(state['running'])
    for rid in running + list(state['waiting']):
        r = state['requests'][rid]
        if rid not in running and len(running) >= max_seqs:
            break
        if r['placeholders']>0 and r['computed']+2-r['placeholders'] >= r['prompt_tokens']+r['max_tokens']:
            continue
        need = r['num_tokens'] + r['placeholders'] - r['computed']
        n = min(remaining, need)
        if long_threshold > 0:
            n = min(n, long_threshold)
        if n <= 0:
            continue
        answer[rid] = n
        remaining -= n
        if rid not in running:
            running.append(rid)
        if remaining == 0:
            break
    return answer


def audit_capture(capture, worker_rows, *, require_steps=True, forced_outputs=True):
    """Reconcile all IDs/progress/timing scopes before any numerical model score."""
    problems = list(capture.get('errors', []))
    steps = capture.get('steps', [])
    if any(w.get('overflow',0) or w.get('pending',0) for w in capture.get('workers',[])):
        problems.append('Worker buffer overflow or unresolved events')
    if capture.get('overflow') or capture.get('status') != 'complete':
        problems.append('Incomplete/overflowed capture')
    if require_steps and not steps:
        problems.append('No executed-step evidence')
    workers = {}
    for row in worker_rows:
        key = row['step_id']
        if row.get('gpu_error') or not finite(row.get('gpu_forward_s')) or row.get('gpu_forward_s',0)<0:
            problems.append('Invalid/unavailable GPU forward span')
        if key in workers:
            problems.append('Duplicate worker step ID')
        workers[key] = row
    seen, progress, last_end = set(), {}, -1
    overlap = False
    for s in steps:
        key = s['step_id']
        if key in seen:
            problems.append('Duplicate scheduler step ID')
        seen.add(key)
        if not finite(s.get('start')) or not finite(s.get('end')) or s['end'] < s['start']:
            problems.append('Missing/invalid iteration wall timestamps')
        elif s['start'] < last_end:
            overlap = True
        last_end = max(last_end, s.get('end', -1))
        if not s.get('parity', {}).get('matches'):
            problems.append('Installed schedule differs from restricted FIFO reference')
        for r in s['work']:
            rid = r['request_id']
            if rid in progress and r['position'] != progress[rid]:
                problems.append('Executed work has a gap, repeat or cancellation')
            if rid not in progress and r['position'] != 0:
                problems.append('Unexpected cached/reused initial work')
            progress[rid] = r['position']+r['tokens']
        worker = workers.get(key)
        if require_steps and worker is None:
            problems.append('Missing worker step ID')
        if worker:
            counts={r['request_id']:r['tokens'] for r in s['work']}
            positions={r['request_id']:r['position'] for r in s['work']}
            if worker.get('counts')!=counts or worker.get('executed_counts')!=counts or worker.get('actual_tokens')!=sum(counts.values()):
                problems.append('Scheduler/worker work mismatch')
            if worker.get('executed_positions')!=positions:
                problems.append('Scheduler/worker input position mismatch')
    if set(workers)-seen:
        problems.append('Unmatched worker steps')
    for rid, r in capture.get('requests', {}).items():
        count=r.get('output_tokens',0)
        output_ok=count==r['max_tokens'] if forced_outputs else 0<count<=r['max_tokens']
        if not output_ok or r.get('finished') is None or r.get('finish_reason') in ('abort','error','repetition'):
            problems.append('Missing/short/cancelled request output')
        if require_steps and progress.get(rid) != r['input_tokens']+count-1:
            problems.append('Final input/output progress mismatch')
    return dict(valid=not problems, problems=sorted(set(problems)), async_overlap=overlap,
                wall_label_scope='schedule-start to matched output-commit; overlapping spans are not additive',
                gpu_label_scope='worker current-stream forward span; other streams require deep-trace qualification',
                step_count=len(steps), complete_experiment_passed=False)
