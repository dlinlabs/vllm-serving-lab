"""Loop-owned gateway observations, with an optional bounded GC callback inbox."""
import asyncio
import codecs
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
import gc
from itertools import count
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid


DIAGNOSTICS_SCHEMA = 1
INVESTIGATION_PROTOCOL = 'periodic-peaks-v1'
REQUEST_PHASES = ('body_read', 'json_parse', 'cost_estimation', 'admission_lock_wait',
                  'client_construction', 'http_send_headers', 'upstream_close',
                  'client_close', 'reservation_release')


class CallbackCounter:
    """CPython/GIL tickets; only the consumer reads, including its own probe tickets."""
    def __init__(self):
        self.tickets, self.probes = count(), 0

    def take(self):
        return next(self.tickets)

    def read(self):
        value = next(self.tickets) - self.probes
        self.probes += 1
        return value


class RequestDiagnostics:
    def __init__(self, emit, request_id):
        self.emit, self.request_id = emit, request_id
        self.started_at = time.monotonic()
        self.phases, self.milestones = {}, {}
        self.finished = False

    def begin(self, name):
        self.phases[name] = dict(start=time.monotonic(), end=None, status='incomplete')

    def end(self, name, error=None):
        phase = self.phases[name]
        phase.update(end=time.monotonic(), status=('completed' if error is None else
                     'cancelled' if isinstance(error, asyncio.CancelledError) else 'failed'))
        if error is not None:
            phase['error_type'] = type(error).__name__

    @contextmanager
    def phase(self, name):
        self.begin(name)
        try:
            yield
        except BaseException as exc:
            self.end(name, exc)
            raise
        else:
            self.end(name)

    def mark(self, name):
        if name not in self.milestones:
            self.milestones[name] = time.monotonic()

    def abort(self, error):
        for name, phase in self.phases.items():
            if phase['end'] is None:
                self.end(name, error)

    def finish(self, outcome, cleanup_complete):
        if self.finished:
            return
        self.finished = True
        now = time.monotonic()
        for phase in self.phases.values():
            if phase['end'] is None:
                phase.update(end=now, status='cancelled' if outcome == 'cancelled' else 'failed')
        self.emit('request_diagnostics', self.request_id, diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                  span_start=self.started_at, span_end=now, outcome=outcome,
                  cleanup_complete=cleanup_complete,
                  phases={k: self.phases.get(k, dict(start=None, end=None, status='not_reached'))
                          for k in REQUEST_PHASES},
                  first_body_at=self.milestones.get('first_body'),
                  first_content_at=self.milestones.get('first_content'))


class GatewayDiagnostics:
    """Bounded primitive-only deque; callbacks never call the loop-owned emitter."""
    def __init__(self, emit, capacity=4096):
        if sys.implementation.name != 'cpython' or not getattr(sys, '_is_gil_enabled', lambda: True)():
            raise RuntimeError('Gateway diagnostics require CPython with the GIL enabled')
        if capacity < 2:
            raise ValueError('GC callback capacity must be at least two')
        self.emit, self.capacity = emit, capacity
        self.inbox = deque(maxlen=capacity)
        self.entered, self.exited, self.produced, self.errors = (CallbackCounter() for _ in range(4))
        self.accepting = False
        self.callback = self._callback
        self.loop_thread_id = threading.get_ident()
        self.pending = defaultdict(list)
        self.pending_count = self.received = self.unmatched = self.spans = self.cpu_samples = 0
        self.closed = False
        self.quiesced = False
        self.registration_missing = False

    @staticmethod
    def gc_state():
        return dict(enabled=gc.isenabled(), thresholds=gc.get_threshold(), counts=gc.get_count())

    def start(self):
        self.coverage_start = time.monotonic()
        self.accepting = True
        gc.callbacks.append(self.callback)
        self.emit('diagnostics_started', diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                  coverage_start=self.coverage_start, process_id=os.getpid(),
                  loop_thread_id=self.loop_thread_id, loop_native_thread_id=threading.get_native_id(),
                  callback_capacity=self.capacity, python_implementation=sys.implementation.name,
                  gil_enabled=True, gc=self.gc_state())
        self.sample()

    def _callback(self, phase, info):
        self.entered.take()
        try:
            if not self.accepting:
                return
            sequence = self.produced.take()
            # Only built-in primitives: bounded eviction cannot execute user finalizers.
            self.inbox.append((sequence, phase, info['generation'], threading.get_ident(),
                               threading.get_native_id(), time.monotonic(), time.thread_time(),
                               info.get('collected'), info.get('uncollectable')))
        except MemoryError:
            self.errors.take()
        finally:
            self.exited.take()

    def _unmatched(self, edge, reason):
        self.unmatched += 1
        self.emit('gc_unmatched', diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                  sequence=edge[0], phase=edge[1], generation=edge[2], callback_thread_id=edge[3],
                  observed_at=edge[5], callback_thread_cpu_seconds=edge[6], reason=reason)

    def drain(self, final=False):
        edges = []
        for _ in range(self.capacity):
            try:
                edges.append(self.inbox.popleft())
            except IndexError:
                break
        for edge in sorted(edges):
            self.received += 1
            sequence, phase, generation, thread_id, native_id, at, cpu, collected, uncollectable = edge
            key = (thread_id, generation)
            if phase == 'start':
                if self.pending_count >= self.capacity:
                    self._unmatched(edge, 'pending_span_capacity_exceeded')
                else:
                    self.pending[key].append(edge)
                    self.pending_count += 1
            elif phase == 'stop' and self.pending.get(key):
                start = self.pending[key].pop()
                self.pending_count -= 1
                if not self.pending[key]:
                    del self.pending[key]
                self.spans += 1
                self.emit('gc_span', diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                          start_sequence=start[0], stop_sequence=sequence, generation=generation,
                          callback_thread_id=thread_id, callback_native_thread_id=native_id,
                          on_loop_thread=thread_id == self.loop_thread_id,
                          span_start=start[5], span_end=at, callback_thread_cpu_seconds=cpu-start[6],
                          collected=collected, uncollectable=uncollectable)
            else:
                self._unmatched(edge, 'start_not_observed' if phase == 'stop' else 'unknown_phase')
        if final:
            for pending in self.pending.values():
                for edge in pending:
                    self._unmatched(edge, 'stop_not_observed')
            self.pending.clear()
            self.pending_count = 0

    def health(self):
        produced = self.produced.read()
        inflight = max(0, self.entered.read() - self.exited.read())
        result = dict(callback_capacity=self.capacity, callbacks_produced=produced,
                      callbacks_received=self.received, buffered_callbacks=len(self.inbox),
                      inflight_callbacks=inflight, callback_errors=self.errors.read(),
                      pending_spans=self.pending_count, unmatched_boundaries=self.unmatched,
                      gc_spans=self.spans, cpu_samples=self.cpu_samples, closed=self.closed,
                      quiesced=self.quiesced, registration_missing=self.registration_missing)
        # Before quiescence, undelivered includes in-flight callbacks, not just eviction.
        result['undelivered_callbacks'] = produced - self.received
        result['lost_callbacks'] = produced - self.received if self.closed and self.quiesced else None
        result['complete'] = (self.closed and self.quiesced and not self.registration_missing and
                              not any(result[k] for k in ('undelivered_callbacks', 'buffered_callbacks',
                                  'inflight_callbacks', 'callback_errors', 'pending_spans', 'unmatched_boundaries')))
        return result

    def sample(self):
        if self.closed:
            return
        self.drain()
        self.cpu_samples += 1
        self.emit('diagnostic_cpu', diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                  sampled_at=time.monotonic(), loop_thread_id=threading.get_ident(),
                  loop_thread_cpu_seconds=time.thread_time(), process_cpu_seconds=time.process_time(),
                  gc=self.gc_state(), health=self.health())

    async def stop(self, timeout=2.0):
        self.accepting = False
        self.coverage_end = time.monotonic()
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)
        else:
            self.registration_missing = True
        deadline = time.monotonic() + timeout
        while self.entered.read() != self.exited.read():
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(.001)
        else:
            self.quiesced = True
        self.sample()
        self.drain(final=True)
        self.closed = True
        self.emit('diagnostics_stopped', diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
                  coverage_start=self.coverage_start, coverage_end=self.coverage_end, health=self.health())

    def request(self, request_id):
        return RequestDiagnostics(self.emit, request_id)


def has_sse_content(event, *, completions=False):
    """Validate text-stream fields before counting any chunk as first content."""
    if not isinstance(event, dict):
        raise ValueError('SSE payload must be an object')
    if event.get('error') is not None:
        raise ValueError('SSE reports an upstream error')
    usage = event.get('usage')
    if usage is not None and not isinstance(usage, dict):
        raise ValueError('SSE usage must be an object or null')
    choices = event.get('choices', [])
    if not isinstance(choices, list):
        raise ValueError('SSE choices must be a list')
    content_seen = False
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get('delta', {}), dict):
            raise ValueError('SSE choice and delta must be objects')
        content = choice.get('text') if completions else choice.get('delta', {}).get('content')
        if content is not None and not isinstance(content, str):
            raise ValueError('SSE text content must be a string or null')
        content_seen |= bool(content)
    return content_seen


class SSEObserver:
    """Observe complete SSE events without delaying forwarding of byte chunks."""
    def __init__(self, *, completions=False):
        self.completions = completions
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.buffer = ''
        self.data = []
        self.done = False
        self.has_content = False
        self.invalid = False

    def feed(self, chunk):
        self.buffer += self.decoder.decode(chunk)
        # Bound observation memory for malformed/non-SSE responses.
        if len(self.buffer) + sum(map(len, self.data)) > 1024 * 1024:
            self.invalid = True
            self.buffer = ''
            self.data.clear()
            return
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.rstrip('\r')
            if line.startswith('data:'):
                self.data.append(line[5:].lstrip(' '))
            elif not line and self.data:
                data = '\n'.join(self.data)
                self.data.clear()
                if data == '[DONE]':
                    self.done = True
                    continue
                try:
                    event = json.loads(data)
                    self.has_content |= has_sse_content(event, completions=self.completions)
                except (ValueError, AttributeError, TypeError):
                    self.invalid = True


class Telemetry:
    def __init__(self):
        self.run_id = uuid.uuid4().hex
        self.active = {}
        self.totals = Counter()
        self.previous = Counter()
        self.records = []
        self.enabled = False
        self.dropped = 0
        self.write_errors = 0
        self.last_sample = time.monotonic()
        self.diagnostics = None

    def emit(self, event, request_id=None, **fields):
        record = dict(event=event, run_id=self.run_id, timestamp=time.time(),
                      monotonic=time.monotonic(), **fields)
        if request_id is not None:
            record['request_id'] = request_id
        if self.enabled:
            if len(self.records) < 100000:
                self.records.append(record)
            else:
                self.dropped += 1

    def received(self, request_id=None):
        request_id = request_id or uuid.uuid4().hex
        self.active[request_id] = {'received': time.monotonic(), 'admitted': False, 'first': None}
        self.totals['received'] += 1
        self.emit('received', request_id)
        return request_id

    def admitted(self, request_id, **fields):
        self.active[request_id]['admitted'] = True
        self.totals['accepted'] += 1
        self.emit('accepted', request_id, **fields)

    def first_content(self, request_id):
        entry = self.active[request_id]
        if entry['first'] is None:
            entry['first'] = time.monotonic()
            self.emit('first_content', request_id, ttft=entry['first'] - entry['received'])

    def terminal(self, request_id, outcome, **fields):
        entry = self.active.pop(request_id, None)
        if entry is None:
            return
        self.totals[outcome] += 1
        self.emit(outcome, request_id, end_to_end=time.monotonic() - entry['received'], **fields)

    def observed_state(self, now=None):
        """Read current proxies without consuming interval counters or emitting a sample."""
        now = time.monotonic() if now is None else now
        awaiting = [x for x in self.active.values() if x['admitted'] and x['first'] is None]
        return dict(awaiting_first_content=len(awaiting),
                    oldest_awaiting_first_seconds=max((now - x['received'] for x in awaiting), default=0),
                    streaming_requests=sum(x['first'] is not None for x in self.active.values()),
                    active_requests=len(self.active))

    def sample(self, metrics):
        if self.diagnostics is not None:
            self.diagnostics.sample()
        now = time.monotonic()
        counts = {k: self.totals[k] - self.previous[k] for k in
                  ('received', 'accepted', 'rejected', 'completed', 'failed', 'cancelled')}
        result = dict(interval_seconds=now - self.last_sample, counts=counts,
                      totals=dict(self.totals), **self.observed_state(now), dropped_records=self.dropped,
                      telemetry_write_errors=self.write_errors)
        result.update(metrics)
        self.previous = self.totals.copy()
        self.last_sample = now
        self.emit('snapshot', **result)
        return result

    async def flush(self, path):
        records, self.records = self.records, []
        if not records:
            return
        def write():
            with Path(path).open('a', encoding='utf-8') as handle:
                for row in records:
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
        try:
            await asyncio.to_thread(write)
        except OSError:
            self.write_errors += 1
            self.dropped += len(records)

    async def run(self, path, snapshot, stop):
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self.sample(await snapshot())
            await self.flush(path)

    async def monitor_loop(self, stop, interval=0.1):
        """Record delayed execution after resuming; do not create catch-up samples."""
        while not stop.is_set():
            due = time.monotonic() + interval
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                self.emit('event_loop_lag', scheduled_at=due,
                          lag_seconds=max(0.0, time.monotonic() - due))
