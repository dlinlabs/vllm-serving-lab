"""Gateway observations; all mutable state is owned by the event-loop thread."""
import asyncio
import codecs
from collections import Counter
import json
from pathlib import Path
import time
import uuid


class SSEObserver:
    """Observe complete SSE events without delaying forwarding of byte chunks."""
    def __init__(self):
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
                    self.has_content |= any(
                        bool(choice.get('delta', {}).get('content'))
                        for choice in event.get('choices', [])
                    )
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

    def received(self):
        request_id = uuid.uuid4().hex
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

    def sample(self, metrics):
        now = time.monotonic()
        awaiting = [x for x in self.active.values() if x['admitted'] and x['first'] is None]
        counts = {k: self.totals[k] - self.previous[k] for k in
                  ('received', 'accepted', 'rejected', 'completed', 'failed', 'cancelled')}
        result = dict(interval_seconds=now - self.last_sample, counts=counts,
                      totals=dict(self.totals), awaiting_first_content=len(awaiting),
                      oldest_awaiting_first_seconds=max((now - x['received'] for x in awaiting), default=0),
                      streaming_requests=sum(x['first'] is not None for x in self.active.values()),
                      active_requests=len(self.active), dropped_records=self.dropped,
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
