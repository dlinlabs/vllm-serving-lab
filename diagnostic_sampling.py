"""Optional external observations. HTTP, OS reads and persistence run on one worker thread."""
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit

import httpx

from telemetry import DIAGNOSTICS_SCHEMA

MAX_METRICS_BYTES = 4 * 1024 * 1024
METRICS_BODY_TIMEOUT_SECONDS = 5.0

REQUIRED_METRICS = {
    'waiting': ('gauge', ('vllm:num_requests_waiting',)),
    'running': ('gauge', ('vllm:num_requests_running',)),
    'queue_time': ('histogram', ('vllm:request_queue_time_seconds',)),
    'prefill_time': ('histogram', ('vllm:request_prefill_time_seconds',)),
    'decode_time': ('histogram', ('vllm:request_decode_time_seconds',)),
    'kv_cache': ('gauge', ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')),
    'preemptions': ('counter', ('vllm:num_preemptions',)),
    'prompt_tokens': ('counter', ('vllm:prompt_tokens',)),
    'generation_tokens': ('counter', ('vllm:generation_tokens',)),
}


def metric_inventory(text):
    from prometheus_client.parser import text_string_to_metric_families
    families = list(text_string_to_metric_families(text))
    inventory, counters = {}, {}
    for key, (expected_type, names) in REQUIRED_METRICS.items():
        family = next((f for f in families if f.name in names), None)
        samples = [] if family is None else [
            dict(name=s.name, labels=dict(s.labels), value=s.value) for s in family.samples]
        sample_names = {s['name'] for s in samples}
        reason = ('not_exported' if family is None else 'unexpected_type' if family.type != expected_type
                  else 'no_samples' if not samples else 'nonfinite_samples' if
                  any(not math.isfinite(s['value']) for s in samples) else 'negative_samples' if
                  any(s['value'] < 0 for s in samples) else 'incomplete_histogram' if
                  expected_type == 'histogram' and not
                  {family.name + suffix for suffix in ('_bucket', '_count', '_sum')} <= sample_names else None)
        inventory[key] = dict(available=reason is None, reason=reason,
                              name=family.name if family else None,
                              type=family.type if family else None,
                              sample_names=sorted({s['name'] for s in samples}),
                              samples=samples if reason is None else [])
        if reason is None and family.type in {'counter', 'histogram'}:
            for sample in samples:
                if sample['name'].endswith(('_total', '_count', '_sum', '_bucket')):
                    identity = json.dumps([sample['name'], sample['labels']], sort_keys=True)
                    counters[identity] = sample['value']
    exporter = {s.name: s.value for f in families for s in f.samples
                if s.name in {'process_start_time_seconds', 'process_id'} and math.isfinite(s.value)}
    return inventory, counters, exporter


class ObservationSampler:
    def __init__(self, output, endpoint, gateway_pid, run_id, interval=1.0):
        self.output, self.endpoint = Path(output), endpoint.rstrip('/') + '/metrics'
        self.gateway_pid, self.run_id, self.interval = gateway_pid, run_id, interval
        parsed = urlsplit(self.endpoint)
        self.source = dict(scheme=parsed.scheme, host=parsed.hostname, port=parsed.port, path=parsed.path)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.previous_counters, self.previous_exporter = {}, None
        self.counter_epoch = 0
        self.status = dict(diagnostics_schema_version=DIAGNOSTICS_SCHEMA, run_id=run_id,
                           status='running', samples=0, sample_errors=0, write_errors=0,
                           interval_seconds=interval)

    @staticmethod
    def process_resources(pid):
        import psutil
        start = time.monotonic()
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                created = process.create_time()
                rss = process.memory_info().rss
                cpu = process.cpu_times()
            return dict(available=True, pid=pid, process_created_unix=created,
                        sample_start=start, sample_end=time.monotonic(), rss_bytes=rss,
                        user_cpu_seconds=cpu.user, system_cpu_seconds=cpu.system)
        except psutil.Error as exc:
            return dict(available=False, pid=pid, sample_start=start, sample_end=time.monotonic(),
                        error_type=type(exc).__name__)

    def sample(self, client):
        import psutil
        sent = time.monotonic()
        record = dict(diagnostics_schema_version=DIAGNOSTICS_SCHEMA, run_id=self.run_id,
                      source=self.source, host=socket.gethostname(), sent_at=sent,
                      raw_metrics=None, metrics={}, unavailable_metrics=list(REQUIRED_METRICS),
                      exporter_identity={}, counter_epoch=self.counter_epoch,
                      counter_resets=[], disappeared_counters=[], error_type=None)
        try:
            with client.stream('GET', self.endpoint, headers={'Accept-Encoding': 'identity'}) as response:
                record.update(headers_at=time.monotonic(), http_status=response.status_code)
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() - record['headers_at'] > METRICS_BODY_TIMEOUT_SECONDS:
                        record['error_reason'] = 'metrics_body_time_limit'
                        raise ValueError('Metrics body exceeded collection time limit')
                    if len(body) + len(chunk) > MAX_METRICS_BYTES:
                        record['error_reason'] = 'metrics_body_size_limit'
                        raise ValueError('Metrics body exceeded collection size limit')
                    body.extend(chunk)
                record.update(received_at=time.monotonic(), raw_metrics=body.decode('utf-8'))
            inventory, counters, exporter = metric_inventory(record['raw_metrics'])
            resets = [name for name, value in counters.items()
                      if name in self.previous_counters and value < self.previous_counters[name]]
            disappeared = sorted(self.previous_counters.keys() - counters.keys())
            identity_changed = self.previous_exporter is not None and exporter != self.previous_exporter
            if resets or disappeared or identity_changed:
                self.counter_epoch += 1
            record.update(metrics=inventory, unavailable_metrics=[k for k, v in inventory.items() if not v['available']],
                          exporter_identity=exporter, exporter_identity_changed=identity_changed,
                          counter_epoch=self.counter_epoch, counter_resets=resets, disappeared_counters=disappeared)
            self.previous_counters, self.previous_exporter = counters, exporter
        except (httpx.HTTPError, ValueError) as exc:
            record.setdefault('received_at', time.monotonic())
            record['error_type'] = type(exc).__name__
        host_start = time.monotonic()
        try:
            host_cpu = dict(available=True, sample_start=host_start, cpu_seconds=psutil.cpu_times()._asdict(),
                            logical_cpus=psutil.cpu_count(), sample_end=time.monotonic())
        except (OSError, NotImplementedError) as exc:
            host_cpu = dict(available=False, sample_start=host_start, sample_end=time.monotonic(),
                            error_type=type(exc).__name__)
        record['resources'] = dict(gateway=self.process_resources(self.gateway_pid),
                                   runner=self.process_resources(os.getpid()), host_cpu=host_cpu)
        record['finished_at'] = time.monotonic()
        record['interval_seconds'] = self.interval
        record['late_by_seconds'] = max(0., record['finished_at'] - sent - self.interval)
        return record

    def run(self):
        path = self.output / 'engine_observations.jsonl'
        self.status['started_at'] = time.monotonic()
        try:
            with path.open('w', encoding='utf-8') as stream, httpx.Client(timeout=2, trust_env=False) as client:
                while True:
                    final_sample = self.stop.is_set()
                    sample = self.sample(client)
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
                    stream.flush()
                    self.status['samples'] += 1
                    self.status['sample_errors'] += bool(sample['error_type'] or sample['unavailable_metrics'] or
                        not all(sample['resources'][name]['available'] for name in ('gateway', 'runner', 'host_cpu')))
                    self.ready.set()
                    if final_sample:
                        break
                    self.stop.wait(max(0., self.interval - (time.monotonic() - sample['sent_at'])))
                    # Even stop during an in-flight sample gets a fresh post-stop sample.
                self.status['status'] = 'closed'
        except OSError as exc:
            self.status.update(write_errors=self.status['write_errors'] + 1, error_type=type(exc).__name__)
            raise
        finally:
            self.ready.set()
            if self.status['status'] != 'closed':
                self.status['status'] = 'failed'
            self.status['closed_at'] = time.monotonic()
            status_path = self.output / 'engine_observations_status.json'
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.status, allow_nan=False) + '\n', encoding='utf-8')
            temporary.replace(status_path)
