"""Exercise probe selection after the pinned OpenAI request scalar conversion."""
import asyncio
import json
import httpx
import pytest
from pydantic import TypeAdapter

from prefill_lab import live_runtime
from prefill_lab.async_predictor import Scope
from test_prefill_envelope_shadow import manifest, tracker
from test_prefill_live import D, MODEL


# vLLM 0.30.0 CompletionRequest.vllm_xargs has no bool union member.
XARGS = TypeAdapter(dict[str, str | int | float | list[str | int | float]])


@pytest.mark.parametrize('kind', ['point', 'envelope'])
@pytest.mark.parametrize('flag', [True, 1])
def test_numeric_wire_probe_executes_before_native_enqueue(kind, flag):
    observer, request = tracker(lambda: 10.)
    if kind == 'point':
        observer = live_runtime.LiveTracker(observer.core, MODEL, Scope(), D, clock=lambda: 10., domain='test')
    extra = request.sampling_params.extra_args
    extra.update(prefill_lab_probe=flag, prefill_lab_role='sentinel')
    request.sampling_params.extra_args = XARGS.validate_json(json.dumps(extra))
    assert type(request.sampling_params.extra_args['prefill_lab_probe']) is int
    observer.before_add(request)
    observer.after_add(request)
    row = observer.records.rows[0]
    assert row['status'] == 'predicted'
    assert row['cut'] <= row['prediction_finished'] <= row['engine_enqueue']
    assert row['forecast'] is not None


@pytest.mark.parametrize('observed,flag,role', [
    (False, 1, 'sentinel'), (True, 0, 'background'), (True, 0, 'foreground'),
    (True, False, 'foreground'),
])
def test_off_and_nonprobe_requests_do_not_run_prediction(observed, flag, role):
    observer, request = tracker(lambda: 10., observed=observed)
    request.sampling_params.extra_args.update(prefill_lab_probe=flag, prefill_lab_role=role)
    observer.before_add(request)
    observer.after_add(request)
    row = observer.records.rows[0]
    assert row['status'] == 'off' and row['forecast'] is None
    assert 'prediction_started' not in row


@pytest.mark.parametrize('flag', [None, '1', 'true', 1.0, -1, 2, [], {}])
def test_malformed_probe_marker_is_not_silently_recorded_as_off(flag):
    observer, request = tracker(lambda: 10.)
    request.sampling_params.extra_args['prefill_lab_probe'] = flag
    with pytest.raises(ValueError, match='probe marker'):
        observer.before_add(request)
    assert not observer.records.rows


@pytest.mark.parametrize('flag,role', [(0, 'sentinel'), (1, 'foreground'), (1, 'background'), (1, 'unknown')])
def test_marker_role_mismatch_is_rejected(flag, role):
    observer, request = tracker(lambda: 10.)
    request.sampling_params.extra_args.update(prefill_lab_probe=flag, prefill_lab_role=role)
    with pytest.raises(ValueError, match='probe role'):
        observer.before_add(request)


def test_missing_probe_marker_is_rejected():
    observer, request = tracker(lambda: 10.)
    del request.sampling_params.extra_args['prefill_lab_probe']
    with pytest.raises(ValueError, match='probe marker'):
        observer.before_add(request)


def test_actual_http_offer_roundtrip_runs_shadow_prediction(monkeypatch, tmp_path):
    from scripts import prefill_live_http_trial as trial
    from prefill_lab import plugin
    import time

    monkeypatch.setattr(trial, 'clock_domain', lambda: 'test')
    observer, request = tracker(time.monotonic)
    observer.core._prefill_live = observer
    monkeypatch.setattr(plugin, 'ExternalCapture', type('UnusedCapture', (), {}))
    observer.core._prefill_trial.enqueued = set()
    entered = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"text":"x"}]}\n\n'
            yield b'data: [DONE]\n\n'

    def native_add(core, incoming):
        entered.append(observer.records.rows[0]['prediction_finished'])
        assert incoming is request

    def backend(message):
        payload = json.loads(message.content)
        assert type(payload['vllm_xargs']['prefill_lab_probe']) is int
        request.sampling_params.extra_args = XARGS.validate_python(payload['vllm_xargs'])
        assert request.sampling_params.extra_args['prefill_lab_role'] == 'sentinel'
        plugin.guarded_add(observer.core, request, native_add)
        assert not observer.errors
        return httpx.Response(200, stream=Body(), headers={
            'content-type': 'text/event-stream',
            'x-request-id': message.headers['x-benchmark-request-id'],
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
            await trial.run(http, 'http://test/v1/completions', manifest(), 'es-001', tmp_path/'client.json')

    asyncio.run(run())
    assert len(entered) == 1
    row = observer.records.rows[0]
    assert row['status'] == 'predicted' and row['forecast'] is not None
    assert row['cut'] <= entered[0] <= row['engine_enqueue']
    assert json.loads((tmp_path/'client.json').read_text())['rows'][0]['status'] == 'complete'


def test_bad_marker_records_instrumentation_error_without_changing_native_add():
    from prefill_lab import plugin
    observer, request = tracker(lambda: 10.)
    observer.core._prefill_live = observer
    observer.core._prefill_trial.enqueued = set()
    del request.sampling_params.extra_args['prefill_lab_probe']
    added = []
    plugin.guarded_add(observer.core, request, lambda core, incoming: added.append(incoming))
    assert added == [request] and not observer.records.rows
    assert len(observer.errors) == 1 and 'probe marker' in observer.errors[0]


def test_installed_vllm_completion_request_preserves_numeric_probe_selection():
    from importlib.util import find_spec
    if find_spec('vllm') is None:
        pytest.skip('Native vLLM is not installed in this CPU-only environment')
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    observer, request = tracker(lambda: 10.)
    extra = dict(request.sampling_params.extra_args, prefill_lab_probe=1, prefill_lab_role='sentinel')
    parsed = CompletionRequest(model='test', prompt=[1], max_tokens=128, vllm_xargs=extra)
    request.sampling_params.extra_args = parsed.vllm_xargs
    observer.before_add(request)
    observer.after_add(request)
    assert observer.records.rows[0]['status'] == 'predicted'
