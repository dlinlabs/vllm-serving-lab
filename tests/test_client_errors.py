import asyncio
import httpx
import pytest
from sustained_load import consume_request, WORKLOADS


@pytest.mark.parametrize('error', [httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout])
def test_failure_before_headers_is_recorded_without_retry(error):
    async def scenario():
        calls = []
        def handler(request):
            calls.append(request)
            raise error('transport detail')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await consume_request(client, 0, 'model', WORKLOADS[0])
        assert len(calls) == 1
        assert not result.success and not result.rejected
        assert result.status is None and result.request_id is None
        assert result.error_type == error.__name__
        assert result.error_message == 'transport detail'
        assert result.error_phase == 'before_response_headers'
    asyncio.run(scenario())


def test_failure_during_body_retains_gateway_id():
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            raise httpx.ReadError('connection reset')
    async def scenario():
        def handler(request):
            return httpx.Response(200, headers={'x-request-id': 'gateway-id'}, stream=BrokenStream())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await consume_request(client, 0, 'model', WORKLOADS[0])
        assert result.request_id == 'gateway-id' and result.status == 200
        assert result.ttft is not None and not result.success
        assert result.error_type == 'ReadError' and result.error_phase == 'response_body'
    asyncio.run(scenario())
