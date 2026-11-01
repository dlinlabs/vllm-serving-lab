from dataclasses import asdict

import pytest

from mpc_lab.contracts import ModelParameters, RequestMark, Snapshot
from mpc_lab.runtime import snapshot_to_dict
from state_sample import join_observations


def fixture():
    parameters = ModelParameters()
    cost = parameters.cost_spec.estimate(114,128)
    mark = RequestMark("a"*32,cost,114,128,9.0,9.0)
    snapshot = Snapshot("gateway","clock",10.0,16,1,(mark,),parameters.identity)
    document = snapshot_to_dict(snapshot)
    document["cost_spec"] = asdict(parameters.cost_spec)
    native = dict(schema="pr33-native-read-only-v1",run_id="engine",clock_domain="clock",
        observer_sha256="source",sampled_at=10.02,frontend_sampled_at=10.03,
        complete=True,running=1,waiting=0,pending_batches=1,kv_cache_usage=.1,
        requests=[dict(request_id="chatcmpl-"+"a"*32+"-12345678",prompt_tokens=128,
                       output_tokens=5,maximum_output_tokens=128,computed_tokens=1000000)])
    kwargs = dict(identity=parameters.identity,gateway_run="gateway",native_run="engine",
                  observer_sha="source",domain="clock",gateway_received_at=10.01,
                  native_received_at=10.04,now=10.05)
    return document,native,kwargs


def test_physical_work_does_not_rewrite_sealed_cost_tokens():
    document,native,kwargs = fixture()
    sample = join_observations(document,native,**kwargs)
    request = sample.observations[0]
    assert request.cost_prompt_tokens == 114
    assert request.physical_prompt_tokens == 128
    assert request.charged_cost == document["requests"][0]["cost"]
    assert request.output_tokens == 5


def test_clock_identity_is_checked_before_foreign_request_chronology():
    document,native,kwargs = fixture()
    document["clock_domain"] = "foreign-boot"
    document["requests"][0]["received_at"] = -100000
    with pytest.raises(ValueError,match="observation_clock_or_identity"):
        join_observations(document,native,**kwargs)


def test_stale_native_and_unjoined_request_fail_closed():
    document,native,kwargs = fixture()
    native["sampled_at"] = 9.0
    with pytest.raises(ValueError,match="observation_stale_or_wide"):
        join_observations(document,native,**kwargs)
    native["sampled_at"] = 10.02
    native["requests"][0]["request_id"] = "chatcmpl-"+"b"*32+"-12345678"
    with pytest.raises(ValueError,match="native_unjoined_or_ambiguous"):
        join_observations(document,native,**kwargs)


def test_gateway_mark_not_yet_observed_natively_has_unknown_physical_work():
    document,native,kwargs = fixture()
    native["requests"],native["running"] = [],0
    sample = join_observations(document,native,**kwargs)
    assert sample.observations[0].physical_prompt_tokens is None
    assert sample.observations[0].output_tokens is None
