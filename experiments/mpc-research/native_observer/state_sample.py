"""Receipt-stamped serving observations. This grants no actuation authority."""
from dataclasses import asdict, dataclass
import math
import re
import time

from mpc_lab.runtime import local_clock_domain, snapshot_from_dict
from mpc_lab.controller import validate_snapshot
from mpc_lab.contracts import CostSpec


@dataclass(frozen=True)
class WorkObservation:
    request_id: str
    charged_cost: float
    cost_prompt_tokens: int
    physical_prompt_tokens: int | None
    output_tokens: int | None
    max_output_tokens: int
    gateway_received_at: float


@dataclass(frozen=True)
class ServingObservation:
    gateway: object
    native_run_id: str
    observer_sha256: str
    gateway_received_at: float
    native_received_at: float
    observations: tuple[WorkObservation, ...]
    running: int
    waiting: int
    pending_batches: int
    kv_cache_usage: float


def join_observations(document, native, *, identity, gateway_run, native_run,
                      observer_sha, domain, gateway_received_at, native_received_at,
                      now, max_age_s=.5):
    if (not isinstance(document,dict) or not isinstance(native,dict)
            or document.get("clock_domain") != domain or native.get("clock_domain") != domain
            or document.get("run_id") != gateway_run or native.get("run_id") != native_run
            or document.get("identity") != asdict(identity)
            or native.get("observer_sha256") != observer_sha):
        raise ValueError("observation_clock_or_identity")
    gateway = snapshot_from_dict(document)
    cost_spec = CostSpec(**document["cost_spec"])
    if any(not math.isclose(m.cost,cost_spec.estimate(m.prompt_tokens,m.max_tokens),rel_tol=1e-12)
           for m in gateway.requests):
        raise ValueError("observation_cost_spec_mismatch")
    reason = validate_snapshot(gateway,now=now,clock_domain=domain,
                               expected_identity=identity,max_age_s=max_age_s)
    if reason:
        raise ValueError(reason)
    times = (gateway_received_at,native_received_at,now,native.get("sampled_at"),
             native.get("frontend_sampled_at"))
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in times):
        raise ValueError("observation_time")
    if not (gateway.sampled_at <= gateway_received_at <= native_received_at <= now
            and 0 <= now-native["sampled_at"] <= max_age_s
            and native["sampled_at"] <= native["frontend_sampled_at"] <= native_received_at
            and native["frontend_sampled_at"]-native["sampled_at"] <= .05):
        raise ValueError("observation_stale_or_wide")
    if native.get("schema") != "pr33-native-read-only-v1" or native.get("complete") is not True:
        raise ValueError("native_incomplete")
    marks = {m.request_id:m for m in gateway.requests}
    joined = {}
    for request in native["requests"]:
        match = re.fullmatch(r"chatcmpl-([0-9a-f]{32})-[0-9a-f]{8}",request["request_id"])
        if not match or match[1] not in marks or match[1] in joined:
            raise ValueError("native_unjoined_or_ambiguous")
        mark = marks[match[1]]
        if (type(request["prompt_tokens"]) is not int or request["prompt_tokens"] < 0
                or type(request["output_tokens"]) is not int
                or not 0 <= request["output_tokens"] <= mark.max_tokens
                or request["maximum_output_tokens"] != mark.max_tokens):
            raise ValueError("native_work_invalid")
        joined[mark.request_id] = request
    work = tuple(WorkObservation(m.request_id,m.cost,m.prompt_tokens,
        joined[m.request_id]["prompt_tokens"] if m.request_id in joined else None,
        joined[m.request_id]["output_tokens"] if m.request_id in joined else None,
        m.max_tokens,m.received_at) for m in gateway.requests)
    for key in ("running","waiting","pending_batches"):
        if type(native.get(key)) is not int or native[key] < 0:
            raise ValueError("native_count")
    usage = native.get("kv_cache_usage")
    if type(usage) not in (int,float) or not math.isfinite(usage) or not 0 <= usage <= 1:
        raise ValueError("native_kv_usage")
    if native["running"] != len(native["requests"]):
        raise ValueError("native_running_mismatch")
    return ServingObservation(gateway,native_run,observer_sha,gateway_received_at,
        native_received_at,work,native["running"],native["waiting"],
        native["pending_batches"],float(usage))


async def sample_serving(gateway_client, native_client, *, observer_key, identity,
                         gateway_run, native_run, observer_sha, clock=time.monotonic,
                         domain=None):
    domain = domain or local_clock_domain()
    gateway_reply = await gateway_client.get("/admin/mpc/state")
    gateway_reply.raise_for_status()
    gateway_received_at = clock()
    native_reply = await native_client.get("/pr33-native/state",headers={"x-observer-key":observer_key})
    native_reply.raise_for_status()
    native_received_at = clock()
    return join_observations(gateway_reply.json(),native_reply.json(),identity=identity,
        gateway_run=gateway_run,native_run=native_run,observer_sha=observer_sha,domain=domain,
        gateway_received_at=gateway_received_at,native_received_at=native_received_at,now=clock())
