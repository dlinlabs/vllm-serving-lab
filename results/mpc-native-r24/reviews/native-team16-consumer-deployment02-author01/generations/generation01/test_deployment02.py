"""Nine fresh wiring controls; no real service/main, valid approval or Native world."""

import ast
import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

import authorization02 as auth
import client01 as client
import composition_common01 as c
import launch01 as launch
import service01 as service
from owned01 import Owned

MARKER = "CPU-NOT-CREDENTIAL-" + "0" * 45


def run(awaitable, directory):
    errors = []
    async def keep():
        try:
            return await awaitable
        except BaseException as error:
            errors.append(error)
            raise
    result, joined = client.bounded_loop(keep(), directory)
    if errors:
        raise errors[0]
    assert joined
    return result


def invocation():
    start = time.monotonic()
    return dict(host_started_at=start, startup_deadline=start + 180, outer_deadline=start + 260,
        cleanup_hard_deadline=start + 243, clock_domain=c.domain(),
        consumer_manifest_sha256=c.CONSUMER_SEAL)


async def new_fixture(monkeypatch, path):
    fixture = c.consumer_module("fixture01").SourceFixture(monkeypatch, path)
    original = fixture.case.gateway["mpc_update_budget"]
    async def new_operation(request):
        command = await request.json()
        assert command["operation_id"] == "synthetic-bootstrap32"
        command["operation_id"] = c.protocol()["readiness_operation_id"]
        return await original(request)
    monkeypatch.setitem(fixture.case.gateway, "mpc_update_budget", new_operation)
    await fixture.case.initialize()
    return fixture


def test_C01_pre_host_authority_and_actual_default_off_call_placement(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "remote_only", lambda: pytest.fail("host reached"))
    monkeypatch.setattr(c, "owner", lambda: pytest.fail("procfs reached"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("helper reached"))
    results = []
    for role, name in (("launch","launch01.py"),("client","client01.py"),("service","service01.py")):
        assert auth.require_activation(False, None, role=role) is None
        with pytest.raises(ValueError, match="separate_Mayor"):
            auth.require_activation(True, None, role=role)
        with pytest.raises(ValueError, match="not_author_template"):
            auth.require_activation(True, c.HERE / "MAYOR_APPROVAL_TEMPLATE_INVALID.json", role=role)
        tree = ast.parse((c.HERE / name).read_bytes())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        gate = next(node.lineno for node in calls if isinstance(node.func, ast.Name)
                    and node.func.id == "require_activation")
        remote = next(node.lineno for node in calls if isinstance(node.func, ast.Attribute)
                      and node.func.attr == "remote_only")
        assert gate < remote
        disabled = next(node for node in main.body if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "activation is None")
        assert disabled.lineno < remote and any(isinstance(node, ast.Return) for node in disabled.body)
        assert not any(isinstance(node, ast.Raise) for node in main.body)
        assert "successor_CPU_patch_no_live_entrypoint" not in (c.HERE / name).read_text()
        results.append(dict(role=role, gate_line=gate, first_host_line=remote,
                            actual_main_executed=False, valid_approval_created=False))
    c.write(tmp_path / "authority-placement.json", dict(results=results))


def test_C02_fresh_bootstrap_plan_operation_and_literal_closure(monkeypatch, tmp_path):
    async def scenario():
        authority_path = tmp_path / "caller"
        authority_path.mkdir(mode=0o700)
        authority = c.consumer_module("generation01").Authority(authority_path)
        try:
            fixture = await new_fixture(monkeypatch, tmp_path / "fixture")
            case = fixture.case
            output = tmp_path / "client"
            output.mkdir()
            (tmp_path / "gateway").mkdir()
            initial = {key:deepcopy(case.startup[key]) for key in
                       ("native_owned","warmup_record","drain_records","gateway_initial_record")}
            raw_before = deepcopy(case.startup)
            events = []
            class Publisher:
                def publish(self, kind, evidence):
                    phase = evidence.get("phase","confirmed")
                    for expected, field in (("CAS_receipts","gateway_owned"),
                        ("capture_metadata_received","capture_metadata_record"),
                        ("capture_begin_received","capture_begin_record")):
                        if phase == expected:
                            assert field not in initial
                    value = authority.publish(kind, evidence)
                    events.append(dict(kind=kind, phase=phase, generation=value))
                    return value
            mapping = {"same-budget-CAS":"cas_command_record","confirm-version1":"gateway_confirm_record",
                       "capture-metadata":"capture_metadata_record","capture-begin":"capture_begin_record"}
            async def call(target, method, path, *, role, payload=None, **kw):
                record = case.startup[mapping[role]]
                if payload is not None:
                    assert payload == json.loads(record["request_body"])
                return deepcopy(record)
            wire = SimpleNamespace(api=c.preflight().frozen.control(), pre=c.preflight(),
                consumer=Publisher(), clock=time.monotonic, call=call)
            with monkeypatch.context() as patch:
                patch.setattr(case.proc_api, "LinuxProc", lambda: case.proc)
                await client.bootstrap(wire, initial, {"gateway":case.startup["gateway_owned"]},
                                       output, time.monotonic() + 20)
            assert case.startup == raw_before
            plan, offers = c.load(c.PLAN), c.load(c.OFFERS)
            assert plan["trials"][0]["request_ids"] == [r["request_id"] for r in offers["rows"]]
            assert plan["trace_sha256"] == c.sha(c.OFFERS)
            assert json.loads(initial["cas_command_record"]["request_body"])["operation_id"] == "opteam16-cut-consumer02"
            proof = c.bootstrap_inputs(initial, cutoff=time.monotonic())
            assert proof["confirmation"]["actual_applied_at"] is None
            assert authority.generation > 1
            with monkeypatch.context() as patch:
                patch.setattr(c, "PLAN", c.HERE / "inputs/reference-consumer/NATIVE_PLAN.json")
                bad = tmp_path / "old-plan-binding/client"
                bad.mkdir(parents=True)
                (bad.parent / "gateway").mkdir()
                old_initial = {key:deepcopy(case.startup[key]) for key in
                               ("native_owned","warmup_record","drain_records","gateway_initial_record")}
                wire.consumer = None
                with pytest.raises(ValueError, match="metadata_source_before_capture_begin"):
                    await client.bootstrap(wire, old_initial, {"gateway":case.startup["gateway_owned"]},
                                           bad, time.monotonic() + 20)
            c.write(tmp_path / "new-bootstrap-binding.json", dict(
                events=events, fixture_raw_unchanged=case.startup == raw_before,
                plan_sha256=c.sha(c.PLAN), offers_sha256=c.sha(c.OFFERS),
                Native_execution=False, actual_application_at=None))
        finally:
            authority.close()
    run(scenario(), tmp_path)


def test_C03_literal_source_cwd_module_clock_and_repair_preservation(monkeypatch, tmp_path):
    core = c.consumer_module("common01")
    monkeypatch.chdir(c.PREFLIGHT)
    proof = c.verify_code()
    literal = core.verify_code(literal_cwd=True)
    original = c.HERE / "inputs/reference-consumer"
    for name in ("consumer01.py","generation01.py","worker01.py","validate01.py","ipc01.py"):
        assert (c.CONSUMER / name).read_bytes() == (original / name).read_bytes()
    assert core.proc_api().CLOCK_SOURCE_SHA256 == c.sha(c.SOURCE / "prefill_lab/live_http.py")
    monkeypatch.chdir(c.HERE)
    with pytest.raises(ValueError, match="exclusive_candidate_cwd_required"):
        core.verify_code(literal_cwd=True)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "preflight01", SimpleNamespace(__file__=str(tmp_path / "foreign.py")))
        with pytest.raises(ValueError, match="literal_preflight_origin"):
            core.preflight()
    c.write(tmp_path / "guard-proof.json", dict(private_input=proof, literal_input=literal,
        repaired_runtime_bytes_unchanged=True, original_guards_bypassed=False))


def test_C04_actual_service_route_exec_limits_not_Gateway_budget(tmp_path):
    env = dict(PYTHONPATH=str(c.SOURCE13), PR33_R24_PREPARED_COPY_READY_ENABLED="0",
               PR33_R24_PREPARED_FORWARD_ENABLED="1", PR33_R24_PHASE_OBSERVER_ENABLED="1",
               PR33_R23_INPUT_POST_OBSERVER_ENABLED="1")
    source = c.SOURCE / c.SOURCE13_REL / "launch_queue_service.py"
    tree = ast.parse(source.read_bytes())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignment = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets))
    assert isinstance(assignment.value, ast.List)
    command = []
    for part in assignment.value.elts:
        if isinstance(part, ast.Constant):
            command.append(part.value)
        else:
            assert ast.unparse(part) == "config['MODEL_NAME']"
            command.append(str(c.ALIAS))
    program = str(c.PYTHON.with_name("vllm"))
    argv = [program, *command]
    service.validate_exec("native", program, argv, env)
    wrong = list(argv)
    wrong[wrong.index("--max-num-batched-tokens") + 1] = "32"
    with pytest.raises(ValueError, match="Native_formation_limit_not_Gateway_admission"):
        service.validate_exec("native", program, wrong, env)
    gateway = [str(c.PYTHON),"-m","uvicorn","gateway_prefix_bridge:application",
               "--factory","--host","127.0.0.1","--port","8080","--workers","1"]
    service.validate_exec("gateway", str(c.PYTHON), gateway, {})
    with pytest.raises(ValueError, match="source_exec_gateway_arguments"):
        service.validate_exec("gateway", str(c.PYTHON), [*gateway,"--reload"], {})
    routes = {}
    for role in ("native","gateway"):
        entry, args = service.service_route(role)
        assert args[args.index("--plan") + 1] == str(c.PLAN)
        assert args[args.index("--python") + 1] == "/workspace/pr32-r23/venv/bin/python3.12"
        assert str(c.CONFIG) in args and str(c.PUBLIC / role) in args
        routes[role] = dict(entry=str(entry), argv=args)
    c.write(tmp_path / "service-routes-not-executed.json", dict(routes=routes,
        Native_argv=argv, Gateway_budget=32, Native_tokens=2048, Native_sequences=32,
        actual_exec_called=False))


def test_C05_actual_protocol_original_activation_and_publication(monkeypatch, tmp_path):
    async def scenario():
        output = tmp_path / "client"
        output.mkdir()
        calls, bound = [], invocation()
        client.invocation_check(bound)
        original_active = time.monotonic() - .25
        @dataclass
        class NoNativeClosure:
            CPU_call_placement_only: bool = True
        class Consumer:
            publication_refusals = []
            def activate(self):
                calls.append("one_activate")
                return original_active
            def publish(self, kind, evidence):
                calls.append((kind,evidence.get("request_id")))
        wire = SimpleNamespace(consumer=Consumer(), clock=time.monotonic,
            capture=SimpleNamespace(secret=MARKER), pre=SimpleNamespace(
                frozen=c.preflight().frozen, inspect_cut=lambda *a, **kw: {}),
            api=SimpleNamespace(clock_domain=c.domain, body=lambda row: row))
        async def startup(*args):
            assert args[-1] == bound["startup_deadline"]
            calls.append("startup")
            return dict(drain_records=[dict(base=dict(run_id="CPU-only"))], capture_metadata_record={})
        async def bootstrap(*args):
            assert args[-1] == original_active + 20 and calls[:2] == ["startup","one_activate"]
            calls.append("bootstrap")
            return {"pid":111}
        async def collect(*args, deadline, **kw):
            assert deadline <= original_active + 20
            return NoNativeClosure(), {"cut":{}}
        async def managed(wire, directory, row, initial, closure, previous, *, deadline, memory, **kw):
            assert deadline == original_active + 20
            rid = row["request_id"]
            assert rid in memory and calls[-2:] == [
                ("public_history_published",rid),("offer_intent",rid)]
            calls.append("managed-" + str(row["ordinal"]))
            row.update(cut="CPU-only",fate="completed_length",inflight=dict(
                nonempty_overlap=False,consumer={"status":"refused"} if row["ordinal"] == 2 else {}))
            return closure, previous
        async def final(*args, deadline, **kw):
            assert deadline <= original_active + 23
            calls.append("final")
        async def finish(*args, deadline, **kw):
            assert deadline <= original_active + 23
            calls.append("end_ACK")
            return {"CPU_spy":True}
        for name, function in (("startup",startup),("bootstrap",bootstrap),("collect_one",collect),
            ("managed_offer",managed),("final_drain",final),("finish_capture",finish)):
            monkeypatch.setattr(client,name,function)
        monkeypatch.setattr(c,"load_origins",lambda path: [])
        monkeypatch.setattr(c,"require_origins",lambda *a, **kw: True)
        monkeypatch.setattr(c,"estimate_prompt",lambda *a: 17)
        result = await client.protocol_run(wire,output,{"gateway":{"pid":200}},
                                           bound["startup_deadline"],None)
        assert not result["failures"] and result["engineering_pass"] and not result["component_pass"]
        assert result["active_started_at"] == original_active and calls.count("one_activate") == 1
        assert [x for x in calls if isinstance(x,str)][-4:] == ["managed-1","managed-2","final","end_ACK"]
        anchor = client.shared_teardown_anchor(tmp_path,None,original_active)
        assert anchor == result["finished_at"]
        assert result["actual_application_at"] is result["planner_entry_at"] is result["decision_at"] is None
        c.write(tmp_path / "actual-protocol-placement.json", dict(calls=calls, anchor=anchor,
            original_host_started_at=bound["host_started_at"], Native_execution=False))
    run(scenario(),tmp_path)


def test_C06_real_ordinal2_path_late_consumer_preserves_producer(monkeypatch,tmp_path):
    async def scenario():
        active = time.monotonic() + 20
        gates = {1:asyncio.Event(),2:asyncio.Event()}
        events = []
        class Boundary:
            def publish(self,kind,evidence):
                events.append(("publication",kind))
            def invalidate_publication(self,*args):
                raise AssertionError("unexpected_publication_error")
            def begin(self,*,ordinal,pending_at_trigger):
                assert ordinal == 2 and pending_at_trigger
                self.lease = dict(begin=time.monotonic())
                self.lease["deadline"] = min(active,self.lease["begin"] + .080)
                events.append(("begin",ordinal))
                return self.lease
            async def handoff(self,bundle):
                events.append(("handoff",bundle["name"]))
                assert time.monotonic() >= self.lease["deadline"]
                raise ValueError("CPU_whole80ms_expired_no_rebase")
        boundary = Boundary()
        async def handler(request):
            identifier = request.headers["x-benchmark-request-id"]
            ordinal = next(row["ordinal"] for row in c.load(c.OFFERS)["rows"] if row["request_id"] == identifier)
            await gates[ordinal].wait()
            return httpx.Response(200,headers={"x-request-id":identifier},json=dict(
                id="chatcmpl-" + identifier,model=c.MODEL_ID,choices=[dict(finish_reason="length")],
                usage=dict(completion_tokens=16)))
        def endpoint():
            return httpx.AsyncClient(base_url="http://127.0.0.1",transport=httpx.MockTransport(handler))
        wire = client.Wire(tmp_path,MARKER,c.domain(),endpoint(),endpoint(),consumer=boundary)
        monkeypatch.setattr(wire.pre,"inspect_cut",lambda *a,**kw: {})
        async def collect(wire,initial,directory,name,*,deadline,**kw):
            entered = time.monotonic()
            events.append(("collect",name,deadline,entered))
            assert deadline <= active and deadline > entered
            if name == "offer-2-inflight-cut":
                assert deadline > boundary.lease["deadline"]
                await asyncio.sleep(.09)
                assert time.monotonic() < deadline
                gates[2].set()
            elif name == "offer-1-inflight-cut":
                gates[1].set()
            c.write(directory / (name + ".json"),dict(CPU_call_placement_only=True))
            return None,{"cut":{}}
        monkeypatch.setattr(client,"collect_one",collect)
        monkeypatch.setattr(client,"consumer_bundle",lambda wire,initial,closure,memory,name: {"name":name})
        monkeypatch.setattr(client,"assess_inflight",lambda *a,**kw: dict(
            status="CPU_call_placement_not_physical_overlap",nonempty_overlap=False))
        rows = []
        try:
            for planned in c.load(c.OFFERS)["rows"]:
                row = dict(planned,record=None,fate="not_sent")
                rows.append(row)
                await client.managed_offer(wire,tmp_path,row,{},None,{"cut":{}},
                    deadline=active,memory={},process={},telemetry=tmp_path / "unused",
                    payload=deepcopy(c.load(c.OFFERS)["payload"]))
            assert [e for e in events if e[0] == "begin"] == [("begin",2)]
            assert [e for e in events if e[0] == "handoff"] == [("handoff","offer-2-inflight-cut")]
            assert [e[1] for e in events if e[0] == "collect"] == [
                "offer-1-inflight-cut","offer-1-cut","offer-2-inflight-cut","offer-2-cut"]
            assert all(row["fate"] == "completed_length" for row in rows)
            assert rows[1]["inflight"]["consumer"]["status"] == "refused"
            assert len(wire.capture.records) == 2
            c.write(tmp_path / "late-consumer-producer-preserved.json",dict(
                events=events,rows=rows,Native_execution=False,not_live_fairness_proof=True))
        finally:
            await wire.close()
    run(scenario(),tmp_path)


def test_C07_new_wire_exact_source_bytes_and_genuine_mock_overlap(monkeypatch,tmp_path):
    async def scenario():
        fixture = await new_fixture(monkeypatch,tmp_path / "fixture")
        case = fixture.case
        output = tmp_path / "client"
        output.mkdir()
        gate, entered = asyncio.Event(),asyncio.Event()
        asgi = httpx.ASGITransport(app=case.app)
        class Gateway(httpx.AsyncBaseTransport):
            async def handle_async_request(self,request):
                if request.url.path == "/v1/chat/completions":
                    entered.set()
                    await gate.wait()
                    rid = request.headers["x-benchmark-request-id"]
                    return httpx.Response(200,headers={"x-request-id":rid},json=dict(
                        id="chatcmpl-" + rid,model=c.MODEL_ID,choices=[dict(finish_reason="length")],
                        usage=dict(completion_tokens=16)))
                return await asgi.handle_async_request(request)
            async def aclose(self):
                await asgi.aclose()
        async def native(request):
            frame = case.phase.phase_snapshot(case.core)
            frame["base"]["frontend_sampled_at"] = case.clock()
            return httpx.Response(200,content=json.dumps(frame,indent=3).encode(),
                                  headers={"content-type":"application/json"})
        wire = client.Wire(output,MARKER,c.domain(),
            httpx.AsyncClient(base_url="http://127.0.0.1:8000",transport=httpx.MockTransport(native)),
            httpx.AsyncClient(base_url="http://127.0.0.1:8080",transport=Gateway()))
        task = None
        try:
            closure,_ = await client.collect_one(wire,case.startup,output,"initial-cut",
                deadline=time.monotonic() + .5,memory=case.public_memory,process=case.native_process,
                telemetry=fixture.directory / "gateway.jsonl")
            _,ghost = await case.issue_then_release_with_physical_stock()
            planned = c.load(c.OFFERS)["rows"][1]
            task = asyncio.create_task(wire.call("gateway","POST","/v1/chat/completions",
                role="managed-2",payload=deepcopy(c.load(c.OFFERS)["payload"]),
                identifier=planned["request_id"],deadline=time.monotonic() + 20,limit=20))
            await asyncio.wait_for(entered.wait(),.2)
            closure,current = await client.collect_one(wire,case.startup,output,"offer-2-inflight-cut",
                deadline=time.monotonic() + .5,memory=case.public_memory,process=case.native_process,
                telemetry=fixture.directory / "gateway.jsonl",closure=closure)
            returned = time.monotonic()
            assert not task.done()
            bundle = client.consumer_bundle(wire,case.startup,closure,case.public_memory,"offer-2-inflight-cut")
            for name in ("initial-cut","offer-2-inflight-cut"):
                binding = c.load(output / (name + "-raw-binding.json"))
                raw = binding["raw_bundle"]
                for kind,key in (("snapshot","gateway_record"),("barrier","barrier_record")):
                    assert raw[kind + "_base64"] == wire.capture.records[binding[key]]["response_body_base64"]
                native_record = wire.capture.records[binding["native_record"]]
                body = base64.b64decode(native_record["response_body_base64"])
                assert body != wire.api.canonical(json.loads(body))
                assert raw["journal"]["native_record"]["response_body_base64"] == native_record["response_body_base64"]
                assert base64.b64decode(raw["prefix_base64"]) == (output / raw["journal"]["prefix"]["path"]).read_bytes()
            view = wire.pre.inspect_cut(current["cut"],consume_at=returned,deadline=returned + .5)
            assert view["full_stock"]["queued_ghost_request_ids"] == [ghost]
            assert any(row["work"] == [] for row in view["full_stock"]["queue_oldest_first"])
            gate.set()
            terminal = await task
            assert terminal["client_send_at"] < current["cut"]["cut_at"] <= returned < terminal["client_receive_at"]
            assert len({r["sequence"] for r in wire.capture.records}) == len(wire.capture.records) == 7
            c.write(tmp_path / "exact-new-bundle.json",dict(bundle=bundle,
                actual_mock_overlap=True,source_collector_return_observed_at=returned,
                terminal_at=terminal["client_receive_at"],Native_execution=False,
                actual_application_at=None,planner_entry_at=None))
        finally:
            gate.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
            await wire.close()
    run(scenario(),tmp_path)


def test_C08_actual_owned_registration_shared_anchor_true_close(monkeypatch,tmp_path):
    output,state = tmp_path / "public",tmp_path / "state"
    output.mkdir(mode=0o700)
    state.mkdir(mode=0o700)
    monkeypatch.setattr(c,"STATE",state)
    bounds = invocation()
    bounds.update(owner_pid=os.getpid(),owner_start_ticks=c.owner().snapshot(os.getpid())[0])
    path = state / "CPU-invocation.json"
    c.write(path,bounds)
    arms = []
    monkeypatch.setattr(launch,"arm",lambda deadline: arms.append(deadline))
    owned = Owned(MARKER)
    root = owned.spawn([sys.executable,"-I","-B",str(c.HERE / "cpu_helper02.py"),
        "--bounded-owned-CPU-only","--output",str(output),"--state",str(state),"--invocation",str(path)],
        directory=tmp_path / "producer-root",cwd=c.HERE,env=dict(os.environ))
    supervision = None
    try:
        supervision = launch.monitor(owned,root,bounds["host_started_at"],[],directory=output)
        registration = c.load(output / "consumer-registration02.json")
        ack = c.load(output / "consumer-registered02.json")
        assert ack["pid"] == registration["identity"]["pid"] and ack["client_pid"] == root["process"].pid
        foreign = deepcopy(registration)
        foreign["identity"]["pid"] = os.getpid()
        with pytest.raises(ValueError,match="actual_owned_consumer_leaf_identity"):
            owned.register_leaf(foreign,root)
        assert os.getpid() not in owned.identities
    finally:
        anchor = supervision["anchor"] if supervision else time.monotonic()
        c.publish(output / "host-cleanup-anchor02.json",dict(anchor=anchor,
            original_host_cleanup_cutoff=bounds["cleanup_hard_deadline"],clock_domain=c.domain()))
        cleanup = owned.cleanup(anchor,client=root,cancelled=not bool(supervision and supervision["normal"]),
                                hard_deadline=bounds["cleanup_hard_deadline"])
        c.write(tmp_path / "owned-cleanup.json",cleanup)
    assert supervision["normal"] and cleanup["all_joined"] and cleanup["capture_complete"]
    assert root["process"].returncode == 0 and len(cleanup["owned_pid_start_ticks"]) == 2
    original_IO = min(bounds["outer_deadline"],cleanup["finished_at"] + 17)
    persisted = launch.persist_close(output,anchor=anchor,hard_deadline=bounds["cleanup_hard_deadline"],
                                    io_deadline=original_IO,joined=cleanup["all_joined"])
    before = (state / "consumer-close-returned02.json").read_bytes()
    assert before == (output / "consumer-actual-returned-close02.json").read_bytes()
    assert persisted["persisted_sha256"] == c.sha_bytes(before)
    assert not persisted["receipt"]["component_pass"] and persisted["receipt"]["all_joined"]
    assert all(deadline <= bounds["host_started_at"] + 203 for deadline in arms)
    for pid,ticks in cleanup["owned_pid_start_ticks"].items():
        current = c.owner().snapshot(int(pid))
        assert current is None or current[0] != ticks
    with pytest.raises(ValueError,match="owned_join_before_actual_close_IO"):
        launch.persist_close(tmp_path,anchor=anchor,hard_deadline=bounds["cleanup_hard_deadline"],
                             io_deadline=time.monotonic() - .01,joined=True)
    c.write(tmp_path / "host-actual-close-proof.json",dict(
        staged_sha256=c.sha_bytes(before),persisted=persisted,original_IO_deadline=original_IO,
        Native_execution=False,actual_application_at=None))


def test_C09_private_config_requires_join_and_decoded_safe_export(monkeypatch,tmp_path):
    config = tmp_path / "CPU.private.json"
    c.write(config,dict(MODEL_NAME=str(c.ALIAS),ADMIN_API_KEY=MARKER))
    config.chmod(0o600)
    monkeypatch.setattr(c,"CONFIG",config)
    unjoined,bad,safe = [tmp_path / name for name in ("unjoined","unsafe","safe")]
    for directory in (unjoined,bad,safe):
        directory.mkdir(mode=0o700)
    deadline = time.monotonic() + 17
    with pytest.raises(ValueError,match="join_before_private_config_close"):
        launch.close_private_config(unjoined,joined=False,secret=MARKER,deadline=deadline)
    assert config.exists()
    c.write(bad / "encoded-CPU-marker.json",dict(response_body_base64=base64.b64encode(MARKER.encode()).decode()))
    with pytest.raises(ValueError,match="safe_export_before_private_config_close"):
        launch.close_private_config(bad,joined=True,secret=MARKER,deadline=deadline)
    assert config.exists() and not list(bad.glob("*config-close*"))
    c.write(safe / "retained-close-shape.json",dict(actual_application_at=None,CPU_only=True))
    assert launch.close_private_config(safe,joined=True,secret=MARKER,deadline=deadline)
    before = c.load(safe / "safe-export-before-config-close02.json")
    after = c.load(safe / "private-config-close02.json")
    assert not config.exists() and before["checked_at"] <= after["finished_at"] < deadline
    assert before["safe_to_export"] and after["safe_export_before_removal"] and after["all_helpers_joined"]
    c.write(tmp_path / "private-close-guards.json",dict(
        no_real_credential_created=True,unjoined_refused=True,decoded_unsafe_refused=True,closed=after))
