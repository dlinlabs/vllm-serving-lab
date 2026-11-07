"""Original source fixture on real host clocks; source PIDs remain explicitly fake."""

import base64
from copy import deepcopy
from dataclasses import asdict, replace
import os
from pathlib import Path
import time

import common01 as c

c.preflight()
from synthetic01 import Fixture
import synthetic01 as synthetic


class RealClock:
    def __call__(self):
        return time.monotonic()


class SourceFixture:
    def __init__(self, monkeypatch, directory):
        self.monkeypatch = monkeypatch
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700)
        identity = c.current_identity()
        monkeypatch.setattr(synthetic, "BOOT", identity["boot_id"])
        monkeypatch.setattr(synthetic, "DOMAIN", identity["clock_domain"])
        monkeypatch.setattr(synthetic, "Clock", RealClock)
        self.case = Fixture(monkeypatch, self.directory)
        case = self.case
        for key, process in case.proc.rows.items():
            case.proc.rows[key] = replace(
                process, time_namespace=identity["process"]["time_namespace"],
                pid_namespace=identity["process"]["pid_namespace"])
        case.native_process["pid_namespace"] = case.gateway_process["pid_namespace"] = \
            identity["process"]["pid_namespace"]
        case.plan = c.load(c.HERE / "NATIVE_PLAN.json")
        original_options = case.options
        self.latest = {}
        def options():
            settings = original_options()
            sync, asynchronous = settings["gateway_sync"], settings["gateway_async"]
            original_get = sync.get
            def get(path):
                reply = original_get(path)
                self.latest["snapshot"] = reply.content
                return reply
            sync.get = get
            original_post = asynchronous.post
            async def post(*args, **kwargs):
                reply = await original_post(*args, **kwargs)
                self.latest["barrier"] = reply.content
                return reply
            asynchronous.post = post
            return settings
        case.options = options

    def bundle(self, journal):
        size = journal["cut"]["owner_prefix"]["prefix_bytes"]
        prefix = (self.directory / "gateway.jsonl").read_bytes()[:size]
        result = dict(journal=deepcopy(journal))
        for key, raw in dict(self.latest, prefix=prefix).items():
            result[key + "_base64"] = base64.b64encode(raw).decode()
        return result

    async def initialize(self, *, ghost=False):
        case = self.case
        await case.initialize()
        pre = c.preflight()
        original_document = pre.document
        with self.monkeypatch.context() as patch:
            patch.setattr(pre, "document", lambda name: c.load(c.HERE / "NATIVE_PLAN.json")
                          if name == "PROPOSED_NATIVE_PLAN.json" else original_document(name))
            self.closure, journal = await case.initial()
        self.initial = self.bundle(journal)
        if ghost:
            await case.issue_then_release_with_physical_stock()
        return self

    async def collect(self):
        cut = await self.case.later(self.closure)
        journal = dict(cut=cut, native_record=deepcopy(self.case.records[-1]))
        current = self.bundle(journal)
        return dict(startup=deepcopy(self.case.startup), initial=deepcopy(self.initial),
                    current=current, closure=asdict(self.closure),
                    public_memory={rid:asdict(row) for rid,row in self.case.public_memory.items()})
