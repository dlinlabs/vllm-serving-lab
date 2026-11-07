"""Bounded owned CPU-only registration/close control; never a Native entrypoint."""

import argparse
import asyncio
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition_common01 as c
import client01 as client

MARKER = "CPU-NOT-CREDENTIAL-" + "0" * 45


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--bounded-owned-CPU-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--invocation", type=Path, required=True)
    args = parser.parse_args()
    c.STATE = args.state
    c.verify_code()
    invocation = c.load(args.invocation)
    client.invocation_check(invocation)
    c.owned_parent(invocation["owner_pid"], invocation["owner_start_ticks"])
    output = args.output / "client"
    output.mkdir(mode=0o700)
    boundary = client.make_consumer(output, invocation, MARKER)
    boundary.mode = "cpu-source-fixture"
    def alarm(signum, frame):
        raise TimeoutError("bounded_CPU_helper_20s")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 20)
    async def scenario():
        try:
            ready = await boundary.start()
            await client.register_consumer(boundary, ready, args.output, invocation["startup_deadline"])
            active = boundary.activate()
            c.publish(args.output / "active-start.json", dict(
                started_at=active, deadline=active + 20, clock_domain=c.domain()))
            c.publish(args.output / "protocol-finished.json", dict(
                finished_at=time.monotonic(), active_started_at=active, component_pass=False,
                clock_domain=c.domain(), reason="CPU_ownership_only_no_Native_no_cut"))
        finally:
            anchor = client.shared_teardown_anchor(args.output, boundary, boundary.active_at)
            closed = await boundary.close(cancelled=True, teardown_anchor=anchor,
                                          hard_deadline=invocation["cleanup_hard_deadline"])
            observed = time.monotonic()
            path = args.state / "consumer-close-returned02.json"
            c.write(path, dict(schema="team16-actual-consumer02-close-return-v1", receipt=closed,
                actual_caller_observed_return_at=observed, shared_teardown_anchor=anchor,
                original_host_cleanup_cutoff=invocation["cleanup_hard_deadline"],
                clock_domain=c.domain(), CPU_only=True, actual_application_at=None))
            path.chmod(0o600)
        return closed
    result, joined = client.bounded_loop(scenario(), output, hard_deadline=invocation["cleanup_hard_deadline"])
    signal.setitimer(signal.ITIMER_REAL, 0)
    c.require(joined and result and result["all_joined"] and not result["component_pass"],
              "CPU_ownership_control_cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
