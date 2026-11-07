"""Verify and pin all preserved source/seal inputs before the authorized correction."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SCOPE = Path(".runtime/r24-resume-p0-nominal-episode04")
OLD = Path(".runtime/r24-resume-p0-nominal-episode03")
EARLIER = Path(".runtime/r24-resume-p0-nominal-episode02")
API = Path(".runtime/r24-resume-p0-nominal-client-api03")
PUBLIC = Path("experiments/mpc-research/resume-mvp")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (SCOPE / name).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify_manifest(path, expected):
    assert sha(path / "SHA256SUMS") == expected
    names = []
    for line in (path / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        assert sha(path / name) == checksum, name
        names.append(name)
    return len(names)


assert Path.cwd() == Path("/home/daniel_lin/gt/pr32/mayor/mpc-local-r21")
old_count = verify_manifest(OLD, "07da8f077fc42b636c2bc04fa813a7ffa9d5be4de24deab8b43621062b6bced2")
earlier_count = verify_manifest(EARLIER, "0ecc3f73638cc0bcd312962dd077bc987553ee969d40052a5be9ef4df7da6f2e")
api_count = verify_manifest(API, "b3e6174797f8d466181ebf42e87b5c45133341ff7351265d166d0cfc1dfc8a1b")
pins = json.loads((OLD / "INPUT_PINS.json").read_text())
assert all(sha(Path(name)) == row["sha256"] for name, row in pins.items())
for name in ("SHA256SUMS", "SEAL_RECEIPT.json", "FREEZE.json", "REPORT.json", "FAILURES.json"):
    path = OLD / name
    pins[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
save("BASELINE_INPUT_PINS.json", pins)
before = SCOPE / "sources/before"
before.mkdir()
for name in ("nominal_episode.py", "test_nominal_episode.py"):
    shutil.copyfile(PUBLIC / name, before / name)
peer_paths = [PUBLIC / name for name in (
    "nominal_client.py", "test_nominal_client.py", "nominal_metrics.py", "nominal_services.py",
)]
peer_paths += [Path("trace_workload.py"), Path("experiments/mpc-research/native-r23/score_http.py")]
peer_rows = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
              "scope": "retained dependency; no new content review"} for path in peer_paths]
assert len(peer_rows) == 6 and sum(row["bytes"] for row in peer_rows) == 200013
save("SOURCE_READ_BUDGET.json", {
    "peer_inputs": peer_rows, "count": len(peer_rows), "total_bytes": 200013,
    "limits": {"count": 6, "bytes": 384 * 1024}, "new_peer_content_reads": 0,
    "note": "Hash verification of sealed proof packets is not a new implementation input. Own/accepted runtime context is retained.",
})
save("STARTED.json", {
    "at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "interpreter": sys.executable,
    "identity": {"pid": os.getpid(), "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]},
    "plan_sha256": sha(SCOPE / "PLAN.json"), "baseline_pin_count": len(pins),
    "preserved_episode03_members": old_count, "preserved_episode02_members": earlier_count,
    "client_API03_members": api_count, "source_generations_executed": 0, "test_attempts_executed": 0,
})
print(json.dumps({"episode03_members_verified": old_count, "episode02_members_verified": earlier_count,
                  "API03_members_verified": api_count, "baseline_bindings": len(pins),
                  "test_attempts": 0}, sort_keys=True))
