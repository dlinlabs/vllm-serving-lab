import hashlib
import json
import math
import sys
from pathlib import Path


def pairs(items):
    seen = {}
    for key, value in items:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def reject_constant(name):
    raise ValueError(f"non-finite constant {name}")


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite float {text}")
    return value


REQUIRED = (
    "schema", "written_utc", "status", "root", "capsule", "author", "mandate",
    "supplied_facts_mayor_not_independently_verified", "verdict", "pins", "known_concern_resolution",
    "blockers", "findings_low", "verified_wiring_facts", "unverified_contracts",
    "prior_context_applicable", "witnesses", "read_disclosure", "nonclaims", "limitations", "artifacts",
)
NOT_YET_WRITTEN = {"report-check01-py314.log", "report-check02-py312.log", "SHA256SUMS"}


def main():
    raw = Path(sys.argv[1]).read_bytes()
    problems = []
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as error:
        problems.append(f"non-ascii: {error}")
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject_constant, parse_float=finite_float)
    problems += [f"missing {key}" for key in REQUIRED if key not in data]
    author = data.get("author", {})
    if (author.get("model_id"), author.get("reasoning_effort"), author.get("child_agents")) != ("claude-opus-5.5", "max", 0):
        problems.append("author identity")
    if not str(data.get("verdict", {}).get("overall", "")).startswith("FAIL_PRE_RUN"):
        problems.append("verdict overall")
    blockers = data.get("blockers", [])
    if [b.get("id") for b in blockers] != ["B1", "B2", "B3"]:
        problems.append("blocker ids")
    for blocker in blockers:
        problems += [f"{blocker.get('id')} missing {key}" for key in
                     ("severity", "confidence", "evidence", "effect", "smallest_fix") if not blocker.get(key)]
    if data.get("witnesses", {}).get("count") != 0:
        problems.append("witness count")
    pins = {}
    for line in Path("SOURCE-PINS.sha256").read_text().splitlines():
        digest, path = line.split(None, 1)
        pins[path.lstrip("*")] = digest
    if {e["path"]: e["sha256"] for e in data.get("pins", {}).get("inputs", [])} != pins:
        problems.append("report pins differ from SOURCE-PINS.sha256")
    for path, digest in pins.items():
        if hashlib.sha256((Path("inputs") / Path(path).name).read_bytes()).hexdigest() != digest:
            problems.append(f"snapshot drift {path}")
    for name in data.get("artifacts", []):
        if name not in NOT_YET_WRITTEN and not Path(name.rstrip("/")).exists():
            problems.append(f"artifact missing {name}")
    print(json.dumps({"ok": not problems, "problems": problems, "python": sys.version.split()[0],
                      "report_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
