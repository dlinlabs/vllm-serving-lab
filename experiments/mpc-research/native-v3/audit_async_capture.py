"""Account for observed one-token async terminal overhang without changing v2."""
from prefill_lab.contracts import audit_capture, finite


VERSION = "native-v3-async-terminal-accounting-v1"


def audit_async_capture(capture, worker_rows, *, require_steps=True):
    try:
        strict = audit_capture(capture, worker_rows, require_steps=require_steps,
                               forced_outputs=False)
        problems = set(strict["problems"]) - {"Final input/output progress mismatch"}
        requests = capture["requests"]
        workers = {row["step_id"]: row for row in worker_rows}
        progress, work = {}, {rid: [] for rid in requests}
        for step in capture["steps"]:
            for row in step["work"]:
                rid = row["request_id"]
                if (rid not in requests or type(row["position"]) is not int
                        or row["position"] < 0 or type(row["tokens"]) is not int
                        or row["tokens"] <= 0):
                    problems.add("Invalid or unknown executed request work")
                    continue
                progress[rid] = row["position"] + row["tokens"]
                work[rid].append((step, row))
        accounting = {}
        for rid, request in requests.items():
            prompt, output, limit = (request[k] for k in
                                     ("input_tokens", "output_tokens", "max_tokens"))
            terminal = request["finished"]
            if (any(type(n) is not int for n in (prompt, output, limit))
                    or prompt <= 0 or not 0 < output <= limit
                    or not finite(terminal) or terminal < 0
                    or request["finish_reason"] not in ("stop", "length")):
                problems.add("Invalid delivered output or terminal")
                continue
            if not require_steps:
                continue
            minimum = prompt + output - 1
            actual = progress.get(rid, -1)
            extra = actual - minimum
            proof = []
            if extra not in (0, 1):
                problems.add("Unqualified final executed-work difference")
            elif extra == 1:
                tails = [(s, r) for s, r in work[rid] if r["position"] >= minimum]
                if len(tails) != 1:
                    problems.add("Missing unique terminal-overhang suffix")
                else:
                    step, row = tails[0]
                    before = step["state_before"]["requests"][rid]
                    worker = workers[step["step_id"]]
                    if (row["position"] != minimum or row["tokens"] != 1
                            or any(type(before[k]) is not int for k in
                                   ("computed", "num_tokens", "placeholders"))
                            or before["computed"] != minimum
                            or before["num_tokens"] != minimum
                            or before["placeholders"] != 1
                            or not finite(worker.get("host_start"))
                            or not 0 <= step["start"] <= terminal <= step["end"]
                            or not 0 <= worker["host_start"] <= terminal):
                        problems.add("Unproved pre-terminal placeholder forward work")
                    else:
                        proof.append(step["step_id"])
            accounting[rid] = dict(delivered_output_tokens=output,
                delivered_token_forward_minimum=minimum,
                observed_forward_tokens=actual, terminal_overhang_tokens=extra,
                placeholder_worker_step_ids=proof)
        return dict(schema=VERSION, valid=not problems, problems=sorted(problems),
            strict_v2_audit=strict, request_work=accounting,
            scope="Observed contiguous physical forward work and delivered output are separate."
                  " A one-token suffix requires pre-terminal placeholder/worker evidence."
                  " CUDA elapsed spans are not absolute GPU clocks or exclusive service.",
            complete_experiment_passed=False)
    except (KeyError, TypeError, ValueError, AttributeError):
        return dict(schema=VERSION, valid=False, problems=["Malformed native capture"],
                    complete_experiment_passed=False)
