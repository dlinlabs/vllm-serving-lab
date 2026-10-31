"""Mayor-only lifecycle control for the one already-authorized research Pod."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import urllib.error
import urllib.request

POD = "i243ycb94oaxy2"
ENDPOINT = "https://api.runpod.io/graphql"
DEFAULT_CREDENTIAL = Path.home() / ".config/pr33-runpod-control" / f"{POD}.json"
FIELDS = """id desiredStatus costPerHr volumeInGb containerDiskInGb
runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type }
gpus { id gpuUtilPercent memoryUtilPercent } }"""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_credential(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("Lifecycle credential must be an owned mode0600 regular file")
        credential = json.load(stream)
    if credential.get("pod_id") != POD or not credential.get("api_key"):
        raise ValueError("Credential does not identify the authorized existing Pod")
    return credential["api_key"]


def require_plan(plan):
    required = ("experiment_id", "hypothesis", "workload", "completion_rule",
                "artifact_directory")
    if (not all(plan.get(k) for k in required)
            or not re.fullmatch(r"[0-9a-f]{40}", plan.get("source_commit", ""))
            or plan.get("pod_id") != POD):
        raise ValueError("Resume requires an executable, versioned experiment plan")


def require_backup(receipt):
    captured = datetime.fromisoformat(receipt["captured_utc"])
    age = (datetime.now(timezone.utc) - captured).total_seconds()
    if (receipt.get("pod_id") != POD
            or receipt.get("sealed_remote_evidence_matches_local") is not True
            or not 0 <= age <= 600 or not receipt.get("artifacts")):
        raise ValueError("Stop requires a fresh, complete remote-to-local sealing receipt")
    for artifact in receipt["artifacts"]:
        actual = hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise ValueError("Sealed local artifact changed")


class ExistingPod:
    def __init__(self, key, ledger, transport=None):
        self._key = key
        self.ledger = Path(ledger)
        self._transport = transport or self._post

    def _post(self, query):
        request = urllib.request.Request(
            ENDPOINT, data=json.dumps({"query": query}).encode(),
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as reply:
                body = json.load(reply)
        except (urllib.error.URLError, ValueError):
            raise RuntimeError("Provider request failed; do not retry mutations blindly") from None
        if (not isinstance(body, dict) or body.get("errors")
                or not isinstance(body.get("data"), dict)):
            raise RuntimeError("Provider rejected request; billing state is unverified")
        return body["data"]

    def _record(self, action, result):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        entry = {"observed_utc": datetime.now(timezone.utc).isoformat(),
                 "pod_id": POD, "action": action, **result}
        with self.ledger.open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return entry

    def status(self):
        pod = self._transport(
            f'query {{ pod(input: {{podId: "{POD}"}}) {{ {FIELDS} }} }}').get("pod")
        if not isinstance(pod, dict) or pod.get("id") != POD:
            raise RuntimeError("Provider returned no matching authorized Pod")
        safe = {k: pod.get(k) for k in (
            "id", "desiredStatus", "costPerHr", "volumeInGb",
            "containerDiskInGb", "runtime")}
        return self._record("status", {"provider": safe,
            "gpu_release_verified": pod.get("desiredStatus") == "EXITED"
                and "runtime" in pod and pod["runtime"] is None,
            "price_scope": "Hourly quote, not an invoice or storage-billing guarantee"})

    def change(self, action, evidence, *, timeout_s=120):
        if action not in ("stop", "resume"):
            raise ValueError("Only status,stop and resume of the existing single Pod are supported")
        before = self.status()
        provider = before["provider"]
        if action == "stop" and before["gpu_release_verified"]:
            return before
        if action == "resume":
            require_plan(evidence)
            if not before["gpu_release_verified"]:
                raise ValueError("Resume requires a positively confirmed stopped Pod")
        else:
            require_backup(evidence)
        self._record(action + "_intent", {
            "evidence_sha256": hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
            "quoted_usd_per_hour": provider["costPerHr"]})
        operation = "podResume" if action == "resume" else "podStop"
        count = ", gpuCount: 1" if action == "resume" else ""
        try:
            reply = self._transport(
                f'mutation {{ {operation}(input: {{podId: "{POD}"{count}}}) '
                "{ id desiredStatus } }")
            if reply.get(operation, {}).get("id") != POD:
                raise RuntimeError("Mutation acknowledgement did not identify the authorized Pod")
        except Exception:
            self._record(action + "_unverified", {"do_not_retry_mutation": True})
            raise
        deadline = time.monotonic() + timeout_s
        while True:
            observed = self.status()
            runtime = observed["provider"]["runtime"]
            if action == "stop" and observed["gpu_release_verified"]:
                return observed
            if (action == "resume"
                    and observed["provider"]["desiredStatus"] == "RUNNING"
                    and isinstance(runtime, dict) and len(runtime.get("gpus", [])) == 1):
                return observed
            if time.monotonic() >= deadline:
                self._record(action + "_unverified", {"do_not_retry_mutation": True})
                raise RuntimeError("Provider transition not verified; stop related experiment work")
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop", "resume"))
    parser.add_argument("--credential", type=Path, default=DEFAULT_CREDENTIAL)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    control = ExistingPod(load_credential(args.credential), args.ledger)
    if args.action == "status":
        result = control.status()
    else:
        if args.evidence is None:
            parser.error("--evidence is required for stop/resume")
        result = control.change(args.action, json.loads(args.evidence.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
