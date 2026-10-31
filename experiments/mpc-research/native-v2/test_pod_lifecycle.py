from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pod_lifecycle import ExistingPod, POD, load_credential


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.provider = {
            "id": POD, "desiredStatus": "EXITED", "costPerHr": .89,
            "runtime": None, "volumeInGb": 0, "containerDiskInGb": 50}

    def client(self, transport=None):
        def fake(query):
            self.calls.append(query)
            if "podResume" in query:
                self.provider.update(desiredStatus="RUNNING",
                                     runtime={"gpus": [{"id": "test-gpu"}], "ports": []})
                return {"podResume": {"id": POD, "desiredStatus": "RUNNING"}}
            if "podStop" in query:
                self.provider.update(desiredStatus="EXITED", runtime=None)
                return {"podStop": {"id": POD, "desiredStatus": "EXITED"}}
            return {"pod": dict(self.provider)}
        return ExistingPod("do-not-log-this", self.root / "ledger.jsonl",
                           transport or fake)

    def plan(self):
        return dict(pod_id=POD, source_commit="a" * 40, experiment_id="nv2-test",
                    hypothesis="batch composition changes service",
                    workload={"seed": 170001}, completion_rule="one bounded arm",
                    artifact_directory="native-v2-test")

    def backup(self):
        artifact = self.root / "sealed.raw"
        artifact.write_bytes(b"actual saved raw evidence")
        return dict(pod_id=POD, captured_utc=datetime.now(timezone.utc).isoformat(),
                    sealed_remote_evidence_matches_local=True,
                    artifacts=[dict(path=str(artifact),
                        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())])

    def test_resume_only_single_existing_pod_and_receipt_never_logs_key(self):
        result = self.client().change("resume", self.plan())
        mutations = [q for q in self.calls if q.startswith("mutation")]
        self.assertEqual(len(mutations), 1)
        self.assertIn(f'podId: "{POD}", gpuCount: 1', mutations[0])
        self.assertEqual(result["provider"]["desiredStatus"], "RUNNING")
        ledger = (self.root / "ledger.jsonl").read_text()
        self.assertNotIn("do-not-log-this", ledger)
        self.assertIn("resume_intent", ledger)

    def test_refuses_unqualified_plan_before_mutation(self):
        plan = self.plan()
        plan["source_commit"] = "uncommitted"
        with self.assertRaises(ValueError):
            self.client().change("resume", plan)
        self.assertFalse(any(q.startswith("mutation") for q in self.calls))

    def test_stop_requires_unchanged_sealed_remote_evidence(self):
        self.provider.update(desiredStatus="RUNNING", runtime={"gpus": [{}]})
        receipt = self.backup()
        Path(receipt["artifacts"][0]["path"]).write_text("changed after sealing")
        with self.assertRaises(ValueError):
            self.client().change("stop", receipt)
        self.assertFalse(any(q.startswith("mutation") for q in self.calls))

    def test_stop_verifies_both_exit_and_released_runtime(self):
        self.provider.update(desiredStatus="RUNNING", runtime={"gpus": [{}]})
        result = self.client().change("stop", self.backup())
        self.assertTrue(result["gpu_release_verified"])

    def test_exit_without_runtime_field_is_not_release_proof(self):
        self.provider.pop("runtime")
        self.assertFalse(self.client().status()["gpu_release_verified"])

    def test_no_blind_retry_after_ambiguous_mutation_failure(self):
        def ambiguous(query):
            self.calls.append(query)
            if query.startswith("mutation"):
                raise RuntimeError("ambiguous response")
            return {"pod": dict(self.provider)}
        with self.assertRaises(RuntimeError):
            self.client(ambiguous).change("resume", self.plan())
        self.assertEqual(sum(q.startswith("mutation") for q in self.calls), 1)
        self.assertIn("resume_unverified", (self.root / "ledger.jsonl").read_text())

    def test_mutation_ack_alone_cannot_claim_gpu_release(self):
        self.provider.update(desiredStatus="RUNNING", runtime={"gpus": [{}]})
        def pending(query):
            self.calls.append(query)
            if query.startswith("mutation"):
                return {"podStop": {"id": POD, "desiredStatus": "EXITED"}}
            return {"pod": dict(self.provider)}
        with self.assertRaises(RuntimeError):
            self.client(pending).change("stop", self.backup(), timeout_s=0)

    def test_credential_permissions_and_symlinks_are_rejected(self):
        credential = self.root / "key.json"
        credential.write_text(json.dumps({"pod_id": POD, "api_key": "test-key"}))
        credential.chmod(0o644)
        with self.assertRaises(ValueError):
            load_credential(credential)
        credential.chmod(0o600)
        self.assertEqual(load_credential(credential), "test-key")
        link = self.root / "link"
        link.symlink_to(credential)
        with self.assertRaises(OSError):
            load_credential(link)

    def test_provider_errors_do_not_echo_secret_bearing_bodies(self):
        control = ExistingPod("do-not-log-this", self.root / "ledger.jsonl")
        with patch("pod_lifecycle.urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = ValueError("do-not-log-this")
            with self.assertRaises(RuntimeError) as error:
                control.status()
        self.assertNotIn("do-not-log-this", str(error.exception))


if __name__ == "__main__":
    unittest.main()
