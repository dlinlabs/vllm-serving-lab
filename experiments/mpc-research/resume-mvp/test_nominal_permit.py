from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mpc_lab.actuator import ActuationPermit
from mpc_lab.contracts import Identity, Snapshot
import nominal_permit
from nominal_permit import NominalDemoPermit


class NominalPermitTests(unittest.TestCase):
    def setUp(self):
        self.identity = Identity(
            "cpu-lifecycle-v1", "synthetic-unqualified-v1", "a" * 64, "b" * 64, "c" * 64,
        )
        self.snapshot = Snapshot(
            "cpu-test-only", "cpu-test-clock", 10.1, 16.0, 0, (), self.identity,
        )
        self.permit = NominalDemoPermit(
            self.identity, self.snapshot.run_id, self.snapshot.clock_domain,
            "d" * 64, "e" * 64, 10.0, 20.0, approved=True,
        )
        self.arguments = {
            "enabled": True, "expected_identity": self.identity,
            "current_source_sha256": "d" * 64, "protocol_sha256": "e" * 64,
            "current_clock_domain": "cpu-test-clock",
            "budget": 32.0, "entered_at": 10.0, "deadline": 10.4,
            "now": 10.2, "issued_commands": 0,
        }

    def refusal(self, permit=None, snapshot=None, **changes):
        return (self.permit if permit is None else permit).refusal_reason(
            self.snapshot if snapshot is None else snapshot,
            **(self.arguments | changes),
        )

    def test_explicit_cpu_fixture_does_not_grant_native_authority(self):
        self.assertIsNone(self.refusal())
        self.assertNotIsInstance(self.permit, ActuationPermit)
        self.assertFalse(ActuationPermit(self.identity).allows(self.identity))

    def test_default_off_and_non_boolean_acknowledgment(self):
        for permit, enabled in (
            (replace(self.permit, approved=False), True),
            (replace(self.permit, approved=1), True),
            (self.permit, False),
            (self.permit, 1),
            (replace(self.permit, scope="native-qualified"), True),
        ):
            with self.subTest(permit=permit, enabled=enabled):
                self.assertEqual(self.refusal(permit, enabled=enabled), "nominal_not_authorized")

    def test_source_protocol_and_calibration_cannot_be_substituted(self):
        self.assertEqual(self.refusal(current_source_sha256="f" * 64),
                         "nominal_source_or_protocol_mismatch")
        self.assertEqual(self.refusal(protocol_sha256="f" * 64),
                         "nominal_source_or_protocol_mismatch")
        qualified = replace(self.identity, calibration_version="native-qualified")
        self.assertEqual(
            self.refusal(replace(self.permit, identity=qualified), expected_identity=qualified),
            "nominal_identity_mismatch",
        )

    def test_forecast_dependencies_invalidate_source_authorization(self):
        sources = {
            name: name.encode() for name in (
                "nominal_permit.py", "nominal_planner.py", "nominal_runtime.py",
                "nominal_scenario.py", "nominal_workload.py",
            )
        }
        with patch.object(nominal_permit, "Path") as path:
            directory = path.return_value.resolve.return_value.parent
            directory.__truediv__.side_effect = lambda name: SimpleNamespace(
                read_bytes=lambda: sources[name],
            )
            with patch.object(nominal_permit, "implementation_hash", return_value="a" * 64):
                initial = nominal_permit.source_digest()
                permit = replace(self.permit, source_sha256=initial)
                self.assertIsNone(self.refusal(permit, current_source_sha256=initial))
                for name in ("nominal_scenario.py", "nominal_workload.py"):
                    original = sources[name]
                    sources[name] += b"\nchanged"
                    changed = nominal_permit.source_digest()
                    self.assertNotEqual(initial, changed)
                    self.assertEqual(
                        self.refusal(permit, current_source_sha256=changed),
                        "nominal_source_or_protocol_mismatch",
                    )
                    sources[name] = original

    def test_run_clock_and_incomplete_snapshot_are_refused(self):
        for field in ("run_id", "clock_domain"):
            self.assertEqual(
                self.refusal(snapshot=replace(self.snapshot, **{field: "other"})),
                "nominal_run_or_clock_mismatch",
            )
        self.assertEqual(
            self.refusal(
                replace(self.permit, clock_domain="other"),
                snapshot=replace(self.snapshot, clock_domain="other"),
            ),
            "nominal_run_or_clock_mismatch",
        )
        self.assertIsNotNone(self.refusal(snapshot=replace(self.snapshot, complete=False)))

    def test_entry_deadline_cannot_restart_after_get_or_solver(self):
        self.assertEqual(self.refusal(deadline=10.6), "nominal_expired_or_restarted_deadline")
        self.assertEqual(self.refusal(entered_at=10.15), "nominal_expired_or_restarted_deadline")
        self.assertEqual(self.refusal(now=10.4), "nominal_expired_or_restarted_deadline")
        self.assertEqual(self.refusal(now=float("nan")), "nominal_invalid_deadline")

    def test_permit_expiry_and_duration_are_bounded(self):
        self.assertEqual(self.refusal(replace(self.permit, expires_at=10.2)),
                         "nominal_expired_or_restarted_deadline")
        self.assertEqual(self.refusal(replace(self.permit, expires_at=70.1)),
                         "nominal_invalid_limits")

    def test_command_limit_is_checked_on_every_submission(self):
        self.assertEqual(self.refusal(issued_commands=8), "nominal_command_limit")
        self.assertEqual(self.refusal(issued_commands=True), "nominal_invalid_limits")
        self.assertEqual(self.refusal(replace(self.permit, max_commands=9)), "nominal_invalid_limits")

    def test_unsupported_budget_and_stale_or_future_state_are_refused(self):
        for budget in (True, 24.0, float("inf")):
            with self.subTest(budget=budget):
                self.assertEqual(self.refusal(budget=budget), "nominal_unsupported_budget")
        self.assertIsNotNone(self.refusal(snapshot=replace(self.snapshot, sampled_at=9.9)))
        self.assertIsNotNone(self.refusal(snapshot=replace(self.snapshot, sampled_at=10.3)))


if __name__ == "__main__":
    unittest.main()
