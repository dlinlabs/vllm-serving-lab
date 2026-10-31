"""Unit and integration tests for gemini_timing_contract.py."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from gemini_timing_contract import (
    BatchExecutionInterval,
    ClockDomain,
    NativeFairStatisticalProtocol,
    PairedEvaluationBlock,
    ProbeOverheadGuard,
    ProbeOverheadRecord,
    RequestTimingTrace,
    SerialCapacityScreener,
)


class TestClockDomain(unittest.TestCase):
    def setUp(self):
        self.domain1 = ClockDomain(domain_id="host-monotonic", boot_id="boot-001")
        self.domain1_clone = ClockDomain(domain_id="host-monotonic", boot_id="boot-001")
        self.domain2 = ClockDomain(domain_id="container-monotonic", boot_id="boot-001")
        self.domain_reboot = ClockDomain(domain_id="host-monotonic", boot_id="boot-002")
        self.non_monotonic = ClockDomain(domain_id="host-monotonic", boot_id="boot-001", is_monotonic=False)

    def test_compatible_domain(self):
        self.domain1.validate_compatible(self.domain1_clone)

    def test_incompatible_domain_id(self):
        with self.assertRaises(ValueError) as ctx:
            self.domain1.validate_compatible(self.domain2)
        self.assertIn("Incompatible clock domains", str(ctx.exception))

    def test_incompatible_boot_id(self):
        with self.assertRaises(ValueError) as ctx:
            self.domain1.validate_compatible(self.domain_reboot)
        self.assertIn("boot mismatch", str(ctx.exception))

    def test_non_monotonic_clock(self):
        with self.assertRaises(ValueError) as ctx:
            self.domain1.validate_compatible(self.non_monotonic)
        self.assertIn("Non-monotonic clocks cannot be used", str(ctx.exception))


class TestRequestTimingTrace(unittest.TestCase):
    def setUp(self):
        self.domain = ClockDomain("node-0", "boot-1")

    def test_valid_admitted_trace(self):
        trace = RequestTimingTrace(
            request_id="req-1",
            clock_domain=self.domain,
            prompt_tokens=128,
            output_tokens=64,
            scheduled_arrival=10.000,
            gateway_received=10.010,
            gateway_admitted=10.012,
            vllm_enqueued=10.015,
            first_token_sampled=10.055,
            first_content_delivered=10.065,
            stream_terminal=11.200,
        )
        trace.validate_monotonic_order()

        self.assertAlmostEqual(trace.ingress_lag_s, 0.010)
        self.assertAlmostEqual(trace.ttft_s, 0.065)
        self.assertAlmostEqual(trace.engine_first_residence_s, 0.040)
        self.assertAlmostEqual(trace.egress_transport_s, 0.010)
        # (11.200 - 10.065) / 63 = 1.135 / 63 ≈ 0.01801587s (18ms per token)
        self.assertAlmostEqual(trace.request_mean_tpot_s, 1.135 / 63)
        self.assertTrue(trace.is_composite_good(max_ttft_s=0.5, max_tpot_s=0.05))

    def test_slo_violation_ttft(self):
        trace = RequestTimingTrace(
            request_id="req-slow-ttft",
            clock_domain=self.domain,
            prompt_tokens=4096,
            output_tokens=10,
            scheduled_arrival=10.0,
            gateway_received=10.0,
            gateway_admitted=10.0,
            vllm_enqueued=10.0,
            first_token_sampled=10.6,
            first_content_delivered=10.7,  # 700ms > 500ms
            stream_terminal=10.9,
        )
        self.assertFalse(trace.is_composite_good())

    def test_rejected_request_handling(self):
        trace = RequestTimingTrace(
            request_id="req-reject",
            clock_domain=self.domain,
            prompt_tokens=128,
            output_tokens=0,
            scheduled_arrival=1.0,
            gateway_received=1.005,
            gateway_admitted=None,
            vllm_enqueued=None,
            first_token_sampled=None,
            first_content_delivered=None,
            stream_terminal=None,
            rejected=True,
        )
        trace.validate_monotonic_order()
        self.assertFalse(trace.is_composite_good())

    def test_causality_violations(self):
        # gateway received before scheduled arrival
        with self.assertRaises(ValueError):
            RequestTimingTrace(
                request_id="req-bad",
                clock_domain=self.domain,
                prompt_tokens=128,
                output_tokens=10,
                scheduled_arrival=2.0,
                gateway_received=1.9,
                gateway_admitted=2.1,
                vllm_enqueued=2.2,
                first_token_sampled=2.3,
                first_content_delivered=2.4,
                stream_terminal=2.5,
            ).validate_monotonic_order()


class TestBatchExecutionInterval(unittest.TestCase):
    def setUp(self):
        self.domain = ClockDomain("node-0", "boot-1")

    def test_valid_interval(self):
        interval = BatchExecutionInterval(
            batch_id=1,
            clock_domain=self.domain,
            cpu_call_start=1.000,
            cpu_call_end=1.003,       # 3ms CPU return
            gpu_span_start=1.001,
            gpu_span_end=1.011,       # 10ms GPU span
            kernel_busy_union_s=0.009, # 9ms active kernels
            prefill_requests=0,
            prefill_tokens=0,
            decode_requests=8,
            decode_tokens=8,
        )
        interval.validate()
        self.assertAlmostEqual(interval.cpu_duration_s, 0.003)
        self.assertAlmostEqual(interval.gpu_span_s, 0.010)
        self.assertAlmostEqual(interval.gpu_gap_s, 0.001)
        self.assertAlmostEqual(interval.gpu_utilization, 0.9)

    def test_kernel_busy_exceeds_gpu_span(self):
        interval = BatchExecutionInterval(
            batch_id=2,
            clock_domain=self.domain,
            cpu_call_start=1.0,
            cpu_call_end=1.1,
            gpu_span_start=1.0,
            gpu_span_end=1.010,
            kernel_busy_union_s=0.015,  # 15ms > 10ms
            prefill_requests=1,
            prefill_tokens=128,
            decode_requests=0,
            decode_tokens=0,
        )
        with self.assertRaises(ValueError):
            interval.validate()


class TestSerialCapacityScreener(unittest.TestCase):
    def setUp(self):
        self.screener = SerialCapacityScreener()

    def test_empirical_falsification_case(self):
        """Replicates exact falsification: 2004 serial bound vs 2449 actual completions."""
        eval_result = self.screener.evaluate_falsification(
            window_s=90.0,
            prompt_tokens=128,
            actual_completions=2449,
            carry_in_completions=0,
        )
        self.assertTrue(eval_result["is_falsified"])
        self.assertEqual(eval_result["serial_capacity_bound"], 2004)
        self.assertAlmostEqual(eval_result["time_deficit_s"], 19.983, places=2)
        self.assertEqual(eval_result["required_carry_in_to_reconcile"], 445)
        self.assertIn("FALSIFIED", eval_result["conclusion"])

    def test_within_capacity_case(self):
        eval_result = self.screener.evaluate_falsification(
            window_s=90.0,
            prompt_tokens=128,
            actual_completions=1500,
            carry_in_completions=0,
        )
        self.assertFalse(eval_result["is_falsified"])
        self.assertEqual(eval_result["serial_capacity_bound"], 2004)
        self.assertEqual(eval_result["time_deficit_s"], 0.0)


class TestProbeOverheadGuard(unittest.TestCase):
    def setUp(self):
        self.guard = ProbeOverheadGuard(max_probe_latency_s=0.015, max_slot_fraction=0.05)
        self.domain = ClockDomain("node-0", "boot-1")

    def test_lightweight_rpc_passes(self):
        probe = ProbeOverheadRecord(
            probe_name="read_only_rpc",
            clock_domain=self.domain,
            start_time=1.0,
            end_time=1.002,  # 2ms
            payload_bytes=4096,
            is_blocking_gpu=False,
            is_tracing_cupti=False,
        )
        valid, violations = self.guard.validate_probe(probe, control_slot_s=1.0)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

    def test_cupti_profiling_rejected(self):
        probe = ProbeOverheadRecord(
            probe_name="cupti_trace",
            clock_domain=self.domain,
            start_time=1.0,
            end_time=1.005,
            payload_bytes=100000,
            is_blocking_gpu=False,
            is_tracing_cupti=True,
        )
        valid, violations = self.guard.validate_probe(probe, control_slot_s=1.0)
        self.assertFalse(valid)
        self.assertTrue(any("CUPTI" in v for v in violations))

    def test_gpu_blocking_rejected(self):
        probe = ProbeOverheadRecord(
            probe_name="blocking_sync",
            clock_domain=self.domain,
            start_time=1.0,
            end_time=1.003,
            payload_bytes=512,
            is_blocking_gpu=True,
            is_tracing_cupti=False,
        )
        valid, violations = self.guard.validate_probe(probe, control_slot_s=1.0)
        self.assertFalse(valid)
        self.assertTrue(any("synchronous GPU synchronization" in v for v in violations))


class TestNativeFairStatisticalProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = NativeFairStatisticalProtocol(
            min_superiority_margin_req_s=0.5,
            max_solver_overrun_rate=0.01,
        )

    def test_held_out_leakage_detected(self):
        with self.assertRaises(ValueError) as ctx:
            self.protocol.check_held_out_isolation([160001, 160002, 160021])
        self.assertIn("CRITICAL PROTOCOL VIOLATION", str(ctx.exception))
        self.assertIn("160021", str(ctx.exception))

    def test_paired_evaluation_superior(self):
        blocks = [
            PairedEvaluationBlock(
                block_id=f"block-{i}",
                seed=160001 + i,
                offered_requests=1000,
                duration_s=90.0,
                policy_a_name="Fixed16",
                policy_a_good_completions=500,
                policy_a_rejections=200,
                policy_a_solver_overruns=0,
                policy_b_name="AdaptiveMPC",
                policy_b_good_completions=570,  # +70 completions in 90s ≈ +0.777 req/s
                policy_b_rejections=130,
                policy_b_solver_overruns=2,     # 2 / 1000 = 0.002 <= 0.01
            )
            for i in range(8)
        ]
        res = self.protocol.evaluate_paired_blocks(blocks)
        self.assertEqual(res["verdict"], "CONFIRMED_SUPERIOR")
        self.assertTrue(res["margin_satisfied"])
        self.assertTrue(res["statistically_significant"])
        self.assertTrue(res["guards_passed"])

    def test_paired_evaluation_insufficient_margin(self):
        blocks = [
            PairedEvaluationBlock(
                block_id=f"block-{i}",
                seed=160001 + i,
                offered_requests=1000,
                duration_s=90.0,
                policy_a_name="Fixed16",
                policy_a_good_completions=500,
                policy_a_rejections=200,
                policy_a_solver_overruns=0,
                policy_b_name="AdaptiveMPC",
                policy_b_good_completions=515,  # +15 completions in 90s ≈ +0.166 req/s < 0.5 req/s
                policy_b_rejections=185,
                policy_b_solver_overruns=0,
            )
            for i in range(8)
        ]
        res = self.protocol.evaluate_paired_blocks(blocks)
        self.assertEqual(res["verdict"], "EQUIVALENT_OR_INSUFFICIENT")
        self.assertFalse(res["margin_satisfied"])


if __name__ == "__main__":
    unittest.main()
