import random
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import bsdiff4

from deployment.semantics import DeploymentConfig
from env.ota_env import OTAEnv
from evaluation.baselines import (
    ACTION_M,
    POLICIES,
    PUBLICATION_BASELINE_NAMES,
    greedy_smallest_delta,
    run_policy,
)


BLOCK_SIZE = 4
A = b"AAAA"
B = b"BBBB"
C = b"CCCC"


def write_fixture(directory, name, blocks):
    path = Path(directory) / name
    path.write_bytes(b"".join(blocks))
    return path


class BaselinePolicyTests(unittest.TestCase):
    def test_publication_baselines_produce_valid_replay_and_required_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = Path(tmp) / "new.bin"
            new_file.write_bytes(C + B + b"DDDD" + b"AA")

            for policy_name in PUBLICATION_BASELINE_NAMES:
                with self.subTest(policy=policy_name):
                    result = run_policy(
                        policy_name,
                        old_file,
                        new_file,
                        block_size=BLOCK_SIZE,
                    )
                    deployment = result["deployment"]

                    self.assertIn(policy_name, POLICIES)
                    self.assertTrue(result["completed"])
                    self.assertTrue(result["replay_validity"])
                    self.assertTrue(result["replay_valid"])
                    self.assertEqual(result["replay_errors"], "")
                    self.assertGreater(result["encoding_op_count"], 0)
                    self.assertIn("package_size_bytes", deployment)
                    self.assertIn("network_bytes", deployment)
                    self.assertIn("peak_ram_bytes", deployment)
                    self.assertIn("peak_persistent_storage_bytes", deployment)
                    self.assertIn("flash_write_bytes", deployment)
                    self.assertIn("install_time_s", deployment)
                    self.assertIn("rollback_ready", deployment)
                    self.assertIn("runtime", result)
                    self.assertIn("runtime_s", result)

    def test_backup_safe_copy_delta_preserves_rollback_for_shrink(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = Path(tmp) / "new.bin"
            new_file.write_bytes(A + b"BB")

            result = run_policy(
                "backup_safe_copy_delta",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
            )

            self.assertTrue(result["replay_validity"])
            self.assertTrue(result["deployment"]["rollback_ready"])
            self.assertGreaterEqual(result["deployment"]["backup_area_bytes"], 2 * BLOCK_SIZE)

    def test_deployment_aware_greedy_preserves_rollback_for_shrink(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = Path(tmp) / "new.bin"
            new_file.write_bytes(A + b"BB")

            result = run_policy(
                "deployment_aware_greedy",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(ram_budget_bytes=BLOCK_SIZE),
            )

            self.assertTrue(result["completed"])
            self.assertTrue(result["replay_validity"])
            self.assertTrue(result["deployment"]["rollback_ready"])
            self.assertEqual(result["deployment"]["budget_violation_count"], 0)
            self.assertGreaterEqual(result["deployment"]["backup_area_bytes"], 2 * BLOCK_SIZE)

    def test_deployment_aware_greedy_avoids_ram_budget_violation_when_raw_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [b"AAAZ"])

            result = run_policy(
                "deployment_aware_greedy",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(ram_budget_bytes=BLOCK_SIZE),
            )

            self.assertTrue(result["completed"])
            self.assertTrue(result["replay_validity"])
            self.assertTrue(result["deployment"]["rollback_ready"])
            self.assertEqual(result["deployment"]["budget_violation_count"], 0)
            self.assertEqual(result["deployment"]["budget_violations"], [])

    def test_ab_deployment_invalidity_does_not_change_replay_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [C])

            result = run_policy(
                "sequential_m",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(
                    enable_ab_slots=True,
                    health_check_mode="forced_fail",
                ),
                interruption_percentages=(),
            )

            self.assertTrue(result["replay_validity"])
            self.assertFalse(result["completed"])
            self.assertFalse(result["ab_update_valid"])
            self.assertTrue(result["ab_rollback_ready"])
            self.assertEqual(result["deployment"]["install_state"], "rolled_back")
            self.assertTrue(result["deployment"]["rollback_after_failed_boot"])

    def test_interruption_benchmark_reports_checkpoint_recovery_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = write_fixture(tmp, "new.bin", [C, A, B])

            result = run_policy(
                "backup_safe_copy_delta",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(checkpoint_interval_ops=2),
                interruption_percentages=(0.5,),
            )

            interruption = result["interruption_results"][0]
            summary = result["interruption_summary"]

            self.assertTrue(interruption["checkpoint_available"])
            self.assertTrue(interruption["resumed_from_checkpoint"])
            self.assertTrue(interruption["recovery_success"])
            self.assertTrue(interruption["final_replay_validity"])
            self.assertTrue(interruption["rollback_success"])
            self.assertGreater(interruption["recovery_cost"]["replayed_operation_count"], 0)
            self.assertGreater(interruption["extra_network_bytes"], 0)
            self.assertGreater(interruption["extra_flash_writes"], 0)
            self.assertEqual(summary["checkpoint_resume_count"], 1)
            self.assertEqual(summary["failed_recovery_count"], 0)

    def test_interruption_without_checkpoint_rolls_back_and_reinstalls_when_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = write_fixture(tmp, "new.bin", [C, A, B])

            result = run_policy(
                "sequential_mb",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
                interruption_percentages=(0.5,),
            )

            interruption = result["interruption_results"][0]

            self.assertFalse(interruption["checkpoint_available"])
            self.assertEqual(interruption["recovery_strategy"], "rollback_reinstall")
            self.assertTrue(interruption["rollback_success"])
            self.assertTrue(interruption["recovery_success"])
            self.assertTrue(interruption["final_replay_validity"])
            self.assertGreater(interruption["extra_network_bytes"], 0)
            self.assertGreater(interruption["extra_flash_writes"], 0)

    def test_interruption_recovery_fails_when_no_checkpoint_and_rollback_is_unsafe(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, C])

            result = run_policy(
                "sequential_m",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
                interruption_percentages=(0.5,),
            )

            interruption = result["interruption_results"][0]
            summary = result["interruption_summary"]

            self.assertFalse(interruption["checkpoint_available"])
            self.assertFalse(interruption["rollback_ready_at_interruption"])
            self.assertFalse(interruption["rollback_success"])
            self.assertFalse(interruption["recovery_success"])
            self.assertFalse(interruption["final_replay_validity"])
            self.assertIn("rollback is not ready", interruption["recovery_errors"])
            self.assertEqual(summary["failed_recovery_count"], 1)

    def test_sequential_policy_handles_empty_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [])

            result = run_policy(
                "sequential_m",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
            )

            self.assertTrue(result["completed"])
            self.assertTrue(result["replay_validity"])
            self.assertEqual(result["blocks_total"], 0)
            self.assertEqual(result["encoding_op_count"], 1)

    def test_sequential_policy_handles_empty_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [])
            new_file = write_fixture(tmp, "new.bin", [A, b"BB"])

            result = run_policy(
                "sequential_m",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
            )

            self.assertTrue(result["completed"])
            self.assertTrue(result["replay_validity"])
            self.assertEqual(result["encoding_op_count"], 2)
            self.assertEqual(result["new_size_bytes"], len(A + b"BB"))

    def test_sequential_policy_handles_empty_source_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [])
            new_file = write_fixture(tmp, "new.bin", [])

            result = run_policy(
                "sequential_m",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
            )

            self.assertTrue(result["completed"])
            self.assertTrue(result["replay_validity"])
            self.assertEqual(result["encoding_op_count"], 0)

    def test_backup_aware_copy_delta_preserves_rollback_with_less_memory_than_sequential_mb(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = write_fixture(tmp, "new.bin", [C, B, A])

            backup_aware = run_policy(
                "backup_aware_copy_delta",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
            )
            sequential_m = run_policy(
                "sequential_m",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
            )
            sequential_mb = run_policy(
                "sequential_mb",
                old_file,
                new_file,
                block_size=BLOCK_SIZE,
            )

            self.assertIn("backup_aware_copy_delta", POLICIES)
            self.assertTrue(backup_aware["replay_validity"])
            self.assertTrue(backup_aware["deployment"]["rollback_ready"])
            self.assertFalse(sequential_m["deployment"]["rollback_ready"])
            self.assertLess(backup_aware["memory_cost"], sequential_mb["memory_cost"])
            self.assertGreaterEqual(
                backup_aware["deployment"]["backup_area_bytes"],
                2 * BLOCK_SIZE,
            )

    def test_greedy_smallest_delta_uses_top_k_similarity_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(
                tmp,
                "old.bin",
                [b"AAAA", b"BBBB", b"CCCC", b"DDDD", b"EEEE"],
            )
            new_file = write_fixture(tmp, "new.bin", [b"AAAZ", b"BBCZ"])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE, similarity_top_k=3)
            env.reset()

            with mock.patch("evaluation.baselines.bsdiff4.diff", wraps=bsdiff4.diff) as diff:
                action, block_index = greedy_smallest_delta(env, random.Random(1))

            env.close()

            self.assertEqual(action, ACTION_M)
            self.assertIn(block_index, [0, 1])
            self.assertLessEqual(diff.call_count, 6)


if __name__ == "__main__":
    unittest.main()
