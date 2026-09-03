import tempfile
import unittest
from pathlib import Path

import bsdiff4

from deployment.semantics import (
    DeploymentConfig,
    DeploymentSimulator,
    simulate_deployment,
)
from encoding.replay import (
    bytes_sha256,
    make_backup_operation,
    make_checkpoint_operation,
    make_commit_operation,
    make_copy_operation,
    make_delete_operation,
    make_delta_operation,
    make_raw_insert_operation,
    make_truncate_operation,
    make_verify_operation,
)
from env.ota_env import OTAEnv


BLOCK_SIZE = 4
A = b"AAAA"
B = b"BBBB"
C = b"CCCC"
D = b"DDDD"


def write_fixture(directory, name, blocks):
    path = Path(directory) / name
    path.write_bytes(b"".join(blocks))
    return path


class DeploymentSemanticsTests(unittest.TestCase):
    def test_deployment_config_rejects_nonpositive_transfer_rates(self):
        invalid_configs = [
            {"bandwidth_bytes_per_s": 0.0},
            {"flash_write_bytes_per_s": 0.0},
            {"patch_apply_bytes_per_s": -1.0},
        ]

        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DeploymentConfig(**kwargs)

    def test_overwrite_without_backup_marks_rollback_unsafe(self):
        operations = [
            make_delta_operation(
                source=0,
                target=0,
                patch_bytes=bsdiff4.diff(A, C),
                target_block=C,
            )
        ]

        result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[C, B])

        self.assertTrue(result.valid)
        self.assertFalse(result.metrics["rollback_ready"])
        self.assertEqual(result.metrics["unsafe_overwrite_count"], 1)

    def test_backup_before_overwrite_preserves_rollback_readiness(self):
        operations = [
            make_backup_operation(source=0, backup_id="old-block-0"),
            make_delta_operation(
                source=0,
                target=0,
                patch_bytes=bsdiff4.diff(A, C),
                target_block=C,
            ),
        ]

        result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[C, B])

        self.assertTrue(result.valid)
        self.assertTrue(result.metrics["rollback_ready"])
        self.assertEqual(result.metrics["backup_area_bytes"], BLOCK_SIZE)

    def test_delta_can_read_from_backup_area(self):
        operations = [
            make_backup_operation(source=0, backup_id="old-block-0"),
            make_backup_operation(source=1, backup_id="old-block-1"),
            make_delta_operation(
                source=0,
                source_area="backup",
                source_backup_id="old-block-0",
                target=1,
                patch_bytes=bsdiff4.diff(A, C),
                target_block=C,
            ),
        ]

        result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[A, C])

        self.assertTrue(result.valid)
        self.assertTrue(result.metrics["rollback_ready"])

    def test_storage_budget_violation_is_reported(self):
        config = DeploymentConfig(storage_budget_bytes=8, staging_strategy="full_package")
        operations = [
            make_backup_operation(source=0, backup_id="old-block-0"),
            make_delta_operation(
                source=0,
                target=0,
                patch_bytes=bsdiff4.diff(A, C),
                target_block=C,
            ),
        ]

        result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[C, B], config=config)

        self.assertIn("storage_budget_exceeded", result.metrics["budget_violations"])
        self.assertGreater(result.metrics["budget_violation_count"], 0)

    def test_ram_budget_violation_is_reported(self):
        config = DeploymentConfig(ram_budget_bytes=1)
        operations = [
            make_copy_operation(source=0, target=1, target_block=A),
        ]

        result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[A, A], config=config)

        self.assertIn("ram_budget_exceeded", result.metrics["budget_violations"])

    def test_bandwidth_controls_download_time(self):
        slow = DeploymentConfig(bandwidth_bytes_per_s=100.0)
        fast = DeploymentConfig(bandwidth_bytes_per_s=10_000.0)
        operations = [
            make_copy_operation(source=0, target=1, target_block=A),
        ]

        slow_result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[A, A], config=slow)
        fast_result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[A, A], config=fast)

        self.assertGreater(slow_result.metrics["download_time_s"], fast_result.metrics["download_time_s"])

    def test_checkpoint_can_resume_after_interruption(self):
        config = DeploymentConfig(checkpoint_interval_ops=1)
        operations = [
            make_backup_operation(source=0, backup_id="old-block-0"),
            make_delta_operation(
                source=0,
                target=0,
                patch_bytes=bsdiff4.diff(A, C),
                target_block=C,
            ),
        ]
        simulator = DeploymentSimulator([A, B], BLOCK_SIZE, config=config)
        simulator.apply_operation(operations[0])
        checkpoint = simulator.latest_checkpoint()
        simulator.interrupt()

        resumed = DeploymentSimulator([], BLOCK_SIZE, config=config, state=simulator.state)
        resumed.restore_checkpoint(checkpoint)
        resumed.apply_operation(operations[1])
        resumed.complete()

        self.assertEqual(resumed.state.install_state, "complete")
        self.assertEqual(b"".join(resumed.state.installed_blocks), C + B)
        self.assertEqual(resumed.metrics()["checkpoint_count"], 2)

    def test_checkpoint_restore_restores_operation_index_and_rollback_readiness(self):
        simulator = DeploymentSimulator([A, B], BLOCK_SIZE)
        simulator.apply_operation(make_checkpoint_operation(step_id="safe-point"))
        checkpoint = simulator.latest_checkpoint()
        simulator.apply_operation(make_raw_insert_operation(target=0, data=C))
        simulator.interrupt()

        self.assertFalse(simulator.metrics()["rollback_ready"])
        self.assertEqual(simulator.metrics()["unsafe_overwrite_count"], 1)

        resumed = DeploymentSimulator([], BLOCK_SIZE, state=simulator.state)
        resumed.restore_checkpoint(checkpoint)
        metrics = resumed.metrics()

        self.assertEqual(resumed.state.applied_operations, 1)
        self.assertTrue(metrics["rollback_ready"])
        self.assertEqual(metrics["unsafe_overwrite_count"], 0)
        self.assertEqual(b"".join(resumed.state.installed_blocks), A + B)

    def test_automatic_checkpoint_records_completed_operation_index(self):
        simulator = DeploymentSimulator(
            [A, B],
            BLOCK_SIZE,
            config=DeploymentConfig(checkpoint_interval_ops=2),
        )
        simulator.apply_operation(make_backup_operation(source=0, backup_id="old-block-0"))
        simulator.apply_operation(make_raw_insert_operation(target=0, data=C))

        checkpoint = simulator.latest_checkpoint()

        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.operation_index, 2)

    def test_raw_insert_verify_checkpoint_commit_update_deployment_metrics(self):
        operations = [
            make_checkpoint_operation(step_id="start"),
            make_backup_operation(source=0, backup_id="old-block-0"),
            make_raw_insert_operation(target=0, data=D),
            make_verify_operation(sha256=bytes_sha256(D + B), size_bytes=len(D + B)),
            make_commit_operation(),
        ]

        result = simulate_deployment([A, B], operations, BLOCK_SIZE, expected_blocks=[D, B])

        self.assertTrue(result.valid)
        self.assertEqual(result.metrics["install_state"], "complete")
        self.assertGreaterEqual(result.metrics["checkpoint_count"], 1)
        self.assertTrue(result.metrics["rollback_ready"])
        self.assertGreater(result.metrics["package_size_bytes"], 0)

    def test_delete_and_truncate_support_shrinking_deployment(self):
        deleted = simulate_deployment(
            [A, B, C, D],
            [make_delete_operation(target=1, count=2)],
            BLOCK_SIZE,
            expected_blocks=[A, D],
        )
        truncated = simulate_deployment(
            [A, B, C],
            [make_truncate_operation(new_size_bytes=len(A + B[:2]))],
            BLOCK_SIZE,
            expected_blocks=[A, B[:2]],
        )

        self.assertTrue(deleted.valid)
        self.assertTrue(truncated.valid)
        self.assertEqual(truncated.metrics["installed_image_bytes"], len(A + B[:2]))

    def test_verify_hash_mismatch_fails_deployment(self):
        result = simulate_deployment(
            [A],
            [make_verify_operation(sha256=bytes_sha256(B), size_bytes=len(A))],
            BLOCK_SIZE,
            expected_blocks=[A],
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.errors, ["verify sha256 mismatch"])

    def test_full_package_staging_uses_more_peak_storage_than_streaming(self):
        operations = [
            make_backup_operation(source=0, backup_id="old-block-0"),
            make_delta_operation(
                source=0,
                target=0,
                patch_bytes=bsdiff4.diff(A, C),
                target_block=C,
            ),
        ]

        streaming = simulate_deployment(
            [A, B],
            operations,
            BLOCK_SIZE,
            expected_blocks=[C, B],
            config=DeploymentConfig(staging_strategy="streaming"),
        )
        full_package = simulate_deployment(
            [A, B],
            operations,
            BLOCK_SIZE,
            expected_blocks=[C, B],
            config=DeploymentConfig(staging_strategy="full_package"),
        )

        self.assertGreater(
            full_package.metrics["peak_persistent_storage_bytes"],
            streaming.metrics["peak_persistent_storage_bytes"],
        )

    def test_ab_disabled_preserves_existing_metric_shape(self):
        result = simulate_deployment(
            [A, B],
            [make_raw_insert_operation(target=0, data=C)],
            BLOCK_SIZE,
            expected_blocks=[C, B],
        )

        self.assertTrue(result.valid)
        self.assertNotIn("ab_enabled", result.metrics)
        self.assertNotIn("ab_rollback_ready", result.metrics)
        self.assertIn("rollback_ready", result.metrics)

    def test_successful_ab_install_and_activation(self):
        config = DeploymentConfig(
            enable_ab_slots=True,
            slot_capacity_bytes=len(C + B),
            reboot_downtime_seconds=3.5,
            health_check_mode="always_pass",
        )
        result = simulate_deployment(
            [A, B],
            [make_raw_insert_operation(target=0, data=C)],
            BLOCK_SIZE,
            expected_blocks=[C, B],
            config=config,
        )
        metrics = result.metrics

        self.assertTrue(result.valid)
        self.assertTrue(metrics["ab_enabled"])
        self.assertTrue(metrics["ab_update_valid"])
        self.assertTrue(metrics["ab_rollback_ready"])
        self.assertTrue(metrics["activation_success"])
        self.assertTrue(metrics["boot_health_success"])
        self.assertFalse(metrics["rollback_after_failed_boot"])
        self.assertFalse(metrics["rollback_success"])
        self.assertEqual(metrics["active_slot"], "B")
        self.assertEqual(metrics["inactive_slot"], "A")
        self.assertEqual(metrics["boot_attempt_counter"], 1)
        self.assertEqual(metrics["reboot_count"], 1)
        self.assertEqual(metrics["downtime_seconds"], 3.5)
        self.assertEqual(metrics["slot_switch_count"], 1)
        self.assertEqual(metrics["slot_storage_bytes"], len(C + B))
        self.assertFalse(metrics["slot_storage_violation"])

    def test_forced_ab_health_failure_rolls_back_to_previous_active_slot(self):
        config = DeploymentConfig(
            enable_ab_slots=True,
            max_boot_attempts=2,
            reboot_downtime_seconds=4.0,
            health_check_mode="forced_fail",
        )
        result = simulate_deployment(
            [A, B],
            [make_raw_insert_operation(target=0, data=C)],
            BLOCK_SIZE,
            expected_blocks=[C, B],
            config=config,
        )
        metrics = result.metrics

        self.assertFalse(result.valid)
        self.assertEqual(result.state.installed_blocks, [A, B])
        self.assertIn("A/B boot health check failed", result.errors)
        self.assertFalse(metrics["ab_update_valid"])
        self.assertTrue(metrics["ab_rollback_ready"])
        self.assertTrue(metrics["activation_success"])
        self.assertFalse(metrics["boot_health_success"])
        self.assertTrue(metrics["rollback_after_failed_boot"])
        self.assertTrue(metrics["rollback_success"])
        self.assertEqual(metrics["active_slot"], "A")
        self.assertEqual(metrics["inactive_slot"], "B")
        self.assertEqual(metrics["boot_attempt_counter"], 2)
        self.assertEqual(metrics["reboot_count"], 2)
        self.assertEqual(metrics["downtime_seconds"], 8.0)
        self.assertEqual(metrics["slot_switch_count"], 2)

    def test_ab_slot_capacity_violation_is_reported(self):
        config = DeploymentConfig(enable_ab_slots=True, slot_capacity_bytes=len(C + B) - 1)
        result = simulate_deployment(
            [A, B],
            [make_raw_insert_operation(target=0, data=C)],
            BLOCK_SIZE,
            expected_blocks=[C, B],
            config=config,
        )
        metrics = result.metrics

        self.assertFalse(result.valid)
        self.assertFalse(metrics["ab_update_valid"])
        self.assertFalse(metrics["activation_success"])
        self.assertTrue(metrics["ab_rollback_ready"])
        self.assertTrue(metrics["slot_storage_violation"])
        self.assertEqual(metrics["slot_storage_bytes"], len(C + B))
        self.assertIn("slot_capacity_exceeded", metrics["budget_violations"])
        self.assertIn("A/B inactive slot capacity exceeded", result.errors)

    def test_active_slot_install_can_bypass_inactive_slot_capacity_when_allowed(self):
        config = DeploymentConfig(
            enable_ab_slots=True,
            slot_capacity_bytes=len(C + B) - 1,
            require_inactive_slot_install=False,
        )
        result = simulate_deployment(
            [A, B],
            [make_raw_insert_operation(target=0, data=C)],
            BLOCK_SIZE,
            expected_blocks=[C, B],
            config=config,
        )
        metrics = result.metrics

        self.assertTrue(result.valid)
        self.assertTrue(metrics["ab_update_valid"])
        self.assertTrue(metrics["activation_success"])
        self.assertTrue(metrics["boot_health_success"])
        self.assertFalse(metrics["slot_storage_violation"])
        self.assertEqual(metrics["active_slot"], "A")
        self.assertEqual(metrics["inactive_slot"], "B")
        self.assertEqual(metrics["slot_switch_count"], 0)
        self.assertEqual(metrics["slot_storage_bytes"], len(C + B))

    def test_active_slot_health_failure_rolls_back_without_slot_switch(self):
        config = DeploymentConfig(
            enable_ab_slots=True,
            require_inactive_slot_install=False,
            max_boot_attempts=2,
            reboot_downtime_seconds=4.0,
            health_check_mode="forced_fail",
        )
        result = simulate_deployment(
            [A, B],
            [make_raw_insert_operation(target=0, data=C)],
            BLOCK_SIZE,
            expected_blocks=[C, B],
            config=config,
        )
        metrics = result.metrics

        self.assertFalse(result.valid)
        self.assertEqual(result.state.installed_blocks, [A, B])
        self.assertIn("A/B boot health check failed", result.errors)
        self.assertFalse(metrics["ab_update_valid"])
        self.assertTrue(metrics["activation_success"])
        self.assertFalse(metrics["boot_health_success"])
        self.assertTrue(metrics["rollback_after_failed_boot"])
        self.assertTrue(metrics["rollback_success"])
        self.assertEqual(metrics["active_slot"], "A")
        self.assertEqual(metrics["inactive_slot"], "B")
        self.assertEqual(metrics["slot_switch_count"], 0)
        self.assertEqual(metrics["boot_attempt_counter"], 2)
        self.assertEqual(metrics["reboot_count"], 2)
        self.assertEqual(metrics["downtime_seconds"], 8.0)

    def test_ab_boot_attempt_counter_stops_after_successful_health_check(self):
        config = DeploymentConfig(enable_ab_slots=True, max_boot_attempts=5)
        result = simulate_deployment(
            [A],
            [make_raw_insert_operation(target=0, data=C)],
            BLOCK_SIZE,
            expected_blocks=[C],
            config=config,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.metrics["boot_attempt_counter"], 1)
        self.assertEqual(result.metrics["reboot_count"], 1)
        self.assertEqual(result.metrics["max_boot_attempts"], 5)

    def test_environment_exposes_deployment_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [C, B])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()
            env.step(1, 0)
            env.step(0, 1)
            metrics = env.get_metrics()
            replay_result = env.validate_encoding()
            env.close()

            self.assertTrue(replay_result.valid)
            self.assertIn("deployment", metrics)
            self.assertGreater(metrics["deployment"]["package_size_bytes"], 0)
            self.assertTrue(metrics["deployment"]["rollback_ready"])


if __name__ == "__main__":
    unittest.main()
