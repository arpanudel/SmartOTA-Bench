import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from deployment.semantics import DeploymentConfig
from env.ota_env import OTAEnv, RewardConstraintConfig


BLOCK_SIZE = 4
A = b"AAAA"
C = b"CCCC"


def write_fixture(directory, name, blocks):
    path = Path(directory) / name
    path.write_bytes(b"".join(blocks))
    return path


class OTARewardTests(unittest.TestCase):
    def test_budget_violation_and_unsafe_overwrite_are_penalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [C])

            unsafe_env = OTAEnv(
                str(old_file),
                str(new_file),
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(
                    ram_budget_bytes=1,
                    require_backup_for_overwrite=True,
                ),
            )
            unsafe_env.reset()
            _, unsafe_reward, unsafe_done, _ = unsafe_env.step({"op": "raw_insert", "target": 0})
            unsafe_metrics = unsafe_env.get_deployment_metrics()
            unsafe_env.close()

            safe_env = OTAEnv(
                str(old_file),
                str(new_file),
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(
                    ram_budget_bytes=BLOCK_SIZE,
                    require_backup_for_overwrite=True,
                ),
            )
            safe_env.reset()
            safe_env.step({"op": "backup", "source": 0, "backup_id": "old-0"})
            _, safe_reward, safe_done, _ = safe_env.step({"op": "raw_insert", "target": 0})
            safe_total_reward = safe_env.get_metrics()["reward"]
            safe_metrics = safe_env.get_deployment_metrics()
            safe_env.close()

            self.assertTrue(unsafe_done)
            self.assertTrue(safe_done)
            self.assertLess(unsafe_reward, -5.0)
            self.assertLess(unsafe_reward, safe_total_reward)
            self.assertIn("ram_budget_exceeded", unsafe_metrics["budget_violations"])
            self.assertGreater(unsafe_metrics["unsafe_overwrite_count"], 0)
            self.assertEqual(safe_metrics["budget_violation_count"], 0)
            self.assertEqual(safe_metrics["unsafe_overwrite_count"], 0)
            self.assertGreater(safe_reward, 0.0)

    def test_excessive_flash_writes_are_penalized_after_budget_is_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [A])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            env.step({"op": "backup", "source": 0, "backup_id": "backup-1"})
            _, second_reward, _, _ = env.step({"op": "backup", "source": 0, "backup_id": "backup-2"})
            _, third_reward, _, _ = env.step({"op": "backup", "source": 0, "backup_id": "backup-3"})
            metrics = env.get_deployment_metrics()
            env.close()

            self.assertGreater(metrics["flash_write_bytes"], len(A) + len(A))
            self.assertLess(third_reward, second_reward - 0.5)
            self.assertLess(third_reward, -0.5)

    def test_invalid_terminal_replay_is_strongly_penalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [A])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            invalid = SimpleNamespace(valid=False, errors=["target mismatch"])
            with mock.patch.object(env, "validate_encoding", return_value=invalid):
                _, reward, done, _ = env.step({"op": "keep", "target": 0})
            env.close()

            self.assertTrue(done)
            self.assertLess(reward, -9.0)

    def test_configurable_reward_constraints_penalize_deployment_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [C])
            deployment_config = DeploymentConfig(require_backup_for_overwrite=False)

            unconstrained = OTAEnv(
                str(old_file),
                str(new_file),
                block_size=BLOCK_SIZE,
                deployment_config=deployment_config,
            )
            constrained = OTAEnv(
                str(old_file),
                str(new_file),
                block_size=BLOCK_SIZE,
                deployment_config=deployment_config,
                reward_constraints=RewardConstraintConfig(
                    package_size_weight=1.0,
                    flash_write_weight=1.0,
                ),
            )

            unconstrained.reset()
            constrained.reset()
            _, unconstrained_reward, _, _ = unconstrained.step({"op": "raw_insert", "target": 0})
            _, constrained_reward, _, _ = constrained.step({"op": "raw_insert", "target": 0})
            unconstrained.close()
            constrained.close()

            self.assertLess(constrained_reward, unconstrained_reward)


if __name__ == "__main__":
    unittest.main()
