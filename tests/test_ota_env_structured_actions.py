import tempfile
import unittest
from pathlib import Path

from deployment.semantics import DeploymentConfig
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


class OTAStructuredActionTests(unittest.TestCase):
    def test_structured_target_actions_produce_valid_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = write_fixture(tmp, "new.bin", [A, C, D, B])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            env.step({"op": "keep", "target": 0})
            env.step({"op": "copy", "target": 1, "source": 2})
            env.step({"op": "raw_insert", "target": 2})
            _, _, done, _ = env.step({"op": "append", "target": 3})

            operations = env.get_encoding_ops()
            result = env.validate_encoding()
            env.close()

            self.assertTrue(done)
            self.assertTrue(result.valid)
            self.assertEqual([operation["op"] for operation in operations], [
                "keep",
                "copy",
                "raw_insert",
                "append",
            ])

    def test_structured_delta_action_produces_valid_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [b"AAAZ"])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            _, _, done, _ = env.step({"op": "delta", "target": 0, "source": 0})

            operations = env.get_encoding_ops()
            result = env.validate_encoding()
            env.close()

            self.assertTrue(done)
            self.assertTrue(result.valid)
            self.assertEqual(operations[0]["op"], "delta")
            self.assertEqual(operations[0]["codec"], "bsdiff4")

    def test_control_actions_are_replayable_when_target_already_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [A])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            env.step({"op": "checkpoint", "step_id": "before-rollback"})
            env.step({"op": "backup", "source": 0, "backup_id": "old-0"})
            env.step({"op": "rollback", "backup_id": "old-0"})
            env.step({"op": "verify"})
            _, _, done, _ = env.step({"op": "commit"})

            operations = env.get_encoding_ops()
            result = env.validate_encoding()
            metrics = env.get_deployment_metrics()
            env.close()

            self.assertTrue(done)
            self.assertTrue(result.valid)
            self.assertEqual([operation["op"] for operation in operations], [
                "checkpoint",
                "backup",
                "rollback",
                "verify",
                "commit",
            ])
            self.assertGreaterEqual(metrics["checkpoint_count"], 1)

    def test_action_mask_exposes_structured_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [A, C, D])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            first_mask = env.get_action_mask(0)
            append_mask = env.get_action_mask(2)
            action_names = env.get_action_family_names()
            env.close()

            self.assertGreater(env.action_space.n, env.legacy_action_space.n)
            self.assertIn("raw_insert", action_names)
            self.assertEqual(first_mask[env.action_id("M")], 1.0)
            self.assertEqual(first_mask[env.action_id("MB")], 1.0)
            self.assertEqual(first_mask[env.action_id("keep")], 1.0)
            self.assertEqual(first_mask[env.action_id("raw_insert")], 1.0)
            self.assertEqual(first_mask[env.action_id("backup")], 1.0)
            self.assertEqual(first_mask[env.action_id("append")], 0.0)
            self.assertEqual(append_mask[env.action_id("append")], 1.0)

    def test_invalid_structured_actions_are_rejected_without_consuming_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [A, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            with self.assertRaisesRegex(ValueError, "keep is invalid"):
                env.step({"op": "keep", "target": 1})
            with self.assertRaisesRegex(ValueError, "copy source does not match"):
                env.step({"op": "copy", "target": 1, "source": 0})
            with self.assertRaisesRegex(ValueError, "restricted to installed tail"):
                env.step({"op": "delete", "target": 0})

            self.assertEqual(env.get_metrics()["blocks_remaining"], 2)
            self.assertEqual(env.get_metrics()["steps"], 0)
            self.assertEqual(env.get_encoding_ops(), [])
            env.close()

    def test_structured_backup_preserves_deployment_rollback_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [C])
            env = OTAEnv(
                str(old_file),
                str(new_file),
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(require_backup_for_overwrite=True),
            )
            env.reset()

            env.step({"op": "backup", "source": 0, "backup_id": "old-0"})
            _, _, done, _ = env.step({"op": "raw_insert", "target": 0})

            result = env.validate_encoding()
            metrics = env.get_deployment_metrics()
            env.close()

            self.assertTrue(done)
            self.assertTrue(result.valid)
            self.assertTrue(metrics["rollback_ready"])
            self.assertEqual(metrics["unsafe_overwrite_count"], 0)
            self.assertEqual(metrics["backup_area_bytes"], BLOCK_SIZE)


if __name__ == "__main__":
    unittest.main()
