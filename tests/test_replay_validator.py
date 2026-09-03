import hashlib
import tempfile
import unittest
from pathlib import Path

import bsdiff4

from encoding.operations import DeltaCodec
from encoding.replay import (
    blocks_to_bytes,
    make_append_operation,
    make_backup_operation,
    make_checkpoint_operation,
    make_commit_operation,
    make_copy_operation,
    make_delete_operation,
    make_delta_operation,
    make_keep_operation,
    make_raw_insert_operation,
    make_rollback_operation,
    make_truncate_operation,
    make_verify_operation,
    read_blocks,
    replay_operations,
    validate_update,
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


class ReplayValidatorTests(unittest.TestCase):
    def test_copy_operation_recreates_target_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [A, A])

            result = validate_update(
                old_file,
                new_file,
                [make_copy_operation(source=0, target=1, target_block=A)],
                BLOCK_SIZE,
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.errors, [])

    def test_delta_operation_recreates_target_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [C])
            patch = bsdiff4.diff(A, C)
            operation = make_delta_operation(source=0, target=0, patch_bytes=patch, target_block=C)

            result = validate_update(
                old_file,
                new_file,
                [operation],
                BLOCK_SIZE,
            )

            self.assertEqual(operation["codec"], "bsdiff4")
            self.assertTrue(result.valid)
            self.assertEqual(result.errors, [])

    def test_non_bsdiff_delta_codec_is_metadata_only_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [C])
            operation = make_delta_operation(
                source=0,
                target=0,
                patch_bytes=bsdiff4.diff(A, C),
                target_block=C,
                codec=DeltaCodec.XDELTA3,
            )

            result = validate_update(old_file, new_file, [operation], BLOCK_SIZE)

            self.assertEqual(operation["codec"], "xdelta3")
            self.assertFalse(result.valid)
            self.assertEqual(result.errors, ["delta codec 'xdelta3' is metadata-only; replay supports bsdiff4"])

    def test_keep_operation_validates_existing_target_block_without_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [A, B])

            result = validate_update(
                old_file,
                new_file,
                [make_keep_operation(target=0, target_block=A), make_keep_operation(target=1, target_block=B)],
                BLOCK_SIZE,
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.errors, [])

    def test_raw_insert_operation_recreates_uncopied_target_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [D])

            result = validate_update(
                old_file,
                new_file,
                [make_raw_insert_operation(target=0, data=D)],
                BLOCK_SIZE,
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.errors, [])

    def test_backup_operation_preserves_overwritten_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, B])
            operations = [
                make_backup_operation(source=0, backup_id="old-block-0"),
                make_copy_operation(source=1, target=0, target_block=B),
            ]

            initial_blocks = read_blocks(old_file, BLOCK_SIZE)
            replay = replay_operations(initial_blocks, operations)
            result = validate_update(old_file, new_file, operations, BLOCK_SIZE)

            self.assertTrue(result.valid)
            self.assertEqual(replay.state.backups["old-block-0"].block, A)

    def test_copy_can_read_from_backup_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [A, A])
            operations = [
                make_backup_operation(source=0, backup_id="old-block-0"),
                make_copy_operation(
                    source=0,
                    source_area="backup",
                    source_backup_id="old-block-0",
                    target=1,
                    target_block=A,
                ),
            ]

            result = validate_update(old_file, new_file, operations, BLOCK_SIZE)

            self.assertTrue(result.valid)

    def test_append_operation_supports_growing_target_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [A, B])

            result = validate_update(
                old_file,
                new_file,
                [make_append_operation(B, target=1)],
                BLOCK_SIZE,
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.errors, [])

    def test_delete_operation_supports_shrinking_target_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = write_fixture(tmp, "new.bin", [A, C])

            result = validate_update(
                old_file,
                new_file,
                [make_delete_operation(target=1)],
                BLOCK_SIZE,
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.final_size_bytes, len(A + C))

    def test_truncate_operation_supports_exact_byte_shrink(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = Path(tmp) / "new.bin"
            new_file.write_bytes(A + B[:2])

            result = validate_update(
                old_file,
                new_file,
                [make_truncate_operation(new_size_bytes=len(A + B[:2]))],
                BLOCK_SIZE,
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.final_size_bytes, len(A + B[:2]))

    def test_verify_operation_catches_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [A])

            valid = validate_update(
                old_file,
                new_file,
                [make_verify_operation(sha256=hashlib.sha256(A).hexdigest(), size_bytes=len(A))],
                BLOCK_SIZE,
            )
            invalid = validate_update(
                old_file,
                new_file,
                [make_verify_operation(sha256=hashlib.sha256(B).hexdigest(), size_bytes=len(A))],
                BLOCK_SIZE,
            )

            self.assertTrue(valid.valid)
            self.assertFalse(invalid.valid)
            self.assertEqual(invalid.errors, ["verify sha256 mismatch"])

    def test_backup_rollback_restores_a_single_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            restored_file = write_fixture(tmp, "restored.bin", [A, B])
            patch = bsdiff4.diff(A, C)
            operations = [
                make_backup_operation(source=0, backup_id="old-block-0"),
                make_delta_operation(source=0, target=0, patch_bytes=patch, target_block=C),
                make_rollback_operation(backup_id="old-block-0"),
            ]

            result = validate_update(old_file, restored_file, operations, BLOCK_SIZE)

            self.assertTrue(result.valid)
            self.assertTrue(result.rolled_back)

    def test_interrupted_update_can_resume_to_valid_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [C, D])
            operations = [
                make_backup_operation(source=0, backup_id="old-block-0"),
                make_delta_operation(source=0, target=0, patch_bytes=bsdiff4.diff(A, C), target_block=C),
                make_backup_operation(source=1, backup_id="old-block-1"),
                make_delta_operation(source=1, target=1, patch_bytes=bsdiff4.diff(B, D), target_block=D),
            ]

            partial = replay_operations(
                read_blocks(old_file, BLOCK_SIZE),
                operations,
                stop_after=2,
            )
            resumed = validate_update(
                old_file,
                new_file,
                operations[2:],
                BLOCK_SIZE,
                state=partial.state,
            )

            self.assertTrue(partial.interrupted)
            self.assertFalse(partial.completed)
            self.assertTrue(resumed.valid)

    def test_interrupted_replay_reports_partial_state_until_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [C, B])
            operations = [
                make_backup_operation(source=0, backup_id="old-block-0"),
                make_delta_operation(source=0, target=0, patch_bytes=bsdiff4.diff(A, C), target_block=C),
            ]

            interrupted = validate_update(
                old_file,
                new_file,
                operations,
                BLOCK_SIZE,
                stop_after=1,
            )
            resumed = validate_update(
                old_file,
                new_file,
                operations[1:],
                BLOCK_SIZE,
                state=interrupted.state,
            )

            self.assertTrue(interrupted.interrupted)
            self.assertFalse(interrupted.valid)
            self.assertEqual(blocks_to_bytes(interrupted.state.blocks), A + B)
            self.assertTrue(resumed.valid)

    def test_interrupted_update_can_rollback_to_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            operations = [
                make_backup_operation(source=0, backup_id="old-block-0"),
                make_delta_operation(source=0, target=0, patch_bytes=bsdiff4.diff(A, C), target_block=C),
            ]

            partial = replay_operations(read_blocks(old_file, BLOCK_SIZE), operations)
            rolled_back = replay_operations([], [make_rollback_operation()], state=partial.state)

            self.assertTrue(rolled_back.rolled_back)
            self.assertEqual(blocks_to_bytes(rolled_back.state.blocks), old_file.read_bytes())

    def test_checkpoint_commit_and_rollback_are_represented_consistently(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            restored_file = write_fixture(tmp, "restored.bin", [A])
            operations = [
                make_checkpoint_operation(step_id="start"),
                make_backup_operation(source=0, backup_id="old-block-0"),
                make_raw_insert_operation(target=0, data=C),
                make_commit_operation(),
                make_rollback_operation(backup_id="old-block-0"),
            ]

            result = validate_update(old_file, restored_file, operations, BLOCK_SIZE)

            self.assertTrue(result.valid)
            self.assertTrue(result.committed)
            self.assertTrue(result.rolled_back)

    def test_environment_generated_encoding_ops_replay_successfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [C, B])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            env.step(1, 0)
            env.step(0, 1)
            result = env.validate_encoding()
            env.close()

            self.assertTrue(result.valid)
            self.assertEqual(result.errors, [])
            self.assertEqual(env.get_encoding_ops()[0]["op"], "backup")

    def test_environment_legacy_actions_generate_structured_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [A, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            env.step(0, 0)
            env.step(1, 1)
            operations = env.get_encoding_ops()
            result = env.validate_encoding()
            env.close()

            self.assertTrue(result.valid)
            self.assertEqual(operations[0]["op"], "keep")
            self.assertEqual(operations[1]["op"], "backup")
            self.assertEqual(operations[2]["op"], "delta")
            self.assertEqual(operations[2]["codec"], "bsdiff4")

    def test_environment_emits_truncate_for_shrinking_target_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B, C])
            new_file = Path(tmp) / "new.bin"
            new_file.write_bytes(A + B[:2])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            env.step(0, 0)
            env.step(0, 1)
            operations = env.get_encoding_ops()
            metrics = env.get_deployment_metrics()
            result = env.validate_encoding()
            env.close()

            self.assertTrue(result.valid)
            self.assertEqual(operations[-1]["op"], "truncate")
            self.assertEqual(operations[-1]["new_size_bytes"], len(A + B[:2]))
            self.assertEqual(metrics["installed_image_bytes"], len(A + B[:2]))


if __name__ == "__main__":
    unittest.main()
