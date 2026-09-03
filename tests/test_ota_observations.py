import tempfile
import unittest
from unittest import mock
from pathlib import Path

import bsdiff4
import numpy as np

from deployment.semantics import DeploymentConfig
from env.ota_env import OTAEnv


BLOCK_SIZE = 4
A = b"AAAA"
B = b"BBBB"
C = b"CCCC"


def write_fixture(directory, name, blocks):
    path = Path(directory) / name
    path.write_bytes(b"".join(blocks))
    return path


class OTAObservationTests(unittest.TestCase):
    def test_reset_observation_matches_observation_space_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)

            observation, _ = env.reset()
            env.close()

            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(observation.dtype, np.float32)

    def test_step_observation_matches_observation_space_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            env.reset()

            observation, _, _, _ = env.step(0, 0)
            env.close()

            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(observation.dtype, np.float32)

    def test_remaining_mask_changes_after_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            initial_observation, _ = env.reset()
            layout = env.get_observation_feature_layout()

            next_observation, _, _, _ = env.step(0, 0)
            env.close()

            start, end = layout["remaining_mask"]
            self.assertEqual(initial_observation[start:end].tolist(), [1.0, 1.0])
            self.assertEqual(next_observation[start:end].tolist(), [0.0, 1.0])

    def test_exact_match_and_top_k_similarity_features_are_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE, similarity_top_k=2)
            observation, info = env.reset()
            layout = env.get_observation_feature_layout()
            candidates = env.get_similarity_candidates(0)
            env.close()

            exact_start, exact_end = layout["exact_match_mask"]
            self.assertEqual(observation[exact_start:exact_end].tolist(), [1.0, 0.0])

            self.assertEqual(candidates[0]["source_index"], 1)
            self.assertEqual(candidates[0]["similarity_score"], 1.0)
            self.assertTrue(candidates[0]["exact_match"])

            source_index_start, _ = layout["top_0_source_index_fraction"]
            score_start, _ = layout["top_0_similarity_score"]
            exact_start, _ = layout["top_0_exact_match"]
            self.assertEqual(observation[source_index_start], 1.0)
            self.assertEqual(observation[score_start], 1.0)
            self.assertEqual(observation[exact_start], 1.0)
            self.assertIn("current_similarity_candidates", info)
            self.assertEqual(info["current_similarity_candidates"][0]["source_index"], 1)

    def test_numeric_features_are_scaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            initial_observation, _ = env.reset()
            layout = env.get_observation_feature_layout()

            next_observation, _, _, _ = env.step(0, 0)
            env.close()

            block_size_start, _ = layout["block_size_fraction"]
            memory_start, _ = layout["memory_cost_scaled"]
            step_start, _ = layout["step_count_fraction"]
            encoding_start, _ = layout["encoding_cost_scaled"]

            self.assertLess(initial_observation[block_size_start], 1.0)
            self.assertLessEqual(initial_observation[memory_start], 1.0)
            self.assertEqual(initial_observation[step_start], 0.0)
            self.assertGreater(next_observation[step_start], 0.0)
            self.assertGreaterEqual(next_observation[encoding_start], 0.0)
            self.assertTrue(np.all(np.isfinite(next_observation)))

    def test_deployment_budget_features_reflect_simulator_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [A, C])
            env = OTAEnv(
                str(old_file),
                str(new_file),
                block_size=BLOCK_SIZE,
                deployment_config=DeploymentConfig(
                    storage_budget_bytes=16,
                    ram_budget_bytes=2,
                    bandwidth_bytes_per_s=250.0,
                    checkpoint_interval_ops=2,
                    require_backup_for_overwrite=True,
                ),
            )
            initial_observation, _ = env.reset()
            layout = env.get_observation_feature_layout()

            after_keep, _, _, _ = env.step({"op": "keep", "target": 0})
            after_raw, _, _, _ = env.step({"op": "raw_insert", "target": 1})
            metrics = env.get_deployment_metrics()
            env.close()

            storage_start, _ = layout["remaining_storage_budget_fraction"]
            ram_start, _ = layout["remaining_ram_budget_fraction"]
            bandwidth_start, _ = layout["bandwidth_bytes_per_s_scaled"]
            rollback_start, _ = layout["rollback_ready"]
            checkpoint_start, _ = layout["checkpoint_interval_progress"]
            unsafe_start, _ = layout["unsafe_overwrite_count_scaled"]
            peak_storage_start, _ = layout["peak_storage_bytes_scaled"]
            peak_ram_start, _ = layout["peak_ram_bytes_scaled"]

            self.assertAlmostEqual(initial_observation[storage_start], 0.5)
            self.assertEqual(initial_observation[ram_start], 1.0)
            self.assertGreater(initial_observation[bandwidth_start], 0.0)
            self.assertEqual(initial_observation[rollback_start], 1.0)
            self.assertEqual(initial_observation[checkpoint_start], 0.0)
            self.assertAlmostEqual(after_keep[checkpoint_start], 0.5)
            self.assertEqual(after_raw[rollback_start], 0.0)
            self.assertGreater(after_raw[unsafe_start], 0.0)
            self.assertLess(after_raw[ram_start], 0.0)
            self.assertGreater(after_raw[peak_storage_start], initial_observation[peak_storage_start])
            self.assertGreater(after_raw[peak_ram_start], initial_observation[peak_ram_start])
            self.assertGreater(metrics["unsafe_overwrite_count"], 0)
            self.assertIn("ram_budget_exceeded", metrics["budget_violations"])

    def test_top_k_padding_is_fixed_when_there_are_fewer_sources_than_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE, similarity_top_k=3)
            observation, _ = env.reset()
            layout = env.get_observation_feature_layout()
            candidates = env.get_similarity_candidates(0)
            env.close()

            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(len(candidates), 3)
            self.assertEqual(candidates[1]["source_index"], -1)
            self.assertEqual(candidates[2]["source_index"], -1)
            padded_start, _ = layout["top_2_source_index_fraction"]
            self.assertEqual(observation[padded_start], -1.0)

    def test_variable_old_new_block_counts_keep_observation_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A])
            new_file = write_fixture(tmp, "new.bin", [A, B, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE)
            initial_observation, _ = env.reset()
            layout = env.get_observation_feature_layout()

            next_observation, _, _, _ = env.step(0, 2)
            env.close()

            remaining_start, remaining_end = layout["remaining_mask"]
            exact_start, exact_end = layout["exact_match_mask"]
            self.assertEqual(initial_observation.shape, env.observation_space.shape)
            self.assertEqual(next_observation.shape, env.observation_space.shape)
            self.assertEqual(initial_observation[remaining_start:remaining_end].tolist(), [1.0, 1.0, 1.0])
            self.assertEqual(next_observation[remaining_start:remaining_end].tolist(), [1.0, 1.0, 0.0])
            self.assertEqual(initial_observation[exact_start:exact_end].tolist(), [1.0, 0.0, 0.0])

    def test_observation_feature_names_describe_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(tmp, "old.bin", [A, B])
            new_file = write_fixture(tmp, "new.bin", [B, C])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE, similarity_top_k=2)
            observation, _ = env.reset()
            names = env.get_observation_feature_names()
            layout = env.get_observation_feature_layout()
            env.close()

            self.assertEqual(len(names), observation.shape[0])
            self.assertIn("target_position_0", names)
            self.assertIn("encoding_cost_scaled", names)
            self.assertIn("remaining_storage_budget_fraction", names)
            self.assertIn("rollback_ready", names)
            self.assertIn("peak_ram_bytes_scaled", names)
            self.assertIn("top_1_source_index_fraction", names)
            self.assertIn("remaining_mask", layout)

    def test_delta_generation_uses_top_k_similarity_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_fixture(
                tmp,
                "old.bin",
                [b"AAAA", b"BBBB", b"CCCC", b"DDDD", b"EEEE"],
            )
            new_file = write_fixture(tmp, "new.bin", [b"AAAZ"])
            env = OTAEnv(str(old_file), str(new_file), block_size=BLOCK_SIZE, similarity_top_k=2)
            env.reset()

            with mock.patch("env.ota_env.bsdiff4.diff", wraps=bsdiff4.diff) as diff:
                _, _, done, _ = env.step(0, 0)

            replay_result = env.validate_encoding()
            env.close()

            self.assertTrue(done)
            self.assertTrue(replay_result.valid)
            self.assertLessEqual(diff.call_count, 2)


if __name__ == "__main__":
    unittest.main()
