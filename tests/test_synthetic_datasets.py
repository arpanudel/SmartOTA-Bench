import tempfile
import unittest

from datasets import SYNTHETIC_CASES, generate_synthetic_dataset, load_manifest, validate_manifest_schema


class SyntheticDatasetTests(unittest.TestCase):
    def test_generate_synthetic_dataset_writes_all_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = generate_synthetic_dataset(
                tmp,
                block_size_bytes=64,
                block_count=6,
                seed=7,
            )
            validate_manifest_schema(manifest_path)
            manifest = load_manifest(manifest_path)

            self.assertEqual(len(manifest.pairs), len(SYNTHETIC_CASES))
            self.assertEqual(
                {pair.extra["synthetic_case"] for pair in manifest.pairs},
                set(SYNTHETIC_CASES),
            )

            growing = manifest.get_pair("synthetic_growing")
            shrinking = manifest.get_pair("synthetic_shrinking")
            reordered = manifest.get_pair("synthetic_reordered")
            repeated = manifest.get_pair("synthetic_repeated")
            random_pair = manifest.get_pair("synthetic_random")
            compressed = manifest.get_pair("synthetic_compressed_like")
            adversarial_grow = manifest.get_pair("synthetic_adversarial_grow")
            adversarial_shrink = manifest.get_pair("synthetic_adversarial_shrink")
            adversarial_corruption = manifest.get_pair("synthetic_adversarial_corruption")

            self.assertGreater(growing.new_size_bytes, growing.old_size_bytes)
            self.assertLess(shrinking.new_size_bytes, shrinking.old_size_bytes)
            self.assertEqual(reordered.new_size_bytes, reordered.old_size_bytes)
            self.assertNotEqual(reordered.new_sha256, reordered.old_sha256)
            self.assertEqual(random_pair.new_size_bytes, random_pair.old_size_bytes)
            self.assertNotEqual(random_pair.new_sha256, random_pair.old_sha256)
            self.assertGreater(compressed.old_size_bytes, 0)
            self.assertGreater(compressed.new_size_bytes, 0)
            self.assertGreater(adversarial_grow.new_size_bytes, adversarial_grow.old_size_bytes)
            self.assertLess(adversarial_shrink.new_size_bytes, adversarial_shrink.old_size_bytes)
            self.assertNotEqual(adversarial_corruption.new_sha256, adversarial_corruption.old_sha256)

            old_repeated = repeated.old_path.read_bytes()
            blocks = [
                old_repeated[index : index + repeated.block_size_bytes]
                for index in range(0, len(old_repeated), repeated.block_size_bytes)
            ]
            self.assertLess(len(set(blocks)), len(blocks))

    def test_synthetic_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_manifest = load_manifest(
                generate_synthetic_dataset(first, block_size_bytes=128, block_count=8, seed=11)
            )
            second_manifest = load_manifest(
                generate_synthetic_dataset(second, block_size_bytes=128, block_count=8, seed=11)
            )

            first_records = {
                pair.id: (pair.old_size_bytes, pair.new_size_bytes, pair.old_sha256, pair.new_sha256)
                for pair in first_manifest.pairs
            }
            second_records = {
                pair.id: (pair.old_size_bytes, pair.new_size_bytes, pair.old_sha256, pair.new_sha256)
                for pair in second_manifest.pairs
            }

            self.assertEqual(first_records, second_records)


if __name__ == "__main__":
    unittest.main()
