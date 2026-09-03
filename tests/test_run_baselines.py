import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_baselines import main, parse_args


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_file(directory, name, data):
    path = Path(directory) / name
    path.write_bytes(data)
    return path


def write_manifest(directory, payload):
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class BaselineManifestRunnerTests(unittest.TestCase):
    def test_manifest_mode_runs_all_enabled_pairs_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_a = b"AAAABBBB"
            new_a = b"AAAACCCC"
            old_b = b"AAAABBBB"
            new_b = b"BBBBAAAA"
            write_file(tmp, "old-a.bin", old_a)
            write_file(tmp, "new-a.bin", new_a)
            write_file(tmp, "old-b.bin", old_b)
            write_file(tmp, "new-b.bin", new_b)
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "unit_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "enabled_pair_a",
                            "enabled": True,
                            "domain": "synthetic",
                            "scenario": "small changed block",
                            "artifact_type": "binary",
                            "old_file": "old-a.bin",
                            "new_file": "new-a.bin",
                            "old_size_bytes": len(old_a),
                            "new_size_bytes": len(new_a),
                            "old_sha256": sha256(old_a),
                            "new_sha256": sha256(new_a),
                            "block_size_bytes": 4,
                        },
                        {
                            "id": "enabled_pair_b",
                            "enabled": True,
                            "domain": "synthetic",
                            "scenario": "small reordered blocks",
                            "artifact_type": "binary",
                            "old_file": "old-b.bin",
                            "new_file": "new-b.bin",
                            "old_size_bytes": len(old_b),
                            "new_size_bytes": len(new_b),
                            "old_sha256": sha256(old_b),
                            "new_sha256": sha256(new_b),
                            "block_size_bytes": 4,
                        },
                        {
                            "id": "disabled_pair",
                            "enabled": False,
                            "domain": "synthetic",
                            "scenario": "missing disabled files are ignored",
                            "old_file": "missing-old.bin",
                            "new_file": "missing-new.bin",
                            "block_size_bytes": 4,
                        },
                    ],
                },
            )
            output_dir = Path(tmp) / "results"

            args = parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--policies",
                    "sequential_m",
                    "--seed",
                    "7",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            result = main(args)

            jsonl_path = Path(result["paths"]["jsonl"])
            csv_path = Path(result["paths"]["csv"])
            rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual([row["pair_id"] for row in rows], ["enabled_pair_a", "enabled_pair_b"])
            self.assertEqual({row["dataset_id"] for row in rows}, {"unit_manifest"})
            self.assertEqual({row["baseline_name"] for row in rows}, {"sequential_m"})
            self.assertEqual({row["artifact_type"] for row in rows}, {"binary"})
            self.assertEqual({row["block_size"] for row in rows}, {4})
            self.assertEqual({row["seed"] for row in rows}, {7})
            self.assertTrue(all(row["replay_validity"] for row in rows))
            self.assertTrue(all("deployment" in row for row in rows))
            self.assertTrue(all("package_size_bytes" in row["deployment"] for row in rows))
            self.assertTrue(all("interruption_results" in row for row in rows))
            self.assertTrue(all(row["interruption_summary"]["scenario_count"] == 3 for row in rows))

            with csv_path.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual([row["pair_id"] for row in csv_rows], ["enabled_pair_a", "enabled_pair_b"])
            self.assertEqual(
                csv_rows[0]["deployment_package_size_bytes"],
                str(rows[0]["deployment"]["package_size_bytes"]),
            )
            self.assertEqual(csv_rows[0]["replay_validity"], "True")
            self.assertEqual(csv_rows[0]["interruption_summary_scenario_count"], "3")

    def test_manifest_mode_with_no_enabled_pairs_writes_empty_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "empty_enabled_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "planned_pair",
                            "enabled": False,
                            "domain": "synthetic",
                            "scenario": "planned local data",
                            "old_file": "missing-old.bin",
                            "new_file": "missing-new.bin",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )
            output_dir = Path(tmp) / "results"

            result = main(
                parse_args(["--manifest", str(manifest_path), "--output-dir", str(output_dir)])
            )

            jsonl_path = Path(result["paths"]["jsonl"])
            csv_path = Path(result["paths"]["csv"])
            self.assertEqual(jsonl_path.read_text(encoding="utf-8"), "")
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertIn("dataset_id", reader.fieldnames)
                self.assertIn("pair_id", reader.fieldnames)
                self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
