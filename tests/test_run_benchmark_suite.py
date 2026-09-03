import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_benchmark_suite import main, parse_args


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


class BenchmarkSuiteRunnerTests(unittest.TestCase):
    def test_suite_writes_jsonl_csv_and_markdown_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data = b"AAAABBBB"
            new_data = b"AAAACCCC"
            write_file(tmp, "old.bin", old_data)
            write_file(tmp, "new.bin", new_data)
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "suite_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "pair_a",
                            "enabled": True,
                            "domain": "synthetic",
                            "scenario": "small changed block",
                            "artifact_type": "binary",
                            "old_file": "old.bin",
                            "new_file": "new.bin",
                            "old_size_bytes": len(old_data),
                            "new_size_bytes": len(new_data),
                            "old_sha256": sha256(old_data),
                            "new_sha256": sha256(new_data),
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )
            output_dir = Path(tmp) / "suite-results"

            args = parse_args(
                [
                    "--manifests",
                    str(manifest_path),
                    "--policies",
                    "sequential_m",
                    "--block-sizes",
                    "4",
                    "8",
                    "--deployment-config",
                    "default={}",
                    "--deployment-config",
                    'checkpointed={"checkpoint_interval_ops":1}',
                    "--interruption-setting",
                    "short=0.5",
                    "--seeds",
                    "3",
                    "4",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            result = main(args)

            jsonl_path = Path(result["paths"]["jsonl"])
            csv_path = Path(result["paths"]["csv"])
            markdown_path = Path(result["paths"]["markdown"])
            ledger_path = Path(result["paths"]["attempt_ledger"])

            rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            markdown = markdown_path.read_text(encoding="utf-8")
            with csv_path.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 8)
            self.assertEqual(len(ledger_rows), 8)
            self.assertEqual(len(csv_rows), 8)
            self.assertIn("| Policy | Block Size | Deployment Config |", markdown)
            self.assertIn("sequential_m", markdown)
            self.assertEqual({row["dataset_id"] for row in rows}, {"suite_manifest"})
            self.assertEqual({row["pair_id"] for row in rows}, {"pair_a"})
            self.assertEqual({row["policy"] for row in rows}, {"sequential_m"})
            self.assertEqual({row["block_size_bytes"] for row in rows}, {4, 8})
            self.assertEqual({row["deployment_config_name"] for row in rows}, {"default", "checkpointed"})
            self.assertEqual({row["interruption_setting_name"] for row in rows}, {"short"})
            self.assertEqual({row["seed"] for row in rows}, {3, 4})

            for row in rows:
                self.assertEqual(row["old_sha256"], sha256(old_data))
                self.assertEqual(row["new_sha256"], sha256(new_data))
                self.assertEqual(row["artifact_hashes"]["old_sha256"], sha256(old_data))
                self.assertIn("git_commit", row)
                self.assertIn("python_version", row)
                self.assertIn("dependency_versions", row)
                self.assertIn("numpy", row["dependency_versions"])
                self.assertIn("runtime_s", row)
                self.assertIn("suite_result_runtime_s", row)
                self.assertTrue(row["replay_validity"])
                self.assertIn("deployment", row)
                self.assertIn("network_bytes", row["deployment"])
                self.assertIn("flash_write_bytes", row["deployment"])
                self.assertIn("peak_ram_bytes", row["deployment"])
                self.assertIn("interruption_summary", row)

            self.assertIn("dependency_versions_numpy", csv_rows[0])
            self.assertIn("deployment_network_bytes", csv_rows[0])
            self.assertIn("artifact_hashes_old_sha256", csv_rows[0])
            self.assertEqual({row["status"] for row in ledger_rows}, {"completed"})
            self.assertTrue(all(row["result_row_written"] for row in ledger_rows))
            self.assertEqual({row["dataset"] for row in ledger_rows}, {"suite_manifest"})
            self.assertEqual({row["pair_id"] for row in ledger_rows}, {"pair_a"})
            self.assertEqual({row["policy"] for row in ledger_rows}, {"sequential_m"})
            self.assertEqual({row["deployment_config"] for row in ledger_rows}, {"default", "checkpointed"})
            self.assertEqual({row["seed"] for row in ledger_rows}, {3, 4})
            self.assertEqual({row["block_size"] for row in ledger_rows}, {4, 8})
            self.assertTrue(all(row["start_time"] for row in ledger_rows))
            self.assertTrue(all(row["end_time"] for row in ledger_rows))
            self.assertTrue(all(row["runtime_seconds"] >= 0 for row in ledger_rows))

    def test_suite_can_disable_interruption_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data = b"AAAA"
            new_data = b"BBBB"
            write_file(tmp, "old.bin", old_data)
            write_file(tmp, "new.bin", new_data)
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "no_interrupt_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "pair_a",
                            "domain": "synthetic",
                            "scenario": "single block",
                            "old_file": "old.bin",
                            "new_file": "new.bin",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )

            result = main(
                parse_args(
                    [
                        "--manifests",
                        str(manifest_path),
                        "--policies",
                        "sequential_m",
                        "--no-interruption-eval",
                        "--output-dir",
                        str(Path(tmp) / "results"),
                    ]
                )
            )

            row = result["results"][0]

            self.assertEqual(row["interruption_setting_name"], "none")
            self.assertEqual(row["interruption_percentages"], [])
            self.assertEqual(row["interruption_results"], [])
            self.assertEqual(row["interruption_summary"]["scenario_count"], 0)

    def test_suite_accepts_ab_deployment_config_and_writes_flat_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data = b"AAAA"
            new_data = b"BBBB"
            write_file(tmp, "old.bin", old_data)
            write_file(tmp, "new.bin", new_data)
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "ab_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "pair_ab",
                            "domain": "synthetic",
                            "scenario": "single block A/B",
                            "old_file": "old.bin",
                            "new_file": "new.bin",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )
            output_dir = Path(tmp) / "results"

            result = main(
                parse_args(
                    [
                        "--manifests",
                        str(manifest_path),
                        "--policies",
                        "sequential_m",
                        "--deployment-config",
                        'ab={"enable_ab_slots":true,"slot_capacity_bytes":4,"reboot_downtime_seconds":2.5}',
                        "--no-interruption-eval",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            )

            row = result["results"][0]
            csv_path = Path(result["paths"]["csv"])
            with csv_path.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))

            self.assertTrue(row["replay_validity"])
            self.assertTrue(row["completed"])
            self.assertTrue(row["ab_enabled"])
            self.assertTrue(row["ab_update_valid"])
            self.assertTrue(row["ab_rollback_ready"])
            self.assertEqual(row["slot_storage_bytes"], 4)
            self.assertEqual(row["reboot_count"], 1)
            self.assertEqual(row["downtime_seconds"], 2.5)
            self.assertIn("ab_update_valid", csv_rows[0])
            self.assertIn("deployment_ab_update_valid", csv_rows[0])

    def test_forced_fail_ab_profile_writes_failed_boot_rollback_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data = b"AAAA"
            new_data = b"BBBB"
            write_file(tmp, "old.bin", old_data)
            write_file(tmp, "new.bin", new_data)
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "forced_fail_ab_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "pair_ab_forced_fail",
                            "domain": "synthetic",
                            "scenario": "forced failed boot A/B",
                            "old_file": "old.bin",
                            "new_file": "new.bin",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )
            output_dir = Path(tmp) / "results"
            config_path = Path("configs/deployment/forced_fail_ab_supplemental.json")

            result = main(
                parse_args(
                    [
                        "--manifests",
                        str(manifest_path),
                        "--policies",
                        "backup_safe_copy_delta",
                        "--deployment-config-file",
                        str(config_path),
                        "--no-interruption-eval",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            )

            row = result["results"][0]
            ledger_path = Path(result["paths"]["attempt_ledger"])
            csv_path = Path(result["paths"]["csv"])
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            with csv_path.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))

            self.assertTrue(row["replay_validity"])
            self.assertFalse(row["completed"])
            self.assertTrue(row["ab_enabled"])
            self.assertFalse(row["ab_update_valid"])
            self.assertTrue(row["ab_rollback_ready"])
            self.assertTrue(row["activation_success"])
            self.assertFalse(row["boot_health_success"])
            self.assertTrue(row["rollback_after_failed_boot"])
            self.assertTrue(row["rollback_success"])
            self.assertEqual(row["reboot_count"], 2)
            self.assertEqual(row["downtime_seconds"], 60.0)
            self.assertEqual(row["slot_switch_count"], 2)
            self.assertEqual(row["deployment"]["install_state"], "rolled_back")
            self.assertEqual(row["deployment"]["health_verdict"], "fail")
            self.assertEqual(ledger_rows[0]["status"], "error")
            self.assertEqual(ledger_rows[0]["error_type"], "incomplete_run")
            self.assertTrue(ledger_rows[0]["result_row_written"])
            self.assertEqual(csv_rows[0]["ab_update_valid"], "False")
            self.assertEqual(csv_rows[0]["deployment_rollback_success"], "True")

    def test_suite_records_timeout_and_error_attempts_without_result_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data = b"AAAA"
            new_data = b"BBBB"
            write_file(tmp, "old.bin", old_data)
            write_file(tmp, "new.bin", new_data)
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "error_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "pair_a",
                            "domain": "synthetic",
                            "scenario": "single block",
                            "old_file": "old.bin",
                            "new_file": "new.bin",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )
            output_dir = Path(tmp) / "results"
            args = parse_args(
                [
                    "--manifests",
                    str(manifest_path),
                    "--policies",
                    "sequential_m",
                    "sequential_mb",
                    "--no-interruption-eval",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            with mock.patch(
                "scripts.run_benchmark_suite.run_policy",
                side_effect=[TimeoutError("too slow"), RuntimeError("boom")],
            ):
                result = main(args)

            rows = [
                json.loads(line)
                for line in Path(result["paths"]["jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ledger_rows = [
                json.loads(line)
                for line in Path(result["paths"]["attempt_ledger"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(rows, [])
            self.assertEqual([row["status"] for row in ledger_rows], ["timeout", "error"])
            self.assertEqual([row["error_type"] for row in ledger_rows], ["TimeoutError", "RuntimeError"])
            self.assertTrue(all(not row["result_row_written"] for row in ledger_rows))


if __name__ == "__main__":
    unittest.main()
