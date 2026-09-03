import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from datasets import DatasetManifestError, load_manifest, validate_manifest_schema
from scripts.build_autoware_style_artifacts import main as build_autoware_artifacts
from scripts.build_autoware_style_artifacts import parse_args as parse_build_args
from scripts.register_dataset_artifacts import main as register_dataset_artifacts
from scripts.run_benchmark_suite import pair_metadata


def write_manifest(directory, payload):
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def autoware_manifest_payload(enabled=False):
    return {
        "name": "autoware-fixture-test",
        "version": 1,
        "base_dir": ".",
        "pairs": [
            {
                "id": "autoware_style_perception_module_update",
                "enabled": enabled,
                "domain": "autoware_style_av_stack",
                "scenario": "autoware_style_perception_module_update",
                "artifact_type": "autoware_module_tar",
                "compression_status": "uncompressed",
                "compression_status_source": "deterministic_tar_derivation",
                "old_file": "missing-old.tar",
                "new_file": "missing-new.tar",
                "old_size_bytes": 1,
                "new_size_bytes": 1,
                "old_sha256": "0" * 64,
                "new_sha256": "1" * 64,
                "block_size_bytes": 16384,
                "source": "unit test deterministic fixture",
                "license_notes": "synthetic fixture",
            }
        ],
    }


def build_quiet(args):
    with redirect_stdout(StringIO()):
        return build_autoware_artifacts(args)


def register_quiet(args):
    with redirect_stdout(StringIO()):
        return register_dataset_artifacts(args)


class AutowareStyleArtifactTests(unittest.TestCase):
    def test_autoware_style_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_result = build_quiet(parse_build_args(["--output-dir", first]))
            second_result = build_quiet(parse_build_args(["--output-dir", second]))

            self.assertEqual(first_result["old_tar"].read_bytes(), second_result["old_tar"].read_bytes())
            self.assertEqual(first_result["new_tar"].read_bytes(), second_result["new_tar"].read_bytes())
            self.assertEqual(first_result["old_metadata"].sha256, second_result["old_metadata"].sha256)
            self.assertEqual(first_result["new_metadata"].sha256, second_result["new_metadata"].sha256)
            self.assertNotEqual(first_result["new_metadata"].sha256, first_result["old_metadata"].sha256)

            with tarfile.open(first_result["new_tar"], "r:") as archive:
                members = {member.name: member for member in archive.getmembers()}
                self.assertIn(
                    "autoware_stack/perception/lidar_object_detection/src/tracker_adapter.cpp",
                    members,
                )
                self.assertIn(
                    "autoware_stack/perception/lidar_object_detection/diagnostics/health_monitor.yaml",
                    members,
                )
                self.assertIn(
                    "autoware_stack/perception/lidar_object_detection/models/mock_lidar_detector.onnx",
                    members,
                )
                self.assertTrue(all(member.mtime == 0 for member in members.values()))
                self.assertTrue(all(member.uid == 0 and member.gid == 0 for member in members.values()))

    def test_enabled_autoware_pair_missing_artifacts_reports_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(tmp, autoware_manifest_payload(enabled=True))

            validate_manifest_schema(manifest_path)
            with self.assertRaisesRegex(DatasetManifestError, "autoware_style_perception_module_update.*does not exist"):
                load_manifest(manifest_path)

    def test_registered_autoware_pair_validates_and_propagates_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_dir = tmp_path / "artifacts"
            build_result = build_quiet(parse_build_args(["--output-dir", str(artifact_dir)]))
            manifest_path = write_manifest(tmp_path, autoware_manifest_payload(enabled=False))

            register_quiet(
                SimpleNamespace(
                    manifest=str(manifest_path),
                    pair="autoware_style_perception_module_update",
                    old_file=str(build_result["old_tar"]),
                    new_file=str(build_result["new_tar"]),
                    block_size_bytes=16384,
                    domain="autoware_style_av_stack",
                    scenario="autoware_style_perception_module_update",
                    source="unit test deterministic fixture",
                    license_notes="synthetic fixture",
                    artifact_type="autoware_module_tar",
                    compression_status="uncompressed",
                    compression_status_source="deterministic_tar_derivation",
                    tier="automotive",
                    old_url=None,
                    new_url=None,
                    enable=True,
                    write=True,
                )
            )

            manifest = load_manifest(manifest_path)
            pair = manifest.get_pair("autoware_style_perception_module_update")
            metadata = pair_metadata(manifest, pair, pair.block_size_bytes)

            self.assertEqual(pair.old_sha256, build_result["old_metadata"].sha256)
            self.assertEqual(pair.new_sha256, build_result["new_metadata"].sha256)
            self.assertEqual(pair.extra["compression_status"], "uncompressed")
            self.assertEqual(pair.extra["compression_status_source"], "deterministic_tar_derivation")
            self.assertEqual(metadata["compression_status"], "uncompressed")
            self.assertEqual(metadata["compression_status_source"], "deterministic_tar_derivation")


if __name__ == "__main__":
    unittest.main()
