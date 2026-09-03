import json
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from datasets import DatasetManifestError, load_manifest, validate_manifest_schema
from scripts.build_container_yolo_artifacts import (
    MULTIBLOCK_INSERT_MARKER,
    MULTIBLOCK_MAX_ARTIFACT_BYTES,
    MULTIBLOCK_MIN_ARTIFACT_BYTES,
    MULTIBLOCK_MODEL_PATH,
    MULTIBLOCK_NEW_TAR_NAME,
    MULTIBLOCK_OLD_TAR_NAME,
    MULTIBLOCK_PAIR_ID,
    main as build_yolo_artifacts,
    parse_args as parse_build_args,
)
from scripts.register_dataset_artifacts import main as register_dataset_artifacts
from scripts.run_benchmark_suite import pair_metadata


KIB = 1024


def write_manifest(directory, payload):
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def container_manifest_payload(enabled=False):
    return {
        "name": "container-fixture-test",
        "version": 1,
        "base_dir": ".",
        "pairs": [
            {
                "id": "container_yolo_perception_update",
                "enabled": enabled,
                "domain": "container_av_perception",
                "scenario": "containerized_yolo_perception_update",
                "artifact_type": "perception_container_tar",
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
        return build_yolo_artifacts(args)


def register_quiet(args):
    with redirect_stdout(StringIO()):
        return register_dataset_artifacts(args)


def tar_member_bytes(tar_path, member_name):
    with tarfile.open(tar_path, "r:") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise AssertionError(f"tar member is not a regular file: {member_name}")
        return member.read()


class ContainerYoloArtifactTests(unittest.TestCase):
    def test_yolo_artifact_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_result = build_quiet(parse_build_args(["--output-dir", first]))
            second_result = build_quiet(parse_build_args(["--output-dir", second]))

            self.assertEqual(
                first_result["old_tar"].read_bytes(),
                second_result["old_tar"].read_bytes(),
            )
            self.assertEqual(
                first_result["new_tar"].read_bytes(),
                second_result["new_tar"].read_bytes(),
            )
            self.assertEqual(
                first_result["old_metadata"].sha256,
                second_result["old_metadata"].sha256,
            )
            self.assertEqual(
                first_result["new_metadata"].sha256,
                second_result["new_metadata"].sha256,
            )
            self.assertNotEqual(
                first_result["new_metadata"].sha256,
                first_result["old_metadata"].sha256,
            )

            with tarfile.open(first_result["new_tar"], "r:") as archive:
                members = {member.name: member for member in archive.getmembers()}
                self.assertIn("app/healthcheck.py", members)
                self.assertIn("app/model/yolo_mock_weights.bin", members)
                self.assertTrue(all(member.mtime == 0 for member in members.values()))
                self.assertTrue(all(member.uid == 0 and member.gid == 0 for member in members.values()))
                self.assertEqual(members["app/entrypoint.sh"].mode, 0o755)

    def test_multiblock_yolo_generation_is_deterministic_and_sized(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_result = build_quiet(parse_build_args(["--fixture", "multiblock", "--output-dir", first]))
            second_result = build_quiet(parse_build_args(["--fixture", "multiblock", "--output-dir", second]))

            self.assertEqual(first_result["old_tar"].name, MULTIBLOCK_OLD_TAR_NAME)
            self.assertEqual(first_result["new_tar"].name, MULTIBLOCK_NEW_TAR_NAME)
            self.assertEqual(first_result["metadata"]["pair_id"], MULTIBLOCK_PAIR_ID)
            self.assertEqual(first_result["metadata"]["block_size_bytes"], 65536)
            self.assertEqual(
                first_result["old_tar"].read_bytes(),
                second_result["old_tar"].read_bytes(),
            )
            self.assertEqual(
                first_result["new_tar"].read_bytes(),
                second_result["new_tar"].read_bytes(),
            )
            self.assertEqual(
                first_result["old_metadata"].sha256,
                second_result["old_metadata"].sha256,
            )
            self.assertEqual(
                first_result["new_metadata"].sha256,
                second_result["new_metadata"].sha256,
            )
            self.assertNotEqual(
                first_result["new_metadata"].sha256,
                first_result["old_metadata"].sha256,
            )
            self.assertGreater(first_result["old_metadata"].size_bytes, MULTIBLOCK_MIN_ARTIFACT_BYTES)
            self.assertGreater(first_result["new_metadata"].size_bytes, MULTIBLOCK_MIN_ARTIFACT_BYTES)
            self.assertLess(first_result["old_metadata"].size_bytes, MULTIBLOCK_MAX_ARTIFACT_BYTES)
            self.assertLess(first_result["new_metadata"].size_bytes, MULTIBLOCK_MAX_ARTIFACT_BYTES)

    def test_multiblock_yolo_fixture_has_controlled_similarity_and_insertions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build_quiet(parse_build_args(["--fixture", "multiblock", "--output-dir", tmp]))

            with tarfile.open(result["old_tar"], "r:") as old_archive:
                old_members = {member.name: member for member in old_archive.getmembers()}
            with tarfile.open(result["new_tar"], "r:") as new_archive:
                new_members = {member.name: member for member in new_archive.getmembers()}

            self.assertIn("app/telemetry/healthcheck.py", new_members)
            self.assertIn("app/telemetry/metrics.yaml", new_members)
            self.assertNotIn("app/telemetry/healthcheck.py", old_members)
            self.assertTrue(all(member.mtime == 0 for member in new_members.values()))
            self.assertTrue(all(member.uid == 0 and member.gid == 0 for member in new_members.values()))
            self.assertEqual(new_members["app/entrypoint.sh"].mode, 0o755)

            self.assertEqual(
                tar_member_bytes(result["old_tar"], "app/assets/camera_profile.bin"),
                tar_member_bytes(result["new_tar"], "app/assets/camera_profile.bin"),
            )
            self.assertEqual(
                tar_member_bytes(result["old_tar"], "app/assets/preprocess_lut.bin"),
                tar_member_bytes(result["new_tar"], "app/assets/preprocess_lut.bin"),
            )
            self.assertNotEqual(
                tar_member_bytes(result["old_tar"], "app/assets/postprocess_table.bin"),
                tar_member_bytes(result["new_tar"], "app/assets/postprocess_table.bin"),
            )
            self.assertNotEqual(
                tar_member_bytes(result["old_tar"], "app/config.yaml"),
                tar_member_bytes(result["new_tar"], "app/config.yaml"),
            )
            self.assertNotEqual(
                tar_member_bytes(result["old_tar"], "app/detector.py"),
                tar_member_bytes(result["new_tar"], "app/detector.py"),
            )

            old_model = tar_member_bytes(result["old_tar"], MULTIBLOCK_MODEL_PATH)
            new_model = tar_member_bytes(result["new_tar"], MULTIBLOCK_MODEL_PATH)
            self.assertGreater(len(new_model), len(old_model))
            self.assertIn(MULTIBLOCK_INSERT_MARKER, new_model)
            self.assertNotIn(MULTIBLOCK_INSERT_MARKER, old_model)
            self.assertEqual(old_model[0 : 128 * KIB], new_model[0 : 128 * KIB])
            self.assertEqual(old_model[128 * KIB : 640 * KIB], new_model[128 * KIB : 640 * KIB])
            self.assertNotEqual(old_model[640 * KIB : 896 * KIB], new_model[640 * KIB : 896 * KIB])
            self.assertEqual(old_model[896 * KIB : 1280 * KIB], new_model[896 * KIB : 1280 * KIB])
            self.assertEqual(old_model[1472 * KIB : 1984 * KIB], new_model[1728 * KIB : 2240 * KIB])

    def test_container_manifest_declares_enabled_multiblock_yolo_pair(self):
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "datasets" / "manifests" / "smartota-container-layers.json"
        data = validate_manifest_schema(manifest_path)
        pairs = {pair["id"]: pair for pair in data["pairs"]}

        self.assertIn("container_yolo_perception_update", pairs)
        self.assertIn(MULTIBLOCK_PAIR_ID, pairs)
        pair = pairs[MULTIBLOCK_PAIR_ID]
        self.assertTrue(pair["enabled"])
        self.assertEqual(pair["domain"], "container_av_perception")
        self.assertEqual(pair["artifact_type"], "perception_container_tar")
        self.assertEqual(pair["compression_status"], "uncompressed")
        self.assertEqual(pair["compression_status_source"], "deterministic_tar_derivation")
        self.assertEqual(pair["scenario"], "containerized_yolo_perception_multiblock_update")
        self.assertEqual(pair["source"], "Locally generated deterministic synthetic fixture")
        self.assertIn("no third-party model weights", pair["license_notes"])
        self.assertGreater(pair["old_size_bytes"], MULTIBLOCK_MIN_ARTIFACT_BYTES)
        self.assertGreater(pair["new_size_bytes"], MULTIBLOCK_MIN_ARTIFACT_BYTES)
        self.assertLess(pair["old_size_bytes"], MULTIBLOCK_MAX_ARTIFACT_BYTES)
        self.assertLess(pair["new_size_bytes"], MULTIBLOCK_MAX_ARTIFACT_BYTES)

    def test_generated_container_yolo_artifacts_are_git_ignored(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / ".git").exists():
            self.skipTest("git metadata is not available")

        ignored_path = "data/raw/containers/yolo-perception/yolo-perception-multiblock-layer-old.tar"
        result = subprocess.run(
            ["git", "check-ignore", ignored_path],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), ignored_path)

    def test_enabled_container_pair_missing_artifacts_reports_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(tmp, container_manifest_payload(enabled=True))

            validate_manifest_schema(manifest_path)
            with self.assertRaisesRegex(DatasetManifestError, "container_yolo_perception_update.*does not exist"):
                load_manifest(manifest_path)

    def test_registered_yolo_pair_validates_and_propagates_compression_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_dir = tmp_path / "artifacts"
            build_result = build_quiet(parse_build_args(["--output-dir", str(artifact_dir)]))
            manifest_path = write_manifest(tmp_path, container_manifest_payload(enabled=False))

            register_quiet(
                SimpleNamespace(
                    manifest=str(manifest_path),
                    pair="container_yolo_perception_update",
                    old_file=str(build_result["old_tar"]),
                    new_file=str(build_result["new_tar"]),
                    block_size_bytes=16384,
                    domain="container_av_perception",
                    scenario="containerized_yolo_perception_update",
                    source="unit test deterministic fixture",
                    license_notes="synthetic fixture",
                    artifact_type="perception_container_tar",
                    compression_status="uncompressed",
                    compression_status_source="deterministic_tar_derivation",
                    tier="containers",
                    old_url=None,
                    new_url=None,
                    enable=True,
                    write=True,
                )
            )

            manifest = load_manifest(manifest_path)
            pair = manifest.get_pair("container_yolo_perception_update")
            metadata = pair_metadata(manifest, pair, pair.block_size_bytes)

            self.assertEqual(pair.old_sha256, build_result["old_metadata"].sha256)
            self.assertEqual(pair.new_sha256, build_result["new_metadata"].sha256)
            self.assertEqual(pair.extra["compression_status"], "uncompressed")
            self.assertEqual(pair.extra["compression_status_source"], "deterministic_tar_derivation")
            self.assertEqual(metadata["compression_status"], "uncompressed")
            self.assertEqual(metadata["compression_status_source"], "deterministic_tar_derivation")


if __name__ == "__main__":
    unittest.main()
