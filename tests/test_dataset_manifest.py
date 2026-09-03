import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from datasets import (
    DatasetManifestError,
    load_dataset_pairs,
    load_manifest,
    validate_manifest_schema,
)
from scripts.register_dataset_artifacts import main as register_dataset_artifacts


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


def _all_pair_artifacts_exist(manifest_path, data, pairs):
    base_dir = Path(data.get("base_dir", ".")).expanduser()
    if not base_dir.is_absolute():
        base_dir = Path(manifest_path).parent / base_dir
    base_dir = base_dir.resolve()
    for pair in pairs:
        for field_name in ("old_file", "new_file"):
            artifact_path = Path(pair[field_name]).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = base_dir / artifact_path
            if not artifact_path.resolve().is_file():
                return False
    return True


class DatasetManifestTests(unittest.TestCase):
    def test_load_manifest_computes_pair_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_bytes = b"old-image"
            new_bytes = b"new-image-v2"
            old_file = write_file(tmp, "old.bin", old_bytes)
            new_file = write_file(tmp, "new.bin", new_bytes)
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "test_manifest",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "smoke_pair",
                            "domain": "smoke",
                            "scenario": "tiny synthetic update",
                            "old_file": old_file.name,
                            "new_file": new_file.name,
                            "block_size_bytes": 4,
                            "source": "unit test fixture",
                            "license_notes": "test only",
                            "tags": ["unit", "synthetic"],
                        }
                    ],
                },
            )

            manifest = load_manifest(manifest_path)
            pair = manifest.get_pair("smoke_pair")
            records = manifest.to_records()

            self.assertEqual(manifest.name, "test_manifest")
            self.assertEqual(pair.id, "smoke_pair")
            self.assertEqual(pair.domain, "smoke")
            self.assertEqual(pair.scenario, "tiny synthetic update")
            self.assertEqual(pair.block_size_bytes, 4)
            self.assertEqual(pair.old_path, old_file.resolve())
            self.assertEqual(pair.new_path, new_file.resolve())
            self.assertEqual(pair.old_size_bytes, len(old_bytes))
            self.assertEqual(pair.new_size_bytes, len(new_bytes))
            self.assertEqual(pair.old_sha256, sha256(old_bytes))
            self.assertEqual(pair.new_sha256, sha256(new_bytes))
            self.assertEqual(pair.extra["tags"], ["unit", "synthetic"])
            self.assertEqual(records[0]["tags"], ["unit", "synthetic"])

    def test_declared_metadata_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_file(tmp, "old.bin", b"old")
            new_file = write_file(tmp, "new.bin", b"new")
            manifest_path = write_manifest(
                tmp,
                {
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "bad_hash",
                            "domain": "smoke",
                            "scenario": "bad declared hash",
                            "old_file": old_file.name,
                            "new_file": new_file.name,
                            "old_sha256": "0" * 64,
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(DatasetManifestError, "old_sha256 mismatch"):
                load_manifest(manifest_path)

    def test_disabled_pairs_are_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(
                tmp,
                {
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "planned_pair",
                            "enabled": False,
                            "domain": "autonomous_vehicle",
                            "scenario": "future local artifact",
                            "old_file": "missing-old.bin",
                            "new_file": "missing-new.bin",
                            "block_size_bytes": 65536,
                        }
                    ],
                },
            )

            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.pairs, ())

            with self.assertRaisesRegex(DatasetManifestError, "does not exist"):
                load_manifest(manifest_path, include_disabled=True)

    def test_load_dataset_pairs_filters_by_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_file(tmp, "old-a.bin", b"old-a")
            write_file(tmp, "new-a.bin", b"new-a")
            write_file(tmp, "old-b.bin", b"old-b")
            write_file(tmp, "new-b.bin", b"new-b")
            manifest_path = write_manifest(
                tmp,
                {
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "pair_a",
                            "domain": "smoke",
                            "scenario": "first pair",
                            "old_file": "old-a.bin",
                            "new_file": "new-a.bin",
                            "block_size_bytes": 4,
                        },
                        {
                            "id": "pair_b",
                            "domain": "edge_iot",
                            "scenario": "second pair",
                            "artifact_type": "disk_image",
                            "compression_status_source": "unit test fixture",
                            "old_file": "old-b.bin",
                            "new_file": "new-b.bin",
                            "old_size_bytes": len(b"old-b"),
                            "new_size_bytes": len(b"new-b"),
                            "old_sha256": sha256(b"old-b"),
                            "new_sha256": sha256(b"new-b"),
                            "block_size_bytes": 4,
                            "source": "unit test source",
                            "license_notes": "unit test license",
                        },
                    ],
                },
            )

            pairs = load_dataset_pairs(manifest_path)
            manifest = load_manifest(manifest_path)

            self.assertEqual([pair.id for pair in pairs], ["pair_a", "pair_b"])
            self.assertEqual([pair.id for pair in manifest.by_domain("smoke")], ["pair_a"])

    def test_schema_validation_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(
                tmp,
                {
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "duplicate",
                            "domain": "smoke",
                            "scenario": "first pair",
                            "old_file": "old-a.bin",
                            "new_file": "new-a.bin",
                            "block_size_bytes": 4,
                        },
                        {
                            "id": "duplicate",
                            "domain": "smoke",
                            "scenario": "second pair",
                            "old_file": "old-b.bin",
                            "new_file": "new-b.bin",
                            "block_size_bytes": 4,
                        },
                    ],
                },
            )

            with self.assertRaisesRegex(DatasetManifestError, "duplicate dataset pair id"):
                validate_manifest_schema(manifest_path)

    def test_schema_validation_rejects_bad_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(
                tmp,
                {
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "bad_sha",
                            "domain": "smoke",
                            "scenario": "bad sha",
                            "old_file": "old.bin",
                            "new_file": "new.bin",
                            "old_sha256": "not-a-sha",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(DatasetManifestError, "64-character SHA-256"):
                validate_manifest_schema(manifest_path)

    def test_enabled_real_pair_requires_reproducibility_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_file(tmp, "old.tar", b"old")
            new_file = write_file(tmp, "new.tar", b"new")
            manifest_path = write_manifest(
                tmp,
                {
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "real_pair",
                            "enabled": True,
                            "domain": "os_rootfs",
                            "scenario": "real artifact missing reproducibility fields",
                            "old_file": old_file.name,
                            "new_file": new_file.name,
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(DatasetManifestError, "enabled real pair 'real_pair'"):
                validate_manifest_schema(manifest_path)

    def test_enabled_real_pair_missing_artifact_reports_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(
                tmp,
                {
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "missing_artifact",
                            "enabled": True,
                            "domain": "os_rootfs",
                            "scenario": "real artifact path missing locally",
                            "artifact_type": "rootfs_tar",
                            "compression_status": "uncompressed",
                            "old_file": "missing-old.tar",
                            "new_file": "missing-new.tar",
                            "old_size_bytes": 1,
                            "new_size_bytes": 1,
                            "old_sha256": "0" * 64,
                            "new_sha256": "1" * 64,
                            "block_size_bytes": 4,
                            "source": "unit test provenance",
                            "license_notes": "unit test license",
                        }
                    ],
                },
            )

            validate_manifest_schema(manifest_path)
            with self.assertRaisesRegex(
                DatasetManifestError,
                "missing_artifact.*old_file.*does not exist",
            ):
                load_manifest(manifest_path)

    def test_example_manifest_schema_loads_without_local_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "datasets" / "manifest.example.json"

        data = validate_manifest_schema(path)
        manifest = load_manifest(path)

        self.assertEqual(data["name"], "ota_environment_dataset_manifest")
        self.assertEqual(manifest.pairs, ())

    def test_smartota_manifest_plans_validate_with_optional_local_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        manifest_dir = root / "datasets" / "manifests"
        expected_names = {
            "smartota-smoke",
            "smartota-alpine-expanded",
            "smartota-alpine-rootfs",
            "smartota-debian-ubuntu-updates",
            "smartota-container-layers",
            "smartota-synthetic-adversarial",
            "smartota-linux-images",
            "smartota-automotive",
            "smartota-av-software",
        }

        found_names = set()
        for path in sorted(manifest_dir.glob("smartota-*.json")):
            data = validate_manifest_schema(path)
            found_names.add(data["name"])
            self.assertTrue(data["pairs"])
            enabled_pairs = [pair for pair in data["pairs"] if pair.get("enabled", True)]
            if not enabled_pairs:
                manifest = load_manifest(path)
                self.assertEqual(manifest.pairs, ())
                continue

            if _all_pair_artifacts_exist(path, data, enabled_pairs):
                manifest = load_manifest(path)
                self.assertEqual(
                    {pair.id for pair in manifest.pairs},
                    {pair["id"] for pair in enabled_pairs},
                )
            else:
                with self.assertRaisesRegex(DatasetManifestError, "does not exist"):
                    load_manifest(path)

        self.assertEqual(found_names, expected_names)

    def test_register_dataset_artifacts_updates_metadata_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_file(tmp, "old.bin", b"old-rootfs")
            new_file = write_file(tmp, "new.bin", b"new-rootfs")
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "register_test",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "rootfs_pair",
                            "enabled": False,
                            "domain": "placeholder",
                            "scenario": "placeholder",
                            "old_file": "missing-old.bin",
                            "new_file": "missing-new.bin",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )

            register_dataset_artifacts(
                SimpleNamespace(
                    manifest=str(manifest_path),
                    pair="rootfs_pair",
                    old_file=str(old_file),
                    new_file=str(new_file),
                    block_size_bytes=8,
                    domain="os_rootfs",
                    scenario="registered rootfs update",
                    source="unit test source",
                    license_notes="unit test license",
                    artifact_type="rootfs_tar",
                    compression_status="uncompressed",
                    compression_status_source="unit_test_declared",
                    tier="unit",
                    old_url="https://example.invalid/old.tar",
                    new_url="https://example.invalid/new.tar",
                    enable=True,
                    write=True,
                )
            )

            manifest = load_manifest(manifest_path)
            pair = manifest.get_pair("rootfs_pair")

            self.assertEqual(pair.domain, "os_rootfs")
            self.assertEqual(pair.scenario, "registered rootfs update")
            self.assertEqual(pair.block_size_bytes, 8)
            self.assertEqual(pair.old_sha256, sha256(b"old-rootfs"))
            self.assertEqual(pair.new_sha256, sha256(b"new-rootfs"))
            self.assertEqual(pair.source, "unit test source")
            self.assertEqual(pair.license_notes, "unit test license")
            self.assertEqual(pair.extra["artifact_type"], "rootfs_tar")
            self.assertEqual(pair.extra["compression_status"], "uncompressed")
            self.assertEqual(pair.extra["compression_status_source"], "unit_test_declared")
            self.assertEqual(pair.extra["tier"], "unit")
            self.assertEqual(pair.extra["old_url"], "https://example.invalid/old.tar")

    def test_register_dataset_artifacts_rejects_incomplete_enabled_real_pair_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_file = write_file(tmp, "old.bin", b"old-rootfs")
            new_file = write_file(tmp, "new.bin", b"new-rootfs")
            manifest_path = write_manifest(
                tmp,
                {
                    "name": "register_test",
                    "version": 1,
                    "base_dir": ".",
                    "pairs": [
                        {
                            "id": "rootfs_pair",
                            "enabled": False,
                            "domain": "placeholder",
                            "scenario": "placeholder",
                            "old_file": "missing-old.bin",
                            "new_file": "missing-new.bin",
                            "block_size_bytes": 4,
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(DatasetManifestError, "compression_status"):
                register_dataset_artifacts(
                    SimpleNamespace(
                        manifest=str(manifest_path),
                        pair="rootfs_pair",
                        old_file=str(old_file),
                        new_file=str(new_file),
                        block_size_bytes=8,
                        domain="os_rootfs",
                        scenario="registered rootfs update",
                        source="unit test source",
                        license_notes="unit test license",
                        artifact_type="rootfs_tar",
                        compression_status=None,
                        compression_status_source=None,
                        tier="unit",
                        old_url=None,
                        new_url=None,
                        enable=True,
                        write=True,
                    )
                )

            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(data["pairs"][0]["enabled"])


if __name__ == "__main__":
    unittest.main()
