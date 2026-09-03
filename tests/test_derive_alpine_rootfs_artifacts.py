import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from datasets import DatasetManifestError, load_manifest
from scripts.derive_alpine_rootfs_artifacts import main, parse_args


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(directory, payload):
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_tar_gz(path, entries):
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            info = tarfile.TarInfo(entry["name"])
            info.mtime = entry.get("mtime", 123)
            info.mode = entry.get("mode", 0o644)
            info.uid = entry.get("uid", 1000)
            info.gid = entry.get("gid", 1000)
            if entry.get("type") == "dir":
                info.type = tarfile.DIRTYPE
                info.mode = entry.get("mode", 0o755)
                archive.addfile(info)
            elif entry.get("type") == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = entry["linkname"]
                archive.addfile(info)
            else:
                data = entry["data"]
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return path


def source_entries(version, *, reversed_order=False):
    entries = [
        {"name": "./etc", "type": "dir", "mtime": 321},
        {
            "name": "./etc/os-release",
            "data": f"VERSION_ID={version}\n".encode("ascii"),
            "mtime": 321,
        },
        {"name": "./bin/sh", "type": "symlink", "linkname": "/bin/busybox", "mtime": 321},
    ]
    if reversed_order:
        entries.reverse()
        for entry in entries:
            entry["mtime"] = 999
    return entries


def source_manifest_payload(old_path, new_path, old_bytes_sha, new_bytes_sha):
    return {
        "name": "unit-alpine-source",
        "version": 1,
        "base_dir": ".",
        "pairs": [
            {
                "id": "alpine_source_pair",
                "enabled": False,
                "domain": "os_rootfs",
                "scenario": "unit Alpine minirootfs transition",
                "old_file": old_path.name,
                "new_file": new_path.name,
                "old_size_bytes": old_path.stat().st_size,
                "new_size_bytes": new_path.stat().st_size,
                "old_sha256": old_bytes_sha,
                "new_sha256": new_bytes_sha,
                "block_size_bytes": 512,
                "source": "unit test",
                "license_notes": "test only",
            }
        ],
    }


class AlpineRootfsDerivationTests(unittest.TestCase):
    def test_derivation_writes_deterministic_artifacts_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            source_dir.mkdir()
            old_tar = write_tar_gz(
                source_dir / "alpine-old.tar.gz",
                source_entries("3.20"),
            )
            new_tar = write_tar_gz(
                source_dir / "alpine-new.tar.gz",
                source_entries("3.21", reversed_order=True),
            )
            manifest_path = write_manifest(
                source_dir,
                source_manifest_payload(
                    old_tar,
                    new_tar,
                    file_sha256(old_tar),
                    file_sha256(new_tar),
                ),
            )

            result_a = main(
                parse_args(
                    [
                        "--source-manifest",
                        str(manifest_path),
                        "--pair",
                        "alpine_source_pair",
                        "--output-dir",
                        str(tmp_path / "out-a"),
                    ]
                )
            )
            result_b = main(
                parse_args(
                    [
                        "--source-manifest",
                        str(manifest_path),
                        "--pair",
                        "alpine_source_pair",
                        "--output-dir",
                        str(tmp_path / "out-b"),
                    ]
                )
            )

            self.assertEqual(
                result_a["old_metadata"].sha256,
                result_b["old_metadata"].sha256,
            )
            self.assertEqual(
                result_a["new_metadata"].sha256,
                result_b["new_metadata"].sha256,
            )
            self.assertEqual(
                result_a["old_output"].read_bytes(),
                result_b["old_output"].read_bytes(),
            )
            self.assertEqual(
                result_a["new_output"].read_bytes(),
                result_b["new_output"].read_bytes(),
            )
            self.assertEqual(
                result_a["output_manifest"].read_text(encoding="utf-8"),
                result_b["output_manifest"].read_text(encoding="utf-8"),
            )

            manifest = load_manifest(result_a["output_manifest"])
            pair = manifest.get_pair("alpine_source_pair_rootfs_tar")
            self.assertEqual(pair.extra["artifact_type"], "rootfs_tar")
            self.assertEqual(pair.extra["compression_status"], "uncompressed")
            self.assertEqual(
                pair.extra["compression_status_source"],
                "derivation_normalized_uncompressed_tar",
            )
            self.assertEqual(pair.extra["derived_from_pair_id"], "alpine_source_pair")
            self.assertEqual(pair.block_size_bytes, 512)
            self.assertEqual(pair.old_sha256, result_a["old_metadata"].sha256)
            self.assertEqual(pair.new_sha256, result_a["new_metadata"].sha256)

            with tarfile.open(result_a["old_output"], "r:") as archive:
                members = archive.getmembers()
                self.assertTrue(all(member.mtime == 0 for member in members))
                self.assertIn("etc/os-release", {member.name for member in members})
                self.assertNotIn("./etc/os-release", {member.name for member in members})
                self.assertEqual(
                    archive.extractfile("etc/os-release").read(),
                    b"VERSION_ID=3.20\n",
                )

    def test_derivation_is_independent_of_source_tar_order_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_source = tmp_path / "source-a"
            second_source = tmp_path / "source-b"
            first_source.mkdir()
            second_source.mkdir()
            first_old_tar = write_tar_gz(
                first_source / "alpine-old.tar.gz",
                source_entries("3.20"),
            )
            first_new_tar = write_tar_gz(
                first_source / "alpine-new.tar.gz",
                source_entries("3.21"),
            )
            second_old_tar = write_tar_gz(
                second_source / "alpine-old.tar.gz",
                source_entries("3.20", reversed_order=True),
            )
            second_new_tar = write_tar_gz(
                second_source / "alpine-new.tar.gz",
                source_entries("3.21", reversed_order=True),
            )
            first_manifest = write_manifest(
                first_source,
                source_manifest_payload(
                    first_old_tar,
                    first_new_tar,
                    file_sha256(first_old_tar),
                    file_sha256(first_new_tar),
                ),
            )
            second_manifest = write_manifest(
                second_source,
                source_manifest_payload(
                    second_old_tar,
                    second_new_tar,
                    file_sha256(second_old_tar),
                    file_sha256(second_new_tar),
                ),
            )

            first_result = main(
                parse_args(
                    [
                        "--source-manifest",
                        str(first_manifest),
                        "--pair",
                        "alpine_source_pair",
                        "--output-dir",
                        str(tmp_path / "out-a"),
                    ]
                )
            )
            second_result = main(
                parse_args(
                    [
                        "--source-manifest",
                        str(second_manifest),
                        "--pair",
                        "alpine_source_pair",
                        "--output-dir",
                        str(tmp_path / "out-b"),
                    ]
                )
            )

            self.assertEqual(
                first_result["old_output"].read_bytes(),
                second_result["old_output"].read_bytes(),
            )
            self.assertEqual(
                first_result["new_output"].read_bytes(),
                second_result["new_output"].read_bytes(),
            )

    def test_derivation_rejects_unsafe_tar_member_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            source_dir.mkdir()
            unsafe_tar = write_tar_gz(
                source_dir / "unsafe-old.tar.gz",
                [{"name": "../escape", "data": b"bad"}],
            )
            new_tar = write_tar_gz(
                source_dir / "safe-new.tar.gz",
                [{"name": "./etc/os-release", "data": b"VERSION_ID=3.21\n"}],
            )
            manifest_path = write_manifest(
                source_dir,
                source_manifest_payload(
                    unsafe_tar,
                    new_tar,
                    file_sha256(unsafe_tar),
                    file_sha256(new_tar),
                ),
            )

            with self.assertRaisesRegex(DatasetManifestError, "unsafe tar member path"):
                main(
                    parse_args(
                        [
                            "--source-manifest",
                            str(manifest_path),
                            "--pair",
                            "alpine_source_pair",
                            "--output-dir",
                            str(tmp_path / "out"),
                        ]
                    )
                )


if __name__ == "__main__":
    unittest.main()
