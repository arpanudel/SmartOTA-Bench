import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import compute_file_metadata


DEFAULT_OUTPUT_DIR = "data/raw/automotive/autoware-style-perception"
DEFAULT_BLOCK_SIZE_BYTES = 16384
DEFAULT_PAIR_ID = "autoware_style_perception_module_update"
DEFAULT_OLD_TAR_NAME = "autoware-style-perception-old.tar"
DEFAULT_NEW_TAR_NAME = "autoware-style-perception-new.tar"
DEFAULT_METADATA_NAME = "registration-metadata.json"
FIXED_MTIME = 0


def main(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    old_tree = output_dir / "old-tree"
    new_tree = output_dir / "new-tree"
    old_tar = output_dir / args.old_tar_name
    new_tar = output_dir / args.new_tar_name
    metadata_path = (
        Path(args.metadata).expanduser().resolve()
        if args.metadata
        else output_dir / DEFAULT_METADATA_NAME
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tree(old_tree, _old_files())
    _write_tree(new_tree, _new_files())
    _write_tar(old_tree, old_tar)
    _write_tar(new_tree, new_tar)

    old_metadata = compute_file_metadata(old_tar)
    new_metadata = compute_file_metadata(new_tar)
    metadata = _metadata_payload(args, old_metadata, new_metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return {
        "old_tree": old_tree,
        "new_tree": new_tree,
        "old_tar": old_tar,
        "new_tar": new_tar,
        "metadata_path": metadata_path,
        "old_metadata": old_metadata,
        "new_metadata": new_metadata,
        "metadata": metadata,
    }


def _old_files():
    base = "autoware_stack/perception/lidar_object_detection"
    return {
        f"{base}/package.xml": {
            "mode": 0o644,
            "data": (
                "<?xml version=\"1.0\"?>\n"
                "<package format=\"3\">\n"
                "  <name>lidar_object_detection</name>\n"
                "  <version>0.1.0</version>\n"
                "  <description>Synthetic Autoware-style lidar detector fixture.</description>\n"
                "  <maintainer email=\"smartota@example.invalid\">SmartOTA Bench</maintainer>\n"
                "  <license>Synthetic-fixture</license>\n"
                "  <buildtool_depend>ament_cmake</buildtool_depend>\n"
                "  <depend>rclcpp</depend>\n"
                "  <depend>sensor_msgs</depend>\n"
                "</package>\n"
            ).encode("utf-8"),
        },
        f"{base}/CMakeLists.txt": {
            "mode": 0o644,
            "data": (
                "cmake_minimum_required(VERSION 3.16)\n"
                "project(lidar_object_detection)\n"
                "find_package(ament_cmake REQUIRED)\n"
                "find_package(rclcpp REQUIRED)\n"
                "add_library(lidar_detector src/detector_node.cpp src/postprocess.cpp)\n"
                "ament_target_dependencies(lidar_detector rclcpp)\n"
                "install(DIRECTORY config launch models DESTINATION share/${PROJECT_NAME})\n"
                "ament_package()\n"
            ).encode("utf-8"),
        },
        f"{base}/src/detector_node.cpp": {
            "mode": 0o644,
            "data": (
                "#include <algorithm>\n"
                "#include <string>\n"
                "namespace smartota_fixture {\n"
                "double score_cluster(int points, double intensity) {\n"
                "  const double density = std::min(points / 128.0, 1.0);\n"
                "  return 0.62 * density + 0.38 * intensity;\n"
                "}\n"
                "std::string detector_version() { return \"fixture-detector-0.1.0\"; }\n"
                "}  // namespace smartota_fixture\n"
            ).encode("utf-8"),
        },
        f"{base}/src/postprocess.cpp": {
            "mode": 0o644,
            "data": (
                "#include <vector>\n"
                "namespace smartota_fixture {\n"
                "std::vector<int> suppress_overlaps(const std::vector<int>& ids) {\n"
                "  std::vector<int> kept;\n"
                "  for (int id : ids) {\n"
                "    if (id % 3 != 0) kept.push_back(id);\n"
                "  }\n"
                "  return kept;\n"
                "}\n"
                "}  // namespace smartota_fixture\n"
            ).encode("utf-8"),
        },
        f"{base}/config/detector.param.yaml": {
            "mode": 0o644,
            "data": (
                "/lidar_object_detection:\n"
                "  ros__parameters:\n"
                "    model_path: models/mock_lidar_detector.onnx\n"
                "    score_threshold: 0.48\n"
                "    voxel_size: 0.25\n"
                "    max_objects: 64\n"
            ).encode("utf-8"),
        },
        f"{base}/launch/detector.launch.xml": {
            "mode": 0o644,
            "data": (
                "<launch>\n"
                "  <node pkg=\"lidar_object_detection\" exec=\"detector_node\" name=\"lidar_detector\">\n"
                "    <param from=\"$(find-pkg-share lidar_object_detection)/config/detector.param.yaml\"/>\n"
                "  </node>\n"
                "</launch>\n"
            ).encode("utf-8"),
        },
        f"{base}/models/mock_lidar_detector.onnx": {
            "mode": 0o644,
            "data": _mock_model_bytes("autoware-style-lidar-detector-v1", 24576),
        },
    }


def _new_files():
    files = _old_files()
    base = "autoware_stack/perception/lidar_object_detection"
    files[f"{base}/package.xml"] = {
        "mode": 0o644,
        "data": files[f"{base}/package.xml"]["data"].replace(b"<version>0.1.0</version>", b"<version>0.2.0</version>"),
    }
    files[f"{base}/CMakeLists.txt"] = {
        "mode": 0o644,
        "data": (
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(lidar_object_detection)\n"
            "find_package(ament_cmake REQUIRED)\n"
            "find_package(rclcpp REQUIRED)\n"
            "add_library(lidar_detector src/detector_node.cpp src/postprocess.cpp src/tracker_adapter.cpp)\n"
            "ament_target_dependencies(lidar_detector rclcpp)\n"
            "install(DIRECTORY config diagnostics launch models DESTINATION share/${PROJECT_NAME})\n"
            "ament_package()\n"
        ).encode("utf-8"),
    }
    files[f"{base}/src/detector_node.cpp"] = {
        "mode": 0o644,
        "data": (
            "#include <algorithm>\n"
            "#include <string>\n"
            "namespace smartota_fixture {\n"
            "double score_cluster(int points, double intensity) {\n"
            "  const double density = std::min(points / 96.0, 1.0);\n"
            "  const double range_bonus = points > 72 ? 0.04 : 0.0;\n"
            "  return std::min(0.68 * density + 0.32 * intensity + range_bonus, 1.0);\n"
            "}\n"
            "std::string detector_version() { return \"fixture-detector-0.2.0\"; }\n"
            "}  // namespace smartota_fixture\n"
        ).encode("utf-8"),
    }
    files[f"{base}/src/postprocess.cpp"] = {
        "mode": 0o644,
        "data": (
            "#include <vector>\n"
            "namespace smartota_fixture {\n"
            "std::vector<int> suppress_overlaps(const std::vector<int>& ids) {\n"
            "  std::vector<int> kept;\n"
            "  for (int id : ids) {\n"
            "    if (id % 5 != 0) kept.push_back(id);\n"
            "  }\n"
            "  return kept;\n"
            "}\n"
            "}  // namespace smartota_fixture\n"
        ).encode("utf-8"),
    }
    files[f"{base}/src/tracker_adapter.cpp"] = {
        "mode": 0o644,
        "data": (
            "#include <cstdint>\n"
            "namespace smartota_fixture {\n"
            "std::uint64_t stable_track_id(std::uint32_t cluster_id, std::uint32_t frame_id) {\n"
            "  return (static_cast<std::uint64_t>(frame_id) << 32) | cluster_id;\n"
            "}\n"
            "}  // namespace smartota_fixture\n"
        ).encode("utf-8"),
    }
    files[f"{base}/config/detector.param.yaml"] = {
        "mode": 0o644,
        "data": (
            "/lidar_object_detection:\n"
            "  ros__parameters:\n"
            "    model_path: models/mock_lidar_detector.onnx\n"
            "    score_threshold: 0.54\n"
            "    voxel_size: 0.20\n"
            "    max_objects: 96\n"
            "    tracker_adapter: true\n"
        ).encode("utf-8"),
    }
    files[f"{base}/models/mock_lidar_detector.onnx"] = {
        "mode": 0o644,
        "data": _mock_model_bytes("autoware-style-lidar-detector-v2", 28672),
    }
    files[f"{base}/diagnostics/health_monitor.yaml"] = {
        "mode": 0o644,
        "data": (
            "health_monitor:\n"
            "  model_sha256_required: true\n"
            "  max_inference_latency_ms: 55\n"
            "  stale_pointcloud_timeout_ms: 250\n"
        ).encode("utf-8"),
    }
    return files


def _write_tree(root, files):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    directories = {
        str(Path(relative_path).parent)
        for relative_path in files
        if str(Path(relative_path).parent) != "."
    }
    for directory in sorted(directories):
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o755)
        os.utime(path, (FIXED_MTIME, FIXED_MTIME))

    for relative_path, spec in sorted(files.items()):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spec["data"])
        os.chmod(path, spec["mode"])
        os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    os.utime(root, (FIXED_MTIME, FIXED_MTIME))


def _write_tar(root, output_tar):
    temp_tar = output_tar.with_name(f".{output_tar.name}.tmp")
    try:
        with tarfile.open(temp_tar, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in _sorted_tree_paths(root):
                relative = path.relative_to(root).as_posix()
                info = tarfile.TarInfo(relative)
                info.mtime = FIXED_MTIME
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.pax_headers = {}
                if path.is_dir():
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.size = 0
                    archive.addfile(info)
                else:
                    info.type = tarfile.REGTYPE
                    info.mode = 0o644
                    info.size = path.stat().st_size
                    with path.open("rb") as f:
                        archive.addfile(info, f)
        temp_tar.replace(output_tar)
    except Exception:
        temp_tar.unlink(missing_ok=True)
        raise


def _sorted_tree_paths(root):
    paths = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        current = Path(current_root)
        if current != root:
            paths.append(current)
        for filename in filenames:
            paths.append(current / filename)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _mock_model_bytes(label, size):
    output = bytearray(b"SMARTOTA_MOCK_ONNX\0")
    counter = 0
    seed = label.encode("utf-8")
    while len(output) < size:
        output.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(output[:size])


def _metadata_payload(args, old_metadata, new_metadata):
    old_file = Path(os.path.relpath(old_metadata.path, PROJECT_ROOT)).as_posix()
    new_file = Path(os.path.relpath(new_metadata.path, PROJECT_ROOT)).as_posix()
    return {
        "pair_id": args.pair_id,
        "old_file": old_file,
        "new_file": new_file,
        "old_size_bytes": old_metadata.size_bytes,
        "new_size_bytes": new_metadata.size_bytes,
        "old_sha256": old_metadata.sha256,
        "new_sha256": new_metadata.sha256,
        "block_size_bytes": args.block_size_bytes,
        "domain": "autoware_style_av_stack",
        "artifact_type": "autoware_module_tar",
        "compression_status": "uncompressed",
        "compression_status_source": "deterministic_tar_derivation",
        "scenario": "autoware_style_perception_module_update",
        "source": "Locally generated deterministic fixture inspired by Autoware-style module layout",
        "license_notes": (
            "Synthetic fixture; no Autoware source code, no proprietary AV software, "
            "and no third-party model weights"
        ),
        "registration_command": [
            "python",
            "scripts/register_dataset_artifacts.py",
            "--manifest",
            "datasets/manifests/smartota-automotive.json",
            "--pair",
            args.pair_id,
            "--old-file",
            old_file,
            "--new-file",
            new_file,
            "--block-size-bytes",
            str(args.block_size_bytes),
            "--domain",
            "autoware_style_av_stack",
            "--scenario",
            "autoware_style_perception_module_update",
            "--source",
            "Locally generated deterministic fixture inspired by Autoware-style module layout",
            "--license-notes",
            "Synthetic fixture; no Autoware source code, no proprietary AV software, and no third-party model weights",
            "--artifact-type",
            "autoware_module_tar",
            "--compression-status",
            "uncompressed",
            "--compression-status-source",
            "deterministic_tar_derivation",
            "--enable",
            "--write",
        ],
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build deterministic Autoware-style perception module tar artifacts."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-tar-name", default=DEFAULT_OLD_TAR_NAME)
    parser.add_argument("--new-tar-name", default=DEFAULT_NEW_TAR_NAME)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
    parser.add_argument("--block-size-bytes", type=int, default=DEFAULT_BLOCK_SIZE_BYTES)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.block_size_bytes <= 0:
        raise ValueError("--block-size-bytes must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
