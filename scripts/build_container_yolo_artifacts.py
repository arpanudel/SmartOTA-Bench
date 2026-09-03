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


DEFAULT_OUTPUT_DIR = "data/raw/containers/yolo-perception"
DEFAULT_BLOCK_SIZE_BYTES = 16384
DEFAULT_PAIR_ID = "container_yolo_perception_update"
DEFAULT_OLD_TAR_NAME = "yolo-perception-layer-old.tar"
DEFAULT_NEW_TAR_NAME = "yolo-perception-layer-new.tar"
DEFAULT_METADATA_NAME = "registration-metadata.json"
DEFAULT_FIXTURE = "small"
MULTIBLOCK_FIXTURE = "multiblock"
MULTIBLOCK_BLOCK_SIZE_BYTES = 65536
MULTIBLOCK_PAIR_ID = "container_yolo_perception_multiblock_update"
MULTIBLOCK_OLD_TAR_NAME = "yolo-perception-multiblock-layer-old.tar"
MULTIBLOCK_NEW_TAR_NAME = "yolo-perception-multiblock-layer-new.tar"
MULTIBLOCK_METADATA_NAME = "registration-metadata-multiblock.json"
MULTIBLOCK_MODEL_PATH = "app/model/yolo_mock_multiblock_weights.bin"
MULTIBLOCK_INSERT_MARKER = b"SMARTOTA_SYNTHETIC_REGION:multiblock.model.v2.inserted.telemetry_adapter\n"
MULTIBLOCK_MIN_ARTIFACT_BYTES = 1 * 1024 * 1024
MULTIBLOCK_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
FIXED_MTIME = 0


FIXTURES = {
    DEFAULT_FIXTURE: {
        "pair_id": DEFAULT_PAIR_ID,
        "old_tar_name": DEFAULT_OLD_TAR_NAME,
        "new_tar_name": DEFAULT_NEW_TAR_NAME,
        "metadata_name": DEFAULT_METADATA_NAME,
        "block_size_bytes": DEFAULT_BLOCK_SIZE_BYTES,
        "old_tree_name": "old-tree",
        "new_tree_name": "new-tree",
        "scenario": "containerized_yolo_perception_update",
        "source": "Locally generated deterministic SmartOTA-Bench YOLO-style perception container fixture",
        "license_notes": "Synthetic fixture with no third-party model weights",
    },
    MULTIBLOCK_FIXTURE: {
        "pair_id": MULTIBLOCK_PAIR_ID,
        "old_tar_name": MULTIBLOCK_OLD_TAR_NAME,
        "new_tar_name": MULTIBLOCK_NEW_TAR_NAME,
        "metadata_name": MULTIBLOCK_METADATA_NAME,
        "block_size_bytes": MULTIBLOCK_BLOCK_SIZE_BYTES,
        "old_tree_name": "multiblock-old-tree",
        "new_tree_name": "multiblock-new-tree",
        "scenario": "containerized_yolo_perception_multiblock_update",
        "source": "Locally generated deterministic synthetic fixture",
        "license_notes": "Synthetic fixture; no third-party model weights or third-party model files",
    },
}


def main(args):
    fixture_name = getattr(args, "fixture", DEFAULT_FIXTURE)
    fixture = FIXTURES[fixture_name]
    output_dir = Path(args.output_dir).expanduser().resolve()
    old_tree = output_dir / fixture["old_tree_name"]
    new_tree = output_dir / fixture["new_tree_name"]
    old_tar = output_dir / _resolved_arg(args, "old_tar_name", fixture["old_tar_name"])
    new_tar = output_dir / _resolved_arg(args, "new_tar_name", fixture["new_tar_name"])
    metadata_path = (
        Path(args.metadata).expanduser().resolve()
        if args.metadata
        else output_dir / fixture["metadata_name"]
    )
    old_files, new_files = _fixture_files(fixture_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tree(old_tree, old_files)
    _write_tree(new_tree, new_files)
    _write_tar(old_tree, old_tar)
    _write_tar(new_tree, new_tar)

    old_metadata = compute_file_metadata(old_tar)
    new_metadata = compute_file_metadata(new_tar)
    if fixture_name == MULTIBLOCK_FIXTURE:
        _validate_multiblock_artifact_size(old_metadata, "old")
        _validate_multiblock_artifact_size(new_metadata, "new")
    metadata = _metadata_payload(args, fixture, old_metadata, new_metadata)
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


def _resolved_arg(args, name, fixture_default):
    value = getattr(args, name, None)
    if value is None:
        return fixture_default
    return value


def _resolved_pair_id(args, fixture):
    value = getattr(args, "pair_id", None)
    return value if value is not None else fixture["pair_id"]


def _resolved_block_size(args, fixture):
    value = getattr(args, "block_size_bytes", None)
    return value if value is not None else fixture["block_size_bytes"]


def _fixture_files(fixture_name):
    if fixture_name == DEFAULT_FIXTURE:
        return _old_files(), _new_files()
    if fixture_name == MULTIBLOCK_FIXTURE:
        return _old_multiblock_files(), _new_multiblock_files()
    raise ValueError(f"unsupported fixture: {fixture_name}")


def _old_files():
    return {
        "app/detector.py": {
            "mode": 0o644,
            "data": (
                "MODEL_PATH = 'model/yolo_mock_weights.bin'\n"
                "THRESHOLD = 0.35\n"
                "VERSION = 'yolo-fixture-1.0.0'\n"
                "\n"
                "def detect(frame_bytes):\n"
                "    score = (sum(frame_bytes) % 100) / 100.0\n"
                "    return {'label': 'vehicle', 'score': score, 'accepted': score >= THRESHOLD}\n"
            ).encode("utf-8"),
        },
        "app/requirements.txt": {
            "mode": 0o644,
            "data": b"numpy==2.0.0\nopencv-python-headless==4.10.0.84\n",
        },
        "app/config.yaml": {
            "mode": 0o644,
            "data": (
                "model: yolo-mock\n"
                "version: 1\n"
                "threshold: 0.35\n"
                "nms_iou: 0.45\n"
            ).encode("utf-8"),
        },
        "app/model/yolo_mock_weights.bin": {
            "mode": 0o644,
            "data": _deterministic_bytes("yolo-mock-weights-v1", 8192),
        },
        "app/entrypoint.sh": {
            "mode": 0o755,
            "data": b"#!/bin/sh\nexec python /app/detector.py\n",
        },
    }


def _new_files():
    files = _old_files()
    files["app/detector.py"] = {
        "mode": 0o644,
        "data": (
            "MODEL_PATH = 'model/yolo_mock_weights.bin'\n"
            "THRESHOLD = 0.42\n"
            "VERSION = 'yolo-fixture-1.1.0'\n"
            "\n"
            "def detect(frame_bytes):\n"
            "    score = ((sum(frame_bytes) + len(frame_bytes)) % 100) / 100.0\n"
            "    return {'label': 'vehicle', 'score': score, 'accepted': score >= THRESHOLD}\n"
            "\n"
            "def model_version():\n"
            "    return VERSION\n"
        ).encode("utf-8"),
    }
    files["app/config.yaml"] = {
        "mode": 0o644,
        "data": (
            "model: yolo-mock\n"
            "version: 2\n"
            "threshold: 0.42\n"
            "nms_iou: 0.40\n"
        ).encode("utf-8"),
    }
    files["app/model/yolo_mock_weights.bin"] = {
        "mode": 0o644,
        "data": _deterministic_bytes("yolo-mock-weights-v2", 9216),
    }
    files["app/healthcheck.py"] = {
        "mode": 0o644,
        "data": (
            "from detector import model_version\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    assert model_version() == 'yolo-fixture-1.1.0'\n"
            "    print('ok')\n"
        ).encode("utf-8"),
    }
    return files


def _old_multiblock_files():
    return {
        "container/manifest.json": {
            "mode": 0o644,
            "data": _json_bytes(
                {
                    "fixture": "smartota-yolo-perception-multiblock",
                    "version": "2.3.0",
                    "format": "synthetic-container-layer-tar",
                    "notes": "Benchmark fixture only; not a runnable production image.",
                }
            ),
        },
        "app/detector.py": {
            "mode": 0o644,
            "data": (
                "MODEL_PATH = 'model/yolo_mock_multiblock_weights.bin'\n"
                "CONFIG_PATH = '/app/config.yaml'\n"
                "THRESHOLD = 0.37\n"
                "NMS_IOU = 0.46\n"
                "VERSION = 'yolo-multiblock-fixture-2.3.0'\n"
                "\n"
                "def score(frame_bytes):\n"
                "    return ((sum(frame_bytes[:4096]) + len(frame_bytes)) % 1000) / 1000.0\n"
                "\n"
                "def detect(frame_bytes):\n"
                "    confidence = score(frame_bytes)\n"
                "    return {'label': 'vehicle', 'score': confidence, 'accepted': confidence >= THRESHOLD}\n"
            ).encode("utf-8"),
        },
        "app/config.yaml": {
            "mode": 0o644,
            "data": (
                "model: yolo-mock-multiblock\n"
                "fixture_version: 2.3.0\n"
                "threshold: 0.37\n"
                "nms_iou: 0.46\n"
                "input_shape: [1, 3, 640, 640]\n"
                "staging_hint: overwrite_with_backup\n"
            ).encode("utf-8"),
        },
        "app/requirements.txt": {
            "mode": 0o644,
            "data": b"numpy==2.0.0\nopencv-python-headless==4.10.0.84\npyyaml==6.0.2\n",
        },
        "app/entrypoint.sh": {
            "mode": 0o755,
            "data": b"#!/bin/sh\nexec python /app/detector.py\n",
        },
        MULTIBLOCK_MODEL_PATH: {
            "mode": 0o644,
            "data": _multiblock_model_bytes("v1"),
        },
        "app/model/model_manifest.json": {
            "mode": 0o644,
            "data": _json_bytes(
                {
                    "model": "yolo-mock-multiblock",
                    "version": "2.3.0",
                    "synthetic": True,
                    "contains_third_party_weights": False,
                    "regions": [
                        "shared_header",
                        "shared_backbone_a",
                        "changed_detection_head",
                        "shared_neck",
                        "changed_quantization_tables",
                        "shared_embeddings",
                        "changed_postprocess_head",
                        "shared_tail",
                    ],
                }
            ),
        },
        "app/assets/camera_profile.bin": {
            "mode": 0o644,
            "data": _region_bytes("multiblock.assets.shared.camera_profile", 384 * 1024),
        },
        "app/assets/preprocess_lut.bin": {
            "mode": 0o644,
            "data": _region_bytes("multiblock.assets.shared.preprocess_lut", 256 * 1024),
        },
        "app/assets/postprocess_table.bin": {
            "mode": 0o644,
            "data": _region_bytes("multiblock.assets.postprocess_table.v1", 192 * 1024),
        },
        "app/labels/classes.txt": {
            "mode": 0o644,
            "data": (
                "vehicle\n"
                "pedestrian\n"
                "cyclist\n"
                "traffic_light\n"
            ).encode("utf-8"),
        },
        "app/docs/fixture-notes.txt": {
            "mode": 0o644,
            "data": (
                "Synthetic SmartOTA-Bench fixture.\n"
                "The mock model bytes are generated from SHA-256 labels and are not YOLO weights.\n"
                "This tar is intended to stress OTA planning, not container runtime behavior.\n"
            ).encode("utf-8"),
        },
    }


def _new_multiblock_files():
    files = _old_multiblock_files()
    files["container/manifest.json"] = {
        "mode": 0o644,
        "data": _json_bytes(
            {
                "fixture": "smartota-yolo-perception-multiblock",
                "version": "2.4.0",
                "format": "synthetic-container-layer-tar",
                "notes": "Benchmark fixture only; not a runnable production image.",
                "added": ["app/telemetry/healthcheck.py", "app/telemetry/metrics.yaml"],
            }
        ),
    }
    files["app/detector.py"] = {
        "mode": 0o644,
        "data": (
            "MODEL_PATH = 'model/yolo_mock_multiblock_weights.bin'\n"
            "CONFIG_PATH = '/app/config.yaml'\n"
            "THRESHOLD = 0.43\n"
            "NMS_IOU = 0.41\n"
            "VERSION = 'yolo-multiblock-fixture-2.4.0'\n"
            "\n"
            "def score(frame_bytes):\n"
            "    prefix = frame_bytes[:8192]\n"
            "    return ((sum(prefix) + (3 * len(frame_bytes))) % 1000) / 1000.0\n"
            "\n"
            "def detect(frame_bytes):\n"
            "    confidence = score(frame_bytes)\n"
            "    return {'label': 'vehicle', 'score': confidence, 'accepted': confidence >= THRESHOLD}\n"
            "\n"
            "def model_version():\n"
            "    return VERSION\n"
        ).encode("utf-8"),
    }
    files["app/config.yaml"] = {
        "mode": 0o644,
        "data": (
            "model: yolo-mock-multiblock\n"
            "fixture_version: 2.4.0\n"
            "threshold: 0.43\n"
            "nms_iou: 0.41\n"
            "input_shape: [1, 3, 640, 640]\n"
            "staging_hint: overwrite_with_backup\n"
            "telemetry_healthcheck: true\n"
        ).encode("utf-8"),
    }
    files[MULTIBLOCK_MODEL_PATH] = {
        "mode": 0o644,
        "data": _multiblock_model_bytes("v2"),
    }
    files["app/model/model_manifest.json"] = {
        "mode": 0o644,
        "data": _json_bytes(
            {
                "model": "yolo-mock-multiblock",
                "version": "2.4.0",
                "synthetic": True,
                "contains_third_party_weights": False,
                "regions": [
                    "shared_header",
                    "shared_backbone_a",
                    "changed_detection_head",
                    "shared_neck",
                    "inserted_telemetry_adapter",
                    "changed_quantization_tables",
                    "shared_embeddings",
                    "changed_postprocess_head",
                    "shared_tail",
                ],
            }
        ),
    }
    files["app/assets/postprocess_table.bin"] = {
        "mode": 0o644,
        "data": _region_bytes("multiblock.assets.postprocess_table.v2", 192 * 1024),
    }
    files["app/telemetry/healthcheck.py"] = {
        "mode": 0o644,
        "data": (
            "from detector import model_version\n"
            "\n"
            "def healthcheck():\n"
            "    return {'status': 'ok', 'model_version': model_version(), 'synthetic': True}\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(healthcheck())\n"
        ).encode("utf-8"),
    }
    files["app/telemetry/metrics.yaml"] = {
        "mode": 0o644,
        "data": (
            "telemetry:\n"
            "  publish_interval_ms: 250\n"
            "  counters:\n"
            "    - inference_latency_ms\n"
            "    - accepted_detection_count\n"
            "    - dropped_frame_count\n"
            "  fixture_only: true\n"
        ).encode("utf-8"),
    }
    return files


def _write_tree(root, files):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    directories = _file_parent_directories(files)
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


def _file_parent_directories(files):
    directories = set()
    for relative_path in files:
        parent = Path(relative_path).parent
        while str(parent) != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


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
                    info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
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


def _deterministic_bytes(label, size):
    output = bytearray()
    counter = 0
    seed = label.encode("utf-8")
    while len(output) < size:
        output.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(output[:size])


def _region_bytes(label, size):
    marker = f"SMARTOTA_SYNTHETIC_REGION:{label}\n".encode("utf-8")
    return (marker + _deterministic_bytes(label, size))[:size]


def _multiblock_model_bytes(version):
    if version not in {"v1", "v2"}:
        raise ValueError(f"unsupported multiblock model version: {version}")

    segments = [
        _region_bytes("multiblock.model.shared.header", 128 * 1024),
        _region_bytes("multiblock.model.shared.backbone_a", 512 * 1024),
        _region_bytes(f"multiblock.model.{version}.detection_head", 256 * 1024),
        _region_bytes("multiblock.model.shared.neck", 384 * 1024),
    ]
    if version == "v2":
        segments.append(_region_bytes("multiblock.model.v2.inserted.telemetry_adapter", 256 * 1024))
    segments.extend(
        [
            _region_bytes(f"multiblock.model.{version}.quantization_tables", 192 * 1024),
            _region_bytes("multiblock.model.shared.embeddings", 512 * 1024),
            _region_bytes(f"multiblock.model.{version}.postprocess_head", 128 * 1024),
            _region_bytes("multiblock.model.shared.tail", 384 * 1024),
        ]
    )
    return b"".join(segments)


def _json_bytes(payload):
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validate_multiblock_artifact_size(metadata, label):
    if not (MULTIBLOCK_MIN_ARTIFACT_BYTES < metadata.size_bytes < MULTIBLOCK_MAX_ARTIFACT_BYTES):
        raise ValueError(
            f"{label} multiblock artifact size {metadata.size_bytes} is outside "
            f"{MULTIBLOCK_MIN_ARTIFACT_BYTES}..{MULTIBLOCK_MAX_ARTIFACT_BYTES} bytes"
        )


def _metadata_payload(args, fixture, old_metadata, new_metadata):
    old_file = Path(os.path.relpath(old_metadata.path, PROJECT_ROOT)).as_posix()
    new_file = Path(os.path.relpath(new_metadata.path, PROJECT_ROOT)).as_posix()
    pair_id = _resolved_pair_id(args, fixture)
    block_size_bytes = _resolved_block_size(args, fixture)
    return {
        "pair_id": pair_id,
        "old_file": old_file,
        "new_file": new_file,
        "old_size_bytes": old_metadata.size_bytes,
        "new_size_bytes": new_metadata.size_bytes,
        "old_sha256": old_metadata.sha256,
        "new_sha256": new_metadata.sha256,
        "block_size_bytes": block_size_bytes,
        "domain": "container_av_perception",
        "artifact_type": "perception_container_tar",
        "compression_status": "uncompressed",
        "compression_status_source": "deterministic_tar_derivation",
        "scenario": fixture["scenario"],
        "source": fixture["source"],
        "license_notes": fixture["license_notes"],
        "registration_command": [
            "python",
            "scripts/register_dataset_artifacts.py",
            "--manifest",
            "datasets/manifests/smartota-container-layers.json",
            "--pair",
            pair_id,
            "--old-file",
            old_file,
            "--new-file",
            new_file,
            "--block-size-bytes",
            str(block_size_bytes),
            "--domain",
            "container_av_perception",
            "--scenario",
            fixture["scenario"],
            "--source",
            fixture["source"],
            "--license-notes",
            fixture["license_notes"],
            "--artifact-type",
            "perception_container_tar",
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
        description="Build deterministic YOLO-style container perception tar artifacts."
    )
    parser.add_argument(
        "--fixture",
        choices=sorted(FIXTURES),
        default=DEFAULT_FIXTURE,
        help="fixture to generate; default preserves the original small YOLO fixture",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-tar-name", default=None)
    parser.add_argument("--new-tar-name", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--block-size-bytes", type=int, default=None)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.block_size_bytes is not None and args.block_size_bytes <= 0:
        raise ValueError("--block-size-bytes must be positive")
    fixture = FIXTURES[args.fixture]
    if args.old_tar_name is None:
        args.old_tar_name = fixture["old_tar_name"]
    if args.new_tar_name is None:
        args.new_tar_name = fixture["new_tar_name"]
    if args.pair_id is None:
        args.pair_id = fixture["pair_id"]
    if args.block_size_bytes is None:
        args.block_size_bytes = fixture["block_size_bytes"]
    return args


if __name__ == "__main__":
    main(parse_args())
