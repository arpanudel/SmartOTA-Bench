import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DatasetManifestError, compute_file_metadata, load_manifest, validate_manifest_schema


def main(args):
    if args.block_size_bytes is not None and args.block_size_bytes <= 0:
        raise DatasetManifestError("--block-size-bytes must be positive")
    manifest_path = Path(args.manifest).expanduser().resolve()
    data = validate_manifest_schema(manifest_path)
    pair = _find_pair(data, args.pair)
    base_dir = _resolve_base_dir(manifest_path, data)

    old_metadata = compute_file_metadata(args.old_file)
    new_metadata = compute_file_metadata(args.new_file)
    _update_pair(pair, "old", old_metadata, base_dir)
    _update_pair(pair, "new", new_metadata, base_dir)
    _update_optional_pair_metadata(pair, args)
    if args.enable:
        pair["enabled"] = True

    rendered = json.dumps(data, indent=2) + "\n"
    _validate_rendered_schema(rendered)
    if args.write:
        manifest_path.write_text(rendered, encoding="utf-8")
        if args.enable:
            load_manifest(manifest_path)
        print(f"Updated manifest: {manifest_path}")
    else:
        print(rendered, end="")


def _find_pair(data, pair_id):
    for pair in data["pairs"]:
        if pair["id"] == pair_id:
            return pair
    raise DatasetManifestError(f"dataset pair '{pair_id}' was not found")


def _resolve_base_dir(manifest_path, data):
    base_dir = data.get("base_dir")
    if base_dir is None:
        return manifest_path.parent
    candidate = Path(base_dir).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _update_pair(pair, prefix, metadata, base_dir):
    pair[f"{prefix}_file"] = Path(os.path.relpath(metadata.path, base_dir)).as_posix()
    pair[f"{prefix}_size_bytes"] = metadata.size_bytes
    pair[f"{prefix}_sha256"] = metadata.sha256


def _update_optional_pair_metadata(pair, args):
    optional_fields = {
        "block_size_bytes": args.block_size_bytes,
        "domain": args.domain,
        "scenario": args.scenario,
        "source": args.source,
        "license_notes": args.license_notes,
        "artifact_type": args.artifact_type,
        "compression_status": args.compression_status,
        "compression_status_source": args.compression_status_source,
        "tier": args.tier,
        "old_url": args.old_url,
        "new_url": args.new_url,
    }
    for key, value in optional_fields.items():
        if value is not None:
            pair[key] = value


def _validate_rendered_schema(rendered):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            temp_path = Path(f.name)
            f.write(rendered)
        validate_manifest_schema(temp_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register local SmartOTA-Bench artifacts and update manifest metadata."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair", required=True, help="dataset pair id to update")
    parser.add_argument("--old-file", required=True)
    parser.add_argument("--new-file", required=True)
    parser.add_argument("--block-size-bytes", type=int)
    parser.add_argument("--domain")
    parser.add_argument("--scenario")
    parser.add_argument("--source")
    parser.add_argument("--license-notes")
    parser.add_argument("--artifact-type")
    parser.add_argument(
        "--compression-status",
        choices=["compressed", "uncompressed", "mixed", "unknown"],
        help="compression status for the registered artifacts",
    )
    parser.add_argument(
        "--compression-status-source",
        help="how compression status was determined or where it should be recorded",
    )
    parser.add_argument("--tier")
    parser.add_argument("--old-url")
    parser.add_argument("--new-url")
    parser.add_argument("--enable", action="store_true", help="mark the pair enabled after registration")
    parser.add_argument("--write", action="store_true", help="write changes back to the manifest")
    main(parser.parse_args())
