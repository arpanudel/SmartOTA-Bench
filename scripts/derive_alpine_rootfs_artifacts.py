import argparse
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import (
    DatasetManifestError,
    compute_file_metadata,
    load_manifest,
    validate_file_metadata,
    validate_manifest_schema,
)


DEFAULT_SOURCE_MANIFEST = "datasets/manifests/smartota-alpine-expanded.json"
DEFAULT_SOURCE_PAIR = "alpine_3_20_to_3_21_minirootfs_x86_64"
DEFAULT_OUTPUT_DIR = "data/processed/alpine-rootfs"
DEFAULT_MANIFEST_NAME = "smartota-alpine-rootfs"


def main(args):
    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_manifest = validate_manifest_schema(source_manifest_path)
    source_pair = _find_pair(source_manifest, args.pair)
    source_base_dir = _resolve_base_dir(source_manifest_path, source_manifest)

    old_source = _resolve_existing_file(
        source_base_dir,
        source_pair["old_file"],
        source_pair["id"],
        "old_file",
    )
    new_source = _resolve_existing_file(
        source_base_dir,
        source_pair["new_file"],
        source_pair["id"],
        "new_file",
    )
    _validate_source_metadata(source_pair, "old", old_source)
    _validate_source_metadata(source_pair, "new", new_source)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    old_output = output_dir / f"{_artifact_stem(old_source)}.rootfs.tar"
    new_output = output_dir / f"{_artifact_stem(new_source)}.rootfs.tar"

    derive_rootfs_tar(old_source, old_output)
    derive_rootfs_tar(new_source, new_output)

    old_metadata = compute_file_metadata(old_output)
    new_metadata = compute_file_metadata(new_output)
    output_manifest = (
        Path(args.output_manifest).expanduser().resolve()
        if args.output_manifest
        else output_dir / f"{DEFAULT_MANIFEST_NAME}.json"
    )
    pair_id = args.output_pair_id or f"{source_pair['id']}_rootfs_tar"
    rendered = _render_output_manifest(
        output_manifest=output_manifest,
        output_dir=output_dir,
        source_manifest_path=source_manifest_path,
        source_pair=source_pair,
        pair_id=pair_id,
        old_metadata=old_metadata,
        new_metadata=new_metadata,
        manifest_name=args.output_manifest_name,
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(rendered, encoding="utf-8")
    load_manifest(output_manifest)
    print(f"Wrote rootfs old artifact: {old_output}")
    print(f"Wrote rootfs new artifact: {new_output}")
    print(f"Wrote enabled rootfs manifest: {output_manifest}")
    return {
        "old_output": old_output,
        "new_output": new_output,
        "output_manifest": output_manifest,
        "old_metadata": old_metadata,
        "new_metadata": new_metadata,
    }


def derive_rootfs_tar(source_tarball, output_tar):
    """Write a normalized uncompressed tar from a source Alpine minirootfs tarball."""
    source_tarball = Path(source_tarball).expanduser().resolve()
    output_tar = Path(output_tar).expanduser().resolve()
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_tar.with_name(f".{output_tar.name}.tmp")
    try:
        with tarfile.open(source_tarball, "r:*") as source:
            members = _validated_members(source)
            with tarfile.open(temp_output, "w", format=tarfile.PAX_FORMAT) as target:
                for normalized_name, member in members:
                    normalized_member = _canonical_member(member, normalized_name)
                    if member.isfile():
                        source_file = source.extractfile(member)
                        if source_file is None:
                            raise DatasetManifestError(
                                f"could not read regular file member '{member.name}' from {source_tarball}"
                            )
                        target.addfile(normalized_member, source_file)
                    else:
                        target.addfile(normalized_member)
        temp_output.replace(output_tar)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise


def _validated_members(source):
    members = []
    for member in source:
        normalized_name = _normalize_member_name(member.name)
        if normalized_name is None:
            continue
        members.append((normalized_name, member))
    return sorted(
        members,
        key=lambda item: (_path_sort_key(item[0]), _member_type_rank(item[1])),
    )


def _path_sort_key(name):
    return tuple(name.split("/"))


def _member_type_rank(member):
    if member.isdir():
        return 0
    if member.isfile():
        return 1
    if member.islnk():
        return 2
    if member.issym():
        return 3
    return 4


def _find_pair(manifest, pair_id):
    for pair in manifest["pairs"]:
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


def _resolve_existing_file(base_dir, value, pair_id, field_name):
    if not isinstance(value, str) or not value:
        raise DatasetManifestError(
            f"pair '{pair_id}' field '{field_name}' must be a non-empty path string"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise DatasetManifestError(
            f"pair '{pair_id}' field '{field_name}' does not exist: {path}"
        )
    if not path.is_file():
        raise DatasetManifestError(
            f"pair '{pair_id}' field '{field_name}' is not a file: {path}"
        )
    return path


def _validate_source_metadata(pair, prefix, path):
    validate_file_metadata(
        path,
        expected_size_bytes=pair.get(f"{prefix}_size_bytes"),
        expected_sha256=pair.get(f"{prefix}_sha256"),
        label=f"pair '{pair['id']}' source {prefix} artifact",
    )


def _artifact_stem(path):
    name = Path(path).name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _normalize_member_name(name):
    if not isinstance(name, str) or not name:
        raise DatasetManifestError("tar member path must be a non-empty string")
    if "\x00" in name or "\\" in name:
        raise DatasetManifestError(f"unsafe tar member path: {name!r}")

    path = PurePosixPath(name)
    if path.is_absolute():
        raise DatasetManifestError(f"unsafe tar member path: {name!r}")

    parts = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise DatasetManifestError(f"unsafe tar member path: {name!r}")
        parts.append(part)

    if not parts:
        return None
    return "/".join(parts)


def _normalize_hardlink_name(name, member_name):
    if not name:
        raise DatasetManifestError(f"hardlink member '{member_name}' is missing a link target")
    normalized = _normalize_member_name(name)
    if normalized is None:
        raise DatasetManifestError(f"hardlink member '{member_name}' points at archive root")
    return normalized


def _canonical_member(member, normalized_name):
    if not (
        member.isfile()
        or member.isdir()
        or member.issym()
        or member.islnk()
        or member.isfifo()
        or member.ischr()
        or member.isblk()
    ):
        raise DatasetManifestError(f"unsupported tar member type for '{member.name}': {member.type!r}")

    info = tarfile.TarInfo(normalized_name)
    info.type = member.type
    info.mode = member.mode & 0o7777
    info.uid = int(member.uid)
    info.gid = int(member.gid)
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}

    if member.isfile():
        info.size = member.size
    else:
        info.size = 0

    if member.issym():
        if "\x00" in member.linkname:
            raise DatasetManifestError(f"unsafe symlink target for '{member.name}'")
        info.linkname = member.linkname
    elif member.islnk():
        info.linkname = _normalize_hardlink_name(member.linkname, member.name)

    if member.ischr() or member.isblk():
        info.devmajor = int(member.devmajor)
        info.devminor = int(member.devminor)

    return info


def _render_output_manifest(
    *,
    output_manifest,
    output_dir,
    source_manifest_path,
    source_pair,
    pair_id,
    old_metadata,
    new_metadata,
    manifest_name,
):
    base_dir = Path(os.path.relpath(output_dir, output_manifest.parent)).as_posix()
    pair = {
        "id": pair_id,
        "enabled": True,
        "domain": source_pair["domain"],
        "tier": "alpine-rootfs",
        "scenario": f"{source_pair['scenario']} derived as normalized uncompressed rootfs tar artifacts",
        "artifact_type": "rootfs_tar",
        "compression_status": "uncompressed",
        "compression_status_source": "derivation_normalized_uncompressed_tar",
        "old_file": Path(os.path.relpath(old_metadata.path, output_dir)).as_posix(),
        "new_file": Path(os.path.relpath(new_metadata.path, output_dir)).as_posix(),
        "old_size_bytes": old_metadata.size_bytes,
        "new_size_bytes": new_metadata.size_bytes,
        "old_sha256": old_metadata.sha256,
        "new_sha256": new_metadata.sha256,
        "block_size_bytes": source_pair["block_size_bytes"],
        "source": source_pair.get("source", ""),
        "license_notes": source_pair.get("license_notes", ""),
        "derived_from_manifest": Path(os.path.relpath(source_manifest_path, output_manifest.parent)).as_posix(),
        "derived_from_pair_id": source_pair["id"],
        "derivation": {
            "script": "scripts/derive_alpine_rootfs_artifacts.py",
            "format": "normalized_uncompressed_tar",
            "member_order": "path_sorted",
            "normalized_mtime": 0,
            "normalized_user_group_names": True,
            "path_validation": "reject absolute, parent-traversal, backslash, and NUL member paths",
        },
    }
    manifest = {
        "name": manifest_name,
        "version": 1,
        "base_dir": base_dir,
        "pairs": [pair],
    }
    return json.dumps(manifest, indent=2) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Derive normalized uncompressed Alpine rootfs tar artifacts from existing "
            "minirootfs tar.gz files and write an enabled manifest for baseline runs."
        )
    )
    parser.add_argument(
        "--source-manifest",
        "--manifest",
        dest="source_manifest",
        default=DEFAULT_SOURCE_MANIFEST,
        help="manifest containing the source minirootfs tar.gz pair",
    )
    parser.add_argument("--pair", default=DEFAULT_SOURCE_PAIR, help="source pair id to derive")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="manifest path to write; defaults to <output-dir>/smartota-alpine-rootfs.json",
    )
    parser.add_argument(
        "--output-manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help="name field for the generated manifest",
    )
    parser.add_argument(
        "--output-pair-id",
        default=None,
        help="pair id for the generated rootfs pair; defaults to <source-pair>_rootfs_tar",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
