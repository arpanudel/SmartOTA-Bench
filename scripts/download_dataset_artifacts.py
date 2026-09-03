import argparse
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DatasetManifestError, validate_file_metadata, validate_manifest_schema


def main(args):
    manifest_path = Path(args.manifest).expanduser().resolve()
    data = validate_manifest_schema(manifest_path)
    base_dir = _resolve_base_dir(manifest_path, data)
    selected_ids = set(args.pair or [])

    processed = 0
    for pair in data["pairs"]:
        explicit = pair["id"] in selected_ids
        if selected_ids and not explicit:
            continue
        if not selected_ids and not pair.get("enabled", True) and not args.include_disabled:
            continue
        _download_pair(pair, base_dir, dry_run=args.dry_run, force=args.force)
        processed += 1

    if selected_ids and processed != len(selected_ids):
        missing = ", ".join(sorted(selected_ids - {pair["id"] for pair in data["pairs"]}))
        raise DatasetManifestError(f"requested pair id(s) not found: {missing}")
    print(f"Processed {processed} dataset pair(s)")


def _download_pair(pair, base_dir, *, dry_run, force):
    for prefix in ("old", "new"):
        url = pair.get(f"{prefix}_url")
        if not url:
            raise DatasetManifestError(
                f"pair '{pair['id']}' is missing required field '{prefix}_url' for download"
            )
        expected_size = pair.get(f"{prefix}_size_bytes")
        expected_sha = pair.get(f"{prefix}_sha256")
        if expected_size is None or expected_sha is None:
            raise DatasetManifestError(
                f"pair '{pair['id']}' {prefix} artifact needs size and SHA-256 before download"
            )

        path = _resolve_artifact_path(base_dir, pair[f"{prefix}_file"])
        if path.exists() and not force:
            validate_file_metadata(
                path,
                expected_size_bytes=expected_size,
                expected_sha256=expected_sha,
                label=f"pair '{pair['id']}' {prefix} artifact",
            )
            print(f"Validated existing artifact: {path}")
            continue

        print(f"Downloading {url} -> {path}")
        if dry_run:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        _download_url(url, path)
        validate_file_metadata(
            path,
            expected_size_bytes=expected_size,
            expected_sha256=expected_sha,
            label=f"pair '{pair['id']}' {prefix} artifact",
        )


def _resolve_base_dir(manifest_path, data):
    base_dir = data.get("base_dir")
    if base_dir is None:
        return manifest_path.parent
    candidate = Path(base_dir).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _resolve_artifact_path(base_dir, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _download_url(url, path):
    with urllib.request.urlopen(url) as response:
        with path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download URL-backed SmartOTA-Bench artifacts with size and SHA-256 validation."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair", action="append", help="specific pair id to download; may be repeated")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="redownload even if the local file exists")
    main(parser.parse_args())
