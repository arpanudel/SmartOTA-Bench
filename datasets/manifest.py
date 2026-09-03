import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


class DatasetManifestError(ValueError):
    """Raised when a dataset manifest cannot be loaded safely."""


@dataclass(frozen=True)
class ArtifactMetadata:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class OTADatasetPair:
    dataset_id: str
    domain: str
    scenario: str
    block_size_bytes: int
    old_path: Path
    new_path: Path
    old_size_bytes: int
    new_size_bytes: int
    old_sha256: str
    new_sha256: str
    source: str = ""
    license_notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def id(self):
        return self.dataset_id

    def to_record(self):
        return {
            "id": self.dataset_id,
            "domain": self.domain,
            "scenario": self.scenario,
            "block_size_bytes": self.block_size_bytes,
            "old_file": str(self.old_path),
            "new_file": str(self.new_path),
            "old_size_bytes": self.old_size_bytes,
            "new_size_bytes": self.new_size_bytes,
            "old_sha256": self.old_sha256,
            "new_sha256": self.new_sha256,
            "source": self.source,
            "license_notes": self.license_notes,
            **self.extra,
        }


@dataclass(frozen=True)
class OTADatasetManifest:
    name: str
    version: int
    path: Path
    base_dir: Path
    pairs: tuple[OTADatasetPair, ...]

    def get_pair(self, dataset_id):
        for pair in self.pairs:
            if pair.dataset_id == dataset_id:
                return pair
        raise KeyError(f"dataset pair '{dataset_id}' was not found")

    def by_domain(self, domain):
        return tuple(pair for pair in self.pairs if pair.domain == domain)

    def to_records(self):
        return [pair.to_record() for pair in self.pairs]


_PAIR_FIELDS = {
    "id",
    "domain",
    "scenario",
    "old_file",
    "new_file",
    "block_size_bytes",
    "old_size_bytes",
    "new_size_bytes",
    "old_sha256",
    "new_sha256",
    "source",
    "license_notes",
    "enabled",
}


_REAL_ARTIFACT_DOMAINS = {
    "os_rootfs",
    "os_packages",
    "os_image",
    "embedded_linux",
    "edge_iot",
    "autonomous_vehicle",
    "autoware_style_av_stack",
    "av_software",
    "container_layers",
    "container_image",
    "container_av_perception",
}


_ENABLED_REAL_PAIR_REQUIRED_FIELDS = (
    "old_size_bytes",
    "new_size_bytes",
    "old_sha256",
    "new_sha256",
    "artifact_type",
    "source",
    "license_notes",
)


_COMPRESSION_STATUS_VALUES = {
    "compressed",
    "uncompressed",
    "mixed",
    "unknown",
}


def load_manifest(path, base_dir=None, include_disabled=False):
    """Load an OTA dataset manifest and compute metadata for local file pairs."""
    manifest_path = Path(path).expanduser().resolve()
    data = _read_manifest(manifest_path)
    _validate_manifest_data(data, manifest_path)
    resolved_base_dir = _resolve_base_dir(manifest_path, data, base_dir)
    pairs = []

    raw_pairs = data.get("pairs")
    if not isinstance(raw_pairs, list):
        raise DatasetManifestError("manifest field 'pairs' must be a list")

    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, dict):
            raise DatasetManifestError(f"pair at index {index} must be an object")
        if not raw_pair.get("enabled", True) and not include_disabled:
            continue
        pairs.append(_load_pair(raw_pair, index, resolved_base_dir))

    return OTADatasetManifest(
        name=str(data.get("name", manifest_path.stem)),
        version=int(data.get("version", 1)),
        path=manifest_path,
        base_dir=resolved_base_dir,
        pairs=tuple(pairs),
    )


def load_dataset_pairs(path, base_dir=None, include_disabled=False):
    return load_manifest(
        path,
        base_dir=base_dir,
        include_disabled=include_disabled,
    ).pairs


def validate_manifest_schema(path):
    """Validate manifest structure without requiring disabled artifacts to exist."""
    manifest_path = Path(path).expanduser().resolve()
    data = _read_manifest(manifest_path)
    _validate_manifest_data(data, manifest_path)
    return data


def compute_file_metadata(path):
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise DatasetManifestError(f"artifact does not exist: {candidate}")
    if not candidate.is_file():
        raise DatasetManifestError(f"artifact is not a file: {candidate}")
    return ArtifactMetadata(
        path=candidate,
        size_bytes=candidate.stat().st_size,
        sha256=_file_sha256(candidate),
    )


def validate_file_metadata(path, expected_size_bytes=None, expected_sha256=None, label="artifact"):
    metadata = compute_file_metadata(path)
    if expected_size_bytes is not None and metadata.size_bytes != expected_size_bytes:
        raise DatasetManifestError(
            f"{label} size mismatch: expected {expected_size_bytes}, file has {metadata.size_bytes}"
        )
    if expected_sha256 is not None and str(expected_sha256).lower() != metadata.sha256:
        raise DatasetManifestError(
            f"{label} sha256 mismatch: expected {expected_sha256}, file has {metadata.sha256}"
        )
    return metadata


def _read_manifest(manifest_path):
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DatasetManifestError(f"invalid JSON in {manifest_path}: {exc}") from exc
    except OSError as exc:
        raise DatasetManifestError(f"could not read manifest {manifest_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise DatasetManifestError("manifest root must be an object")
    return data


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_manifest_data(data, manifest_path):
    if "name" in data and (not isinstance(data["name"], str) or not data["name"]):
        raise DatasetManifestError(f"manifest {manifest_path} field 'name' must be a non-empty string")

    if "version" in data:
        version = data["version"]
        if not isinstance(version, int) or version <= 0:
            raise DatasetManifestError(f"manifest {manifest_path} field 'version' must be a positive integer")

    if "base_dir" in data and not isinstance(data["base_dir"], str):
        raise DatasetManifestError(f"manifest {manifest_path} field 'base_dir' must be a string")

    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise DatasetManifestError("manifest field 'pairs' must be a list")

    seen_ids = set()
    for index, pair in enumerate(pairs):
        _validate_pair_schema(pair, index, seen_ids)


def _validate_pair_schema(raw_pair, index, seen_ids):
    if not isinstance(raw_pair, dict):
        raise DatasetManifestError(f"pair at index {index} must be an object")

    missing = [
        field_name
        for field_name in ("id", "domain", "scenario", "old_file", "new_file", "block_size_bytes")
        if field_name not in raw_pair
    ]
    if missing:
        raise DatasetManifestError(
            f"pair at index {index} is missing required field(s): {', '.join(missing)}"
        )

    dataset_id = raw_pair["id"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise DatasetManifestError(f"pair at index {index} field 'id' must be a non-empty string")
    if dataset_id in seen_ids:
        raise DatasetManifestError(f"duplicate dataset pair id: {dataset_id}")
    seen_ids.add(dataset_id)

    for field_name in ("domain", "scenario", "old_file", "new_file"):
        if not isinstance(raw_pair[field_name], str) or not raw_pair[field_name]:
            raise DatasetManifestError(
                f"pair '{dataset_id}' field '{field_name}' must be a non-empty string"
            )

    enabled = raw_pair.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise DatasetManifestError(f"pair '{dataset_id}' field 'enabled' must be a boolean")

    block_size = raw_pair["block_size_bytes"]
    if not isinstance(block_size, int) or block_size <= 0:
        raise DatasetManifestError(
            f"pair '{dataset_id}' field 'block_size_bytes' must be a positive integer"
        )

    for field_name in ("old_size_bytes", "new_size_bytes"):
        if field_name in raw_pair:
            value = raw_pair[field_name]
            if not isinstance(value, int) or value < 0:
                raise DatasetManifestError(
                    f"pair '{dataset_id}' field '{field_name}' must be a non-negative integer"
                )

    for field_name in ("old_sha256", "new_sha256"):
        if field_name in raw_pair:
            value = raw_pair[field_name]
            if not isinstance(value, str) or _SHA256_RE.match(value) is None:
                raise DatasetManifestError(
                    f"pair '{dataset_id}' field '{field_name}' must be a 64-character SHA-256 hex string"
                )

    for field_name in ("old_url", "new_url"):
        if field_name in raw_pair and (
            not isinstance(raw_pair[field_name], str) or not raw_pair[field_name]
        ):
            raise DatasetManifestError(
                f"pair '{dataset_id}' field '{field_name}' must be a non-empty string"
            )

    for field_name in ("artifact_type", "compression_status", "compression_status_source"):
        if field_name in raw_pair and (
            not isinstance(raw_pair[field_name], str) or not raw_pair[field_name].strip()
        ):
            raise DatasetManifestError(
                f"pair '{dataset_id}' field '{field_name}' must be a non-empty string"
            )

    compression_status = raw_pair.get("compression_status")
    if (
        compression_status is not None
        and compression_status not in _COMPRESSION_STATUS_VALUES
    ):
        allowed = ", ".join(sorted(_COMPRESSION_STATUS_VALUES))
        raise DatasetManifestError(
            f"pair '{dataset_id}' field 'compression_status' must be one of: {allowed}"
        )

    _validate_enabled_real_pair_contract(raw_pair, dataset_id)


def _validate_enabled_real_pair_contract(raw_pair, dataset_id):
    if not raw_pair.get("enabled", True):
        return
    if raw_pair.get("domain") not in _REAL_ARTIFACT_DOMAINS:
        return

    missing = []
    for field_name in _ENABLED_REAL_PAIR_REQUIRED_FIELDS:
        value = raw_pair.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)

    has_compression_metadata = bool(raw_pair.get("compression_status")) or bool(
        raw_pair.get("compression_status_source")
    )
    if not has_compression_metadata:
        missing.append("compression_status or compression_status_source")

    if missing:
        raise DatasetManifestError(
            f"enabled real pair '{dataset_id}' is missing reproducibility field(s): "
            f"{', '.join(missing)}"
        )


def _resolve_base_dir(manifest_path, data, base_dir):
    if base_dir is not None:
        candidate = Path(base_dir).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        return candidate.resolve()

    manifest_base_dir = data.get("base_dir")
    if manifest_base_dir is not None:
        candidate = Path(manifest_base_dir).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        return candidate.resolve()

    return manifest_path.parent.resolve()


def _load_pair(raw_pair, index, base_dir):
    old_path = _resolve_file_path(base_dir, raw_pair["old_file"], raw_pair["id"], "old_file")
    new_path = _resolve_file_path(base_dir, raw_pair["new_file"], raw_pair["id"], "new_file")
    old_metadata = compute_file_metadata(old_path)
    new_metadata = compute_file_metadata(new_path)

    _validate_declared_metadata(
        raw_pair,
        raw_pair["id"],
        "old",
        old_metadata.size_bytes,
        old_metadata.sha256,
    )
    _validate_declared_metadata(
        raw_pair,
        raw_pair["id"],
        "new",
        new_metadata.size_bytes,
        new_metadata.sha256,
    )

    return OTADatasetPair(
        dataset_id=str(raw_pair["id"]),
        domain=str(raw_pair["domain"]),
        scenario=str(raw_pair["scenario"]),
        block_size_bytes=raw_pair["block_size_bytes"],
        old_path=old_path,
        new_path=new_path,
        old_size_bytes=old_metadata.size_bytes,
        new_size_bytes=new_metadata.size_bytes,
        old_sha256=old_metadata.sha256,
        new_sha256=new_metadata.sha256,
        source=str(raw_pair.get("source", "")),
        license_notes=str(raw_pair.get("license_notes", "")),
        extra={key: value for key, value in raw_pair.items() if key not in _PAIR_FIELDS},
    )


def _resolve_file_path(base_dir, value, dataset_id, field_name):
    if not isinstance(value, str) or not value:
        raise DatasetManifestError(
            f"pair '{dataset_id}' field '{field_name}' must be a non-empty path string"
        )

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if not path.exists():
        raise DatasetManifestError(
            f"pair '{dataset_id}' field '{field_name}' does not exist: {path}"
        )
    if not path.is_file():
        raise DatasetManifestError(
            f"pair '{dataset_id}' field '{field_name}' is not a file: {path}"
        )
    return path


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_declared_metadata(raw_pair, dataset_id, prefix, size_bytes, sha256):
    expected_size = raw_pair.get(f"{prefix}_size_bytes")
    if expected_size is not None and expected_size != size_bytes:
        raise DatasetManifestError(
            f"pair '{dataset_id}' {prefix}_size_bytes mismatch: "
            f"manifest has {expected_size}, file has {size_bytes}"
        )

    expected_sha256 = raw_pair.get(f"{prefix}_sha256")
    if expected_sha256 is not None and str(expected_sha256).lower() != sha256:
        raise DatasetManifestError(
            f"pair '{dataset_id}' {prefix}_sha256 mismatch: "
            f"manifest has {expected_sha256}, file has {sha256}"
        )
