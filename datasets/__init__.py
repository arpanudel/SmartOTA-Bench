"""Dataset manifest helpers for OTA experiments."""

from .manifest import (
    ArtifactMetadata,
    DatasetManifestError,
    OTADatasetManifest,
    OTADatasetPair,
    compute_file_metadata,
    load_dataset_pairs,
    load_manifest,
    validate_file_metadata,
    validate_manifest_schema,
)
from .synthetic import SYNTHETIC_CASES, generate_synthetic_dataset

__all__ = [
    "ArtifactMetadata",
    "DatasetManifestError",
    "OTADatasetManifest",
    "OTADatasetPair",
    "compute_file_metadata",
    "load_dataset_pairs",
    "load_manifest",
    "validate_file_metadata",
    "validate_manifest_schema",
    "SYNTHETIC_CASES",
    "generate_synthetic_dataset",
]
