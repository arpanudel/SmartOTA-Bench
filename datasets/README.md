# SmartOTA-Bench Datasets

This directory stores reproducible dataset manifests, not large OTA artifacts.
Raw downloads and generated binaries belong under ignored `data/` paths.

## Manifests

Tracked dataset plans live in `datasets/manifests/`:

- `smartota-smoke.json`: deterministic synthetic growing, shrinking, reordered,
  repeated, random, compressed-like, and adversarial binary pairs.
- `smartota-alpine-expanded.json`: Alpine minirootfs and APK-cache expansion
  beyond the original bundled pair.
- `smartota-alpine-rootfs.json`: normalized uncompressed Alpine rootfs tar
  pairs derived from downloaded minirootfs artifacts.
- `smartota-debian-ubuntu-updates.json`: Debian/Ubuntu package-cache and
  rootfs update plans, plus a Yocto-style rootfs build plan.
- `smartota-container-layers.json`: OCI/container layer update plans.
- `smartota-synthetic-adversarial.json`: focused generated correctness and
  adversarial reorder/grow/shrink/corruption cases.
- `smartota-linux-images.json`: Debian, Ubuntu, Yocto/OpenEmbedded, and edge
  Linux image update candidates.
- `smartota-automotive.json`: Autoware, ROS 2, calibration, and autonomy
  container update candidates.
- `smartota-av-software.json`: perception models, maps, planning configs, and
  firmware-like AV software bundles.

Pairs are disabled until their files exist locally. Disabled pairs are still
schema-validated, but the loader does not require their artifacts to exist.

## Current Manifest Inventory

All tracked `datasets/manifests/smartota-*.json` pairs are disabled by default.
Generated or locally registered manifests under ignored `data/` paths may enable
pairs after their artifacts are present and validated.

| Manifest | Category | Pair status | What is still missing before publication-scale real runs |
| --- | --- | --- | --- |
| `smartota-smoke.json` | synthetic | 10 disabled tracked plans | Generate local artifacts with `scripts/generate_synthetic_datasets.py`; tracked copy remains disabled so missing generated files do not break clones. |
| `smartota-synthetic-adversarial.json` | synthetic | 6 disabled tracked plans | Same generated synthetic artifacts as smoke; no third-party license work. |
| `smartota-alpine-expanded.json` | Alpine compressed minirootfs/APK | 3 disabled real plans | First minirootfs pair has URLs, sizes, hashes, block size, scenario, provenance, and compression metadata; local `.tar.gz` files and final license verification are still needed. Point-release and APK-cache pairs still need pinned artifact selection, URLs/hashes/sizes, acquisition recipe, and registration. |
| `smartota-alpine-rootfs.json` | Alpine normalized rootfs | 2 disabled real plans | Derive local uncompressed rootfs tar files from downloaded minirootfs artifacts, then register sizes and hashes into the tracked plan or use the generated manifest under `data/processed/alpine-rootfs/`. |
| `smartota-debian-ubuntu-updates.json` | Debian/Ubuntu/Yocto rootfs and package-cache | 4 disabled real plans | Local build/acquisition scripts, artifact files, sizes, hashes, exact snapshot/layer revisions, and final package/license manifests are needed. |
| `smartota-container-layers.json` | container/image layers | 6 real plans | Pinned image digests, selected layer digests/order, exported layer tar files, sizes, hashes, and image/package license notes are needed for external container pairs. The small and multiblock YOLO-style perception fixtures are generated locally with `scripts/build_container_yolo_artifacts.py` and can be enabled after registration. |
| `smartota-linux-images.json` | Debian/Ubuntu/Yocto/edge images | 4 disabled real plans | Pinned image artifacts or reproducible builds, exact image-format/compression notes, sizes, hashes, acquisition scripts, and license/provenance notes are needed. |
| `smartota-automotive.json` | AV/edge/Autoware/ROS/container | 5 real plans | Redistributable or generated artifacts, build/download recipes, hashes/sizes, package/image license manifests, and artifact-specific tests are needed for external automotive pairs. The Autoware-style perception module fixture is generated locally and can be enabled after running `scripts/build_autoware_style_artifacts.py` and registration. |
| `smartota-av-software.json` | AV software bundles | 4 disabled real plans | Redistributable/generated model, map, planning, and firmware-like bundles plus exact provenance, sizes, hashes, licenses, and registration tests are needed. |

Enabled real-artifact pairs must include `old_file`, `new_file`,
`old_size_bytes`, `new_size_bytes`, `old_sha256`, `new_sha256`, `domain`,
`artifact_type`, `compression_status` or `compression_status_source`,
`block_size_bytes`, `scenario`, `source`, and `license_notes`. Schema
validation enforces this contract for real domains when `enabled` is true.

## Storage Layout

Use this layout for local data:

```text
data/
  raw/                     # downloaded or manually registered external artifacts
    alpine/
    debian-ubuntu/
    containers/
    linux-images/
    automotive/
    av-software/
  processed/
    synthetic/             # generated smoke artifacts
```

`data/raw/`, `data/processed/`, and root Alpine tarballs are ignored so raw
artifacts do not enter Git.

## Synthetic Smoke Data

Generate the local smoke dataset with:

```bash
python scripts/generate_synthetic_datasets.py
```

The generator writes small deterministic binaries and a generated manifest under
`data/processed/synthetic/`. It covers:

- growing images with appended blocks;
- shrinking images with removed content;
- reordered images where source blocks move;
- repeated block layouts;
- high-entropy random block changes;
- compressed-like binary payload drift.
- adversarial reorder/grow/shrink/corruption cases.

Run deterministic baselines for every enabled pair in a manifest:

```bash
python scripts/run_baselines.py --manifest data/processed/synthetic/smartota-smoke.json
python scripts/run_baselines.py --manifest data/processed/synthetic/smartota-smoke.json \
  --policies full_replacement whole_file_bsdiff blockwise_bsdiff copy_only copy_delta backup_safe_copy_delta deployment_aware_greedy rsync_rolling_hash
```

Manifest runs default to those deterministic publication baselines. Legacy
action-space baselines such as `sequential_m`, `sequential_mb`, `copy_first`,
`random`, and `backup_aware_copy_delta` can still be selected explicitly.

Tracked manifests under `datasets/manifests/` keep pairs disabled until their
artifacts exist locally. Generate, download, or register artifacts first, then
run the manifest path that contains enabled pairs.

## Downloading And Registering Artifacts

URL-backed pairs with declared size and SHA-256 can be fetched with:

```bash
python scripts/download_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-alpine-expanded.json \
  --pair alpine_3_20_to_3_21_minirootfs_x86_64
```

Compressed rootfs archives hide much of the block-level OTA structure. After
downloading the Alpine minirootfs pair, derive normalized uncompressed rootfs
tar artifacts and a manifest that can be used directly by the baseline runner:

```bash
python scripts/derive_alpine_rootfs_artifacts.py
python scripts/run_baselines.py --manifest data/processed/alpine-rootfs/smartota-alpine-rootfs.json
```

For manually acquired or locally built artifacts, register paths and computed
metadata with:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-linux-images.json \
  --pair debian_bookworm_cloud_image_point_update_amd64 \
  --old-file data/raw/linux-images/debian/bookworm-cloud-old.qcow2 \
  --new-file data/raw/linux-images/debian/bookworm-cloud-new.qcow2 \
  --block-size-bytes 131072 \
  --domain os_image \
  --scenario "Debian cloud image point update" \
  --source "Pinned Debian cloud image artifacts" \
  --license-notes "Record Debian image license and mirror provenance" \
  --artifact-type qcow2_image \
  --compression-status-source "record exact qcow2/image compression during registration" \
  --enable \
  --write
```

The registration script computes SHA-256 and size metadata and can also update
domain, scenario, license notes, source, block size, artifact type, compression
status, compression-status source, tier, and download URLs before marking a pair
enabled after validation.

Detailed acquisition recipes for Alpine rootfs, Debian/Ubuntu package/rootfs
updates, container layers, and synthetic adversarial fixtures are in
`datasets/REPRODUCIBILITY.md`.

## Manifest API

```python
from datasets import load_manifest, validate_manifest_schema

validate_manifest_schema("datasets/manifests/smartota-smoke.json")
manifest = load_manifest("data/processed/synthetic/smartota-smoke.json")
pair = manifest.get_pair("synthetic_growing")

print(pair.old_path, pair.new_path)
print(pair.old_sha256, pair.new_sha256)
```

Required pair fields:

- `id`: stable dataset pair identifier.
- `domain`: dataset domain such as `synthetic`, `os_rootfs`,
  `embedded_linux`, `autonomous_vehicle`, or `av_software`.
- `scenario`: short explanation of the update case.
- `old_file` and `new_file`: local binary artifacts for the update pair.
- `block_size_bytes`: positive integer block size for OTA planning.

Optional fields:

- `old_size_bytes`, `new_size_bytes`, `old_sha256`, `new_sha256`: expected
  metadata validated against local files when a pair is enabled.
- `old_url`, `new_url`: download locations for artifacts with known metadata.
- `artifact_type`: real artifact format such as `compressed_rootfs_tar`,
  `rootfs_tar`, `package_cache_tar`, `oci_layer_tar`, or `qcow2_image`.
- `compression_status` and `compression_status_source`: explicit status
  (`compressed`, `uncompressed`, `mixed`, or `unknown`) and how it was known.
- `source`, `license_notes`, `tier`, and additional experiment metadata.

For enabled real-artifact domains, the metadata listed in the inventory section
is required rather than optional.
