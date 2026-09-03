# Dataset Reproducibility Notes

SmartOTA-Bench tracks manifests and recipes, not large raw artifacts. Keep
downloads, image builds, container layers, and generated rootfs archives under
ignored `data/raw/` or `data/processed/` paths. Enable a manifest pair only
after registering local files with hashes, sizes, license notes, block size,
domain, and scenario.

Do not commit downloaded archives, rootfs tars, container layers, disk images,
or derived binaries. Keep them under ignored `data/raw/` or `data/processed/`
paths and commit only manifests, scripts, tests, and documentation.

Enabled real-artifact pairs must carry enough metadata to reproduce and audit a
run: old/new paths, old/new sizes, old/new SHA-256 hashes, domain, artifact
type, compression status or compression-status source, block size, scenario,
source/provenance, and license notes.

## Common Registration

Use `scripts/register_dataset_artifacts.py` for manually acquired or locally
built pairs:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/<manifest>.json \
  --pair <pair-id> \
  --old-file data/raw/<domain>/old-artifact \
  --new-file data/raw/<domain>/new-artifact \
  --block-size-bytes 65536 \
  --domain <domain> \
  --scenario "<short scenario>" \
  --source "<download URL, build recipe, digest, or revision>" \
  --license-notes "<license and redistribution notes>" \
  --artifact-type <artifact-type> \
  --compression-status <compressed|uncompressed|mixed|unknown> \
  --compression-status-source "<how compression was determined>" \
  --enable \
  --write
```

The script computes `old_size_bytes`, `new_size_bytes`, `old_sha256`, and
`new_sha256`. Review the diff before committing; only manifests and notes
belong in git.

## Synthetic Correctness And Adversarial Fixtures

Tracked manifests:

- `datasets/manifests/smartota-smoke.json`
- `datasets/manifests/smartota-synthetic-adversarial.json`

Generate deterministic local artifacts:

```bash
python scripts/generate_synthetic_datasets.py \
  --output-dir data/processed/synthetic \
  --manifest data/processed/synthetic/smartota-smoke.json \
  --seed 20260501 \
  --block-size-bytes 1024 \
  --block-count 12
```

The generator covers small correctness fixtures plus adversarial reorder, grow,
shrink, and sparse-corruption cases. These generated artifacts are safe to
regenerate locally and should stay under `data/processed/synthetic/`.

## Alpine Compressed Minirootfs

Tracked manifest:

- `datasets/manifests/smartota-alpine-expanded.json`

Acquire the pinned compressed Alpine pair:

```bash
python scripts/download_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-alpine-expanded.json \
  --pair alpine_3_20_to_3_21_minirootfs_x86_64
```

This pair already declares upstream URLs, SHA-256 hashes, sizes, block size,
artifact type, compression status, source, and license notes. Keep downloaded
`.tar.gz` files under `data/raw/alpine/`.

To make the downloaded compressed pair benchmarkable from the tracked plan,
register the local files back into the manifest after download validation:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-alpine-expanded.json \
  --pair alpine_3_20_to_3_21_minirootfs_x86_64 \
  --old-file data/raw/alpine/alpine-minirootfs-3.20.0-x86_64.tar.gz \
  --new-file data/raw/alpine/alpine-minirootfs-3.21.0-x86_64.tar.gz \
  --block-size-bytes 65536 \
  --domain os_rootfs \
  --scenario "Alpine minirootfs compressed release transition 3.20.0 to 3.21.0" \
  --source "Alpine Linux release artifacts at pinned URLs in this manifest" \
  --license-notes "Alpine Linux minirootfs redistribution terms reviewed for this benchmark run" \
  --artifact-type compressed_rootfs_tar \
  --compression-status compressed \
  --compression-status-source release_artifact_tar_gz_suffix \
  --enable \
  --write
```

For the point-release compressed pair, first choose a specific Alpine 3.21.x
minirootfs URL, record its upstream checksum and exact version in `source`,
download it under `data/raw/alpine/`, then run the same registration flow
against `alpine_3_21_point_release_minirootfs_x86_64`.

## Alpine Uncompressed Rootfs

Tracked manifest:

- `datasets/manifests/smartota-alpine-rootfs.json`

Derive normalized uncompressed rootfs tar artifacts from the downloaded Alpine
compressed pair:

```bash
python scripts/derive_alpine_rootfs_artifacts.py \
  --source-manifest datasets/manifests/smartota-alpine-expanded.json \
  --pair alpine_3_20_to_3_21_minirootfs_x86_64 \
  --output-dir data/processed/alpine-rootfs \
  --output-manifest data/processed/alpine-rootfs/smartota-alpine-rootfs.json
```

The derivation sorts tar members, normalizes mtime/user/group names, and rejects
unsafe member paths. To register the derived artifacts into the tracked plan:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-alpine-rootfs.json \
  --pair alpine_3_20_to_3_21_rootfs_tar_x86_64 \
  --old-file data/processed/alpine-rootfs/alpine-minirootfs-3.20.0-x86_64.rootfs.tar \
  --new-file data/processed/alpine-rootfs/alpine-minirootfs-3.21.0-x86_64.rootfs.tar \
  --block-size-bytes 65536 \
  --domain os_rootfs \
  --scenario "normalized uncompressed Alpine rootfs tar 3.20.0 to 3.21.0" \
  --source "Derived from Alpine minirootfs release artifacts" \
  --license-notes "Derived from Alpine Linux minirootfs; verify upstream redistribution terms" \
  --artifact-type rootfs_tar \
  --compression-status uncompressed \
  --compression-status-source derivation_normalized_uncompressed_tar \
  --enable \
  --write
```

## Debian, Ubuntu, And Yocto-Style Updates

Tracked manifests:

- `datasets/manifests/smartota-debian-ubuntu-updates.json`
- `datasets/manifests/smartota-linux-images.json`

Recommended local acquisition patterns:

- Package cache pair: create two tar archives from pinned apt repository
  snapshots or package cache directories before and after a recorded security
  update transaction.
- Rootfs pair: create rootfs tar artifacts with `debootstrap`, `mmdebstrap`,
  `ubuntu-base`, or a reproducible Yocto/OpenEmbedded build at pinned
  timestamps/revisions.
- Image pair: register pinned cloud/image artifacts or locally converted images
  under `data/raw/linux-images/`.

Example Debian rootfs registration:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-debian-ubuntu-updates.json \
  --pair debian_bookworm_rootfs_security_update_amd64 \
  --old-file data/raw/debian-ubuntu/debian/bookworm-rootfs-old.tar \
  --new-file data/raw/debian-ubuntu/debian/bookworm-rootfs-new.tar \
  --block-size-bytes 65536 \
  --domain os_rootfs \
  --scenario "Debian rootfs before and after a pinned apt security update" \
  --source "snapshot.debian.org timestamps plus recorded apt package list" \
  --license-notes "Record Debian base system and package-level license metadata" \
  --artifact-type rootfs_tar \
  --compression-status uncompressed \
  --compression-status-source "created as deterministic rootfs tar" \
  --enable \
  --write
```

For Yocto/OpenEmbedded, record layer names and SHAs, distro/machine settings,
image recipe, generated license manifest path, and build container/toolchain
version in `source` or additional manifest fields. A minimal rootfs pair should
use `yocto_core_image_minimal_packagefeed_update_qemuarm64` from
`smartota-debian-ubuntu-updates.json`:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-debian-ubuntu-updates.json \
  --pair yocto_core_image_minimal_packagefeed_update_qemuarm64 \
  --old-file data/raw/debian-ubuntu/yocto/core-image-minimal-rootfs-old.tar \
  --new-file data/raw/debian-ubuntu/yocto/core-image-minimal-rootfs-new.tar \
  --block-size-bytes 65536 \
  --domain embedded_linux \
  --scenario "Yocto core-image-minimal rootfs across pinned layer revisions" \
  --source "poky/meta-openembedded layer SHAs, MACHINE=qemuarm64, image recipe, and build container digest" \
  --license-notes "Generated Yocto license manifest path and layer license notes recorded locally" \
  --artifact-type rootfs_tar \
  --compression-status uncompressed \
  --compression-status-source "created as deterministic rootfs tar" \
  --enable \
  --write
```

## Container Layer Updates

Tracked manifest:

- `datasets/manifests/smartota-container-layers.json`

### Deterministic YOLO-Style Perception Container Fixture

The pair `container_yolo_perception_update` is a small synthetic bridge toward
future YOLO OTA fleet deployment. It creates old/new deterministic application
trees with a detector script, config, entrypoint, and mock model weights; the
new tree changes detector/config/model bytes and adds `healthcheck.py`.

Build ignored local artifacts for the small pair:

```bash
python scripts/build_container_yolo_artifacts.py \
  --fixture small \
  --output-dir data/raw/containers/yolo-perception
```

The builder writes stable uncompressed tar files with sorted paths, fixed
mtime, fixed uid/gid, fixed uname/gname, and stable file modes:

```text
data/raw/containers/yolo-perception/yolo-perception-layer-old.tar
data/raw/containers/yolo-perception/yolo-perception-layer-new.tar
data/raw/containers/yolo-perception/registration-metadata.json
```

Register and enable the pair after generation:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-container-layers.json \
  --pair container_yolo_perception_update \
  --old-file data/raw/containers/yolo-perception/yolo-perception-layer-old.tar \
  --new-file data/raw/containers/yolo-perception/yolo-perception-layer-new.tar \
  --block-size-bytes 16384 \
  --domain container_av_perception \
  --scenario containerized_yolo_perception_update \
  --source "Locally generated deterministic SmartOTA-Bench YOLO-style perception container fixture" \
  --license-notes "Synthetic fixture with no third-party model weights" \
  --artifact-type perception_container_tar \
  --compression-status uncompressed \
  --compression-status-source deterministic_tar_derivation \
  --enable \
  --write
```

The pair `container_yolo_perception_multiblock_update` is a larger synthetic
YOLO/container-style fixture for block-level OTA planning. It keeps multiple
mock model regions and asset files unchanged, changes several model regions,
changes detector/config metadata, inserts a new mock model region, and adds
telemetry/healthcheck files. It is designed to exercise block-size sensitivity,
copy/delta/raw choices, backup and checkpoint costs, staging pressure, and
deployment-aware metrics. It is not a production container image.

Build ignored local artifacts for the multiblock pair:

```bash
python scripts/build_container_yolo_artifacts.py \
  --fixture multiblock \
  --output-dir data/raw/containers/yolo-perception
```

The multiblock builder output is also a deterministic uncompressed tar
derivation:

```text
data/raw/containers/yolo-perception/yolo-perception-multiblock-layer-old.tar
data/raw/containers/yolo-perception/yolo-perception-multiblock-layer-new.tar
data/raw/containers/yolo-perception/registration-metadata-multiblock.json
```

Register and enable the multiblock pair after generation:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-container-layers.json \
  --pair container_yolo_perception_multiblock_update \
  --old-file data/raw/containers/yolo-perception/yolo-perception-multiblock-layer-old.tar \
  --new-file data/raw/containers/yolo-perception/yolo-perception-multiblock-layer-new.tar \
  --block-size-bytes 65536 \
  --domain container_av_perception \
  --scenario containerized_yolo_perception_multiblock_update \
  --source "Locally generated deterministic synthetic fixture" \
  --license-notes "Synthetic fixture; no third-party model weights or third-party model files" \
  --artifact-type perception_container_tar \
  --compression-status uncompressed \
  --compression-status-source deterministic_tar_derivation \
  --enable \
  --write
```

Both YOLO fixtures are generated from repo code and contain synthetic mock model
bytes only. They do not download, embed, or derive real YOLO weights or
third-party model files. Determinism comes from SHA-256-labeled byte generation,
sorted tar member order, fixed timestamp `0`, fixed uid/gid `0`, empty owner and
group names, stable modes, and uncompressed tar output. The generated tar files
and tree directories belong under ignored `data/raw/`, not in git.

Acquire OCI layers from pinned image digests. Use tools such as `skopeo`,
`crane`, `oras`, `docker save`, or `podman save`, then extract the specific
changed `layer.tar` files into `data/raw/containers/<image>/`.

Record immutable old/new image digests first, export or copy each image, and
extract the selected changed layer tar into paths such as:

```text
data/raw/containers/alpine/base-layer-old.tar
data/raw/containers/alpine/base-layer-new.tar
```

Record at minimum:

- registry and immutable image digest for old and new images;
- selected layer digest and layer order;
- Dockerfile or build revision when locally built;
- package/application dependency manifests;
- base image and package license notes.

Example registration:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-container-layers.json \
  --pair oci_alpine_base_layer_update_amd64 \
  --old-file data/raw/containers/alpine/base-layer-old.tar \
  --new-file data/raw/containers/alpine/base-layer-new.tar \
  --block-size-bytes 65536 \
  --domain container_layers \
  --scenario "OCI Alpine base layer update from pinned image digests" \
  --source "registry image digests and layer digests recorded locally" \
  --license-notes "Record image, package, and base distribution licenses" \
  --artifact-type oci_layer_tar \
  --compression-status uncompressed \
  --compression-status-source exported_oci_layer_tar \
  --enable \
  --write
```

## Autoware-Style AV Module Fixture

Tracked manifest:

- `datasets/manifests/smartota-automotive.json`

The pair `autoware_style_perception_module_update` is a synthetic
Autoware-style perception-module update for SDV/AV OTA planning. It does not
download or embed Autoware, proprietary AV software, or third-party model
weights. The generated trees mimic a ROS-style package layout with
`package.xml`, `CMakeLists.txt`, C++ source files, config, launch XML, a mock
ONNX-like model payload, and diagnostics metadata.

Build ignored local artifacts:

```bash
python scripts/build_autoware_style_artifacts.py \
  --output-dir data/raw/automotive/autoware-style-perception
```

The builder writes stable uncompressed tar files with sorted paths, fixed
mtime, fixed uid/gid, and stable file modes:

```text
data/raw/automotive/autoware-style-perception/autoware-style-perception-old.tar
data/raw/automotive/autoware-style-perception/autoware-style-perception-new.tar
data/raw/automotive/autoware-style-perception/registration-metadata.json
```

Register and enable the pair after generation:

```bash
python scripts/register_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-automotive.json \
  --pair autoware_style_perception_module_update \
  --old-file data/raw/automotive/autoware-style-perception/autoware-style-perception-old.tar \
  --new-file data/raw/automotive/autoware-style-perception/autoware-style-perception-new.tar \
  --block-size-bytes 16384 \
  --domain autoware_style_av_stack \
  --scenario autoware_style_perception_module_update \
  --source "Locally generated deterministic fixture inspired by Autoware-style module layout" \
  --license-notes "Synthetic fixture; no Autoware source code, no proprietary AV software, and no third-party model weights" \
  --artifact-type autoware_module_tar \
  --compression-status uncompressed \
  --compression-status-source deterministic_tar_derivation \
  --enable \
  --write
```

These artifacts are generated from repo code and belong under ignored
`data/raw/`, not in git.

## Running Benchmarks

After enabling local pairs:

```bash
python scripts/run_benchmark_suite.py \
  --manifests \
    datasets/manifests/smartota-alpine-expanded.json \
    datasets/manifests/smartota-alpine-rootfs.json \
    datasets/manifests/smartota-debian-ubuntu-updates.json \
    datasets/manifests/smartota-container-layers.json \
    data/processed/synthetic/smartota-smoke.json \
  --policies backup_safe_copy_delta deployment_aware_greedy \
  --deployment-config default='{}' \
  --interruption-setting default=0.25,0.5,0.75 \
  --seeds 1 2 3
```

The benchmark suite writes JSONL, CSV, and Markdown outputs with artifact
hashes, git commit, Python/dependency versions, replay validity, deployment
metrics, and interruption recovery metrics.
