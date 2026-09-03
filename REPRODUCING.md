# Reproducing SmartOTA-Bench

Run commands from the root of this release. Commands in this document are
portable release-time reconstructions unless a section explicitly labels them
as historical.

## Historical recorded commands

The immutable historical record is
`artifacts/final_deterministic_benchmark_20260717_clean_rerun/commands_used.md`.
It records the broad 16/64 KiB run, the bounded 4 KiB run, report generation,
and the result audit. It contains machine-local absolute paths and does not
record complete environment setup or unit-test commands.

That file is preserved exactly as archived. The portable commands below are
independently reconstructed from the recorded invocations, manifests, source,
profiles, and release-time validation; they are not claimed to be a verbatim
historical execution log.

## Release-time environment setup

Python 3.12 is the validated release environment. Create an isolated
environment and install the pinned deterministic benchmark/report dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The historical rows record Python 3.12.7, NumPy 1.26.4, Gym 0.26.2, bsdiff4
1.2.6, TensorBoard 2.20.0, and Torch 2.12.0. The archive lacks a complete
historical environment file. Torch and TensorBoard are not required for this
curated deterministic artifact, and no learned-model source or weights are
included.

## Release-time tests

Run the focused replay, deployment, dataset, deterministic-policy, benchmark,
report, and audit tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

The suite uses temporary files and does not rewrite the frozen archive or its
extracted results.

## Reproducibility audit

Audit the extracted frozen result ledger against its generation commit. This
is the historical-evidence audit requested for the release; only its input path
has been adjusted to the extraction supplied here:

```bash
.venv/bin/python scripts/audit_result_reproducibility.py \
  artifacts/final_deterministic_benchmark_20260717_clean_rerun/combined_benchmark_results.jsonl \
  --expected-commit 2942cbde2fd344cccea23aabbe8d5bf168610c71
```

Prepare the five enabled inputs without placing raw artifacts in Git:

```bash
.venv/bin/python scripts/download_dataset_artifacts.py \
  --manifest datasets/manifests/smartota-alpine-expanded.json \
  --pair alpine_3_20_to_3_21_minirootfs_x86_64

.venv/bin/python scripts/derive_alpine_rootfs_artifacts.py

.venv/bin/python scripts/build_autoware_style_artifacts.py \
  --output-dir data/raw/automotive/autoware-style-perception

.venv/bin/python scripts/build_container_yolo_artifacts.py \
  --fixture small \
  --output-dir data/raw/containers/yolo-perception

.venv/bin/python scripts/build_container_yolo_artifacts.py \
  --fixture multiblock \
  --output-dir data/raw/containers/yolo-perception
```

The following reconstructed broad sweep preserves the historical dimensions
while using release-relative paths. It writes new results under ignored
`results/reproduction/`; it does not replace the frozen result package:

```bash
.venv/bin/python scripts/run_benchmark_suite.py \
  --manifests \
    datasets/manifests/smartota-alpine-expanded.json \
    datasets/manifests/smartota-alpine-rootfs.json \
    datasets/manifests/smartota-automotive.json \
    datasets/manifests/smartota-av-software.json \
    datasets/manifests/smartota-container-layers.json \
    datasets/manifests/smartota-debian-ubuntu-updates.json \
    datasets/manifests/smartota-linux-images.json \
    datasets/manifests/smartota-smoke.json \
    datasets/manifests/smartota-synthetic-adversarial.json \
  --manifest-base-dir . \
  --policies full_replacement whole_file_bsdiff blockwise_bsdiff copy_only \
    copy_delta backup_safe_copy_delta deployment_aware_greedy rsync_rolling_hash \
  --block-sizes 16384 65536 \
  --seeds 1 \
  --deployment-config-file \
    artifacts/final_deterministic_benchmark_20260717_clean_rerun/deployment_profiles.json \
  --interruption-setting interrupt_025=0.25 \
  --interruption-setting interrupt_050=0.50 \
  --interruption-setting interrupt_075=0.75 \
  --output-dir results/reproduction/broad_16k_64k
```

Run the separately bounded 4 KiB sweep:

```bash
.venv/bin/python scripts/run_benchmark_suite.py \
  --manifests \
    datasets/manifests/smartota-automotive.json \
    artifacts/final_deterministic_benchmark_20260717_clean_rerun/bounded_manifests/smartota-container-yolo-small-only.json \
  --manifest-base-dir . \
  --policies full_replacement whole_file_bsdiff blockwise_bsdiff copy_only \
    copy_delta backup_safe_copy_delta deployment_aware_greedy rsync_rolling_hash \
  --block-sizes 4096 \
  --seeds 1 \
  --deployment-config-file \
    artifacts/final_deterministic_benchmark_20260717_clean_rerun/deployment_profiles.json \
  --interruption-setting interrupt_025=0.25 \
  --interruption-setting interrupt_050=0.50 \
  --interruption-setting interrupt_075=0.75 \
  --output-dir results/reproduction/small_4k
```

Combining the two newly generated JSONL files is a release-time procedure, not
a recovered historical command:

```bash
mkdir -p results/reproduction/combined
awk '1' \
  results/reproduction/broad_16k_64k/benchmark_results.jsonl \
  results/reproduction/small_4k/benchmark_results.jsonl \
  > results/reproduction/combined/combined_benchmark_results.jsonl
```

For a fresh clean rerun, audit against the new repository commit rather than
the historical generation commit:

```bash
.venv/bin/python scripts/audit_result_reproducibility.py \
  results/reproduction/combined/combined_benchmark_results.jsonl \
  --match-current-commit
```

## Table and plot generation

Regenerate reports into a new ignored output directory. Do not point this
command at the frozen extraction or the released `paper/` directory:

```bash
.venv/bin/python scripts/make_paper_reports.py \
  artifacts/final_deterministic_benchmark_20260717_clean_rerun/benchmark_results.jsonl \
  --attempt-ledger \
    artifacts/final_deterministic_benchmark_20260717_clean_rerun/attempt_ledger.jsonl \
  --output-dir results/revalidated-paper \
  --dataset-scope-note \
    "Bounded profile: 16 KiB and 64 KiB cover all five enabled pairs; 4 KiB covers only the Autoware-style and YOLO-small fixtures; deterministic seed 1 only."
```

Expected report material is one Markdown report, 12 table files (CSV and
LaTeX), and 18 plot files (nine PNG/SVG pairs).

## Expected output counts

| Check | Expected value |
| --- | ---: |
| Broad sweep rows | 960 |
| Bounded 4 KiB rows | 192 |
| Combined attempted jobs | 1,152 |
| Combined reported rows | 1,152 |
| Unique configuration keys | 1,152 |
| Completed rows | 1,104 |
| Deployment-incomplete/error rows | 48 |
| Replay-valid rows | 1,152 |
| Replay-invalid rows | 0 |
| A/B-enabled rows | 864 |
| A/B-valid rows | 816 |

The configuration key is the tuple `(dataset_id, pair_id, policy,
deployment_config_name, interruption_setting_name, seed, block_size_bytes)`.
The attempt ledger uses the corresponding dataset, pair, policy, deployment,
seed, block size, and interruption setting encoded in `run_id`.

## Known environment-dependent metrics

Policy operation streams, replay validity, configuration counts, input hashes,
and deterministic modeled resource accounting are the primary reproducibility
targets. Wall-clock `runtime_s` depends on CPU, operating system, filesystem,
Python, and BSDIFF4 implementation/build. Plot bytes and SVG metadata can vary
with Matplotlib, pandas, fonts, and rendering backends even when the underlying
tables agree. Download speed is external to the benchmark. Modeled install and
download times are equations over declared profile rates, not device timing
measurements.

The historical package does not contain `artifact_environment.txt`; do not
claim that release-time package versions reconstruct the exact generation
environment.

## Verification of SHA-256 checksums

Verify every release file listed by the top-level manifest:

```bash
sha256sum -c SHA256SUMS
```

Verify the frozen archive directly:

```bash
printf '%s  %s\n' \
  0fb643d7361a537ab7740b1923168ee46548b8ee3ff4be2a24d509e221a644ef \
  smartota-final-deterministic-benchmark-20260717-clean-rerun.tar.gz | \
  sha256sum -c -
```

To verify the archive's internal checksum manifest without changing the
archive:

```bash
RELEASE_CHECK_DIR="$(mktemp -d)"
tar -xzf smartota-final-deterministic-benchmark-20260717-clean-rerun.tar.gz \
  -C "$RELEASE_CHECK_DIR"
(cd "$RELEASE_CHECK_DIR" && \
  sha256sum -c \
    results/final_deterministic_benchmark_20260717_clean_rerun/SHA256SUMS)
```

The top-level `SHA256SUMS` intentionally excludes itself and `.git/`.
