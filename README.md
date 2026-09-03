# SmartOTA-Bench

SmartOTA-Bench is a reproducible simulator and benchmark for evaluating
over-the-air update planning policies. It measures whether a structured update
plan reconstructs its target exactly and models whether that plan can be
installed within controlled bandwidth, storage, RAM, flash-write, staging,
checkpoint, interruption, rollback, and A/B-slot constraints.

This release is the artifact associated with the SEC 2026 camera-ready paper
*SmartOTA-Bench: Replay-Correct and Deployment-Aware Benchmarking of OTA Update
Planning*.

SmartOTA-Bench is a simulator, not a production OTA stack. It does not provide
production signing, secure transport, fleet authorization, hardware-qualified
installers, or a safety case.

## What the benchmark evaluates

The artifact compares eight deterministic planners:

- `full_replacement`
- `whole_file_bsdiff`
- `blockwise_bsdiff`
- `copy_only`
- `copy_delta`
- `backup_safe_copy_delta`
- `deployment_aware_greedy`
- `rsync_rolling_hash`

The policies emit structured `KEEP`, `COPY`, `DELTA`, `RAW_INSERT`, `BACKUP`,
`TRUNCATE`, `CHECKPOINT`, `VERIFY`, `COMMIT`, and rollback-related operations.
The benchmark reports transfer size, runtime, modeled install time, peak RAM,
peak persistent storage, flash writes, budget violations, rollback readiness,
interruption recovery, and A/B validity.

`rsync_rolling_hash` is a framework-native, aligned fixed-block
checksum-and-copy baseline. It is not native rsync, does not invoke rsync, and
does not implement rsync's sliding-window protocol.

## Correct replay versus feasible deployment

Replay correctness and modeled deployment feasibility are deliberately
separate:

1. `encoding/replay.py` applies a plan and requires byte-for-byte equality with
   the target artifact. This is the correctness contract.
2. `deployment/semantics.py` simulates resource use and lifecycle behavior
   under a selected profile. A replay-correct plan can still be infeasible due
   to RAM, storage, staging, backup, checkpoint, flash-write, or inactive-slot
   constraints.

The deployment profiles are controlled experimental envelopes, not
measurements of named electronic control units (ECUs), vehicles, or production
devices. Modeled feasibility is not hardware qualification.

## Exact 1,152-configuration construction

The frozen deterministic result set is the union of two disjoint sweeps:

- Broad sweep: 5 artifact pairs x 2 block sizes (`16384`, `65536`) x 8
  policies x 1 deterministic seed (`1`) x 4 deployment profiles x 3
  interruption settings = **960 rows**.
- Bounded 4 KiB sweep: 2 small-fixture pairs x 1 block size (`4096`) x 8
  policies x 1 deterministic seed (`1`) x 4 deployment profiles x 3
  interruption settings = **192 rows**.

Thus, `960 + 192 = 1,152` attempted and reported configurations. The five
artifact pairs cover a real compressed Alpine minirootfs transition, a derived
normalized Alpine rootfs transition, a deterministic Autoware-style fixture,
and small and multiblock deterministic YOLO/container-style fixtures. The 4 KiB
sweep includes only the Autoware-style and small YOLO-style pairs; it is not an
all-artifact block-size sweep.

## Expected aggregate outcomes

The frozen ledger and result files report:

- 1,152 attempted jobs and 1,152 result rows;
- 1,152 byte-for-byte replay-valid rows;
- 1,104 completed deployments and 48 deployment-incomplete rows;
- 864 A/B-enabled rows, of which 816 are A/B-valid;
- 48 replay-valid Alpine normalized-rootfs rows that are A/B-invalid under the
  severe edge profile because its inactive slot is too small;
- `whole_file_bsdiff` as byte-optimal by median transfer
  (1,097,106 bytes); and
- `backup_safe_copy_delta` as the safety/rollback and interruption-recovery
  winner (rollback-ready rate 1.0; recovery failure rate 0.0417).

These are properties of this bounded deterministic experiment, not universal
rankings of OTA algorithms.

## Artifact provenance

- Result-generation source commit:
  `2942cbde2fd344cccea23aabbe8d5bf168610c71`
- Paper-evidence promotion commit:
  `f2de5a2dccf15df89b96e4ca7ab0efc0775f43dc`
- Release tag identifier:
  `smartota-bench-sec2026-artifact-v1.0.0`
- Frozen archive:
  `smartota-final-deterministic-benchmark-20260717-clean-rerun.tar.gz`
- Frozen archive SHA-256:
  `0fb643d7361a537ab7740b1923168ee46548b8ee3ff4be2a24d509e221a644ef`

The generation and promotion commits are distinct provenance anchors. Source
needed for the deterministic benchmark was exported from the generation
commit. The historical result archive was copied unchanged; it was not
recomputed to fill documentation gaps.

## Setup and reproduction

Create a Python 3.12 environment and install the pinned release dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
```

See [REPRODUCING.md](REPRODUCING.md) for data preparation, the two benchmark
sweeps, result audits, report generation, expected counts, and checksum
verification. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the boundary
between the historical record and release-time validation.

## Release assets

- `smartota-final-deterministic-benchmark-20260717-clean-rerun.tar.gz` is the
  immutable historical result package and primary release asset.
- `artifacts/final_deterministic_benchmark_20260717_clean_rerun/` is an exact
  extraction supplied for inspection and audit convenience.
- `paper/` contains the generated tables, plots, and report associated with the
  camera-ready artifact. It does not claim to contain the published manuscript.
- `datasets/`, `scripts/`, `encoding/`, `deployment/`, `evaluation/`, and `env/`
  contain the curated deterministic benchmark source; `tests/` contains its
  focused test suite.
- `SHA256SUMS` covers every regular release file except `SHA256SUMS` itself and
  Git administrative files.

## Interpretation limitations

Runtime and plotting bytes can vary with the CPU, operating system, Python,
BSDIFF4, NumPy, pandas, and Matplotlib versions. Install-time, bandwidth,
memory, storage, flash-write, rollback, interruption, and A/B figures are
simulator outputs under declared profiles; they are not measurements from named
ECUs. Only seed 1 was used because the reported policies are deterministic.
Synthetic Autoware- and YOLO-style fixtures contain no production Autoware code
or model weights. The release does not establish production readiness, fleet
safety, cryptographic authenticity, or superiority of learned planners.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Until a DOI exists, cite
the SEC 2026 paper by title and authors and cite software version `1.0.0` with
the repository URL. No placeholder DOI, ORCID, pages, or proceedings identifiers
are supplied.

The MIT License applies to original SmartOTA-Bench source code in this release.
Third-party dependencies, upstream artifacts, and downloaded Alpine content
remain under their respective licenses and terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
