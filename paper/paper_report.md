# SmartOTA-Bench Paper Report

Rows analyzed: 1152.
Datasets: 4, pairs: 5, policies: 8, deployment configs: 4, seeds: 1.

## Headline Results

- Lowest mean transfer size: `whole_file_bsdiff` at 2.0 MiB.
- Lowest mean flash writes: `copy_only` at 2.2 MiB.
- Highest reported-row replay validity rate: `deployment_aware_greedy` at 1.000.

## SmartOTA Comparison

Compared `deployment_aware_greedy` against other policies on 48 matched pair/config/seed groups.

| Metric | Wins | Ties | Losses |
| --- | ---: | ---: | ---: |
| Transfer size | 0 | 2 | 46 |
| Peak storage | 0 | 13 | 35 |
| Peak RAM | 0 | 1 | 47 |
| Install time | 0 | 0 | 48 |
| Flash writes | 0 | 12 | 36 |
| Runtime | 0 | 0 | 48 |

`deployment_aware_greedy` failed to update A to B in 2 matched groups.
`deployment_aware_greedy` was not rollback-ready in 4 matched groups.
`deployment_aware_greedy` had budget violations in 4 matched groups.

## Publication Tables

Unless otherwise stated, policy-level rates are computed over reported JSONL rows, not over all attempted benchmark jobs. Interruption recovery failure rate is computed over simulated interruption scenarios.

- `tables/dataset_summary.csv` and `.tex`: dataset/domain/artifact/scenario/block-size coverage with unique pair counts and artifact size distributions.
- `tables/policy_summary.csv` and `.tex`: policy-level transfer, runtime, RAM, storage, flash-write, rollback, replay, budget, and interruption recovery aggregates.
- `tables/safety_tradeoff.csv` and `.tex`: policy winners for safety, byte efficiency, runtime, and recovery.
- `tables/attempted_vs_reported_runs.csv` and `.tex`: reported-row accounting and any available attempted/timeout/excluded accounting.
- `tables/operation_mix.csv` and `.tex`: normalized operation counts and percentages by dataset, domain, and policy.

Dataset compression status is inferred from artifact path suffixes or artifact type when manifest-ground-truth compression metadata is unavailable. Operation percentages are computed over all recorded operations; `other_pct` captures operations outside the displayed publication categories, such as `truncate`.

Safety tradeoff criteria: safest means highest rollback readiness rate, then lowest budget violation rate, lowest interruption recovery failure rate, and highest replay validity rate. Smallest transfer and fastest runtime are selected by lowest median value among replay-valid runs. Best recovery means lowest interruption recovery failure rate, then highest rollback readiness rate.

Using deterministic tie-breaking after those criteria, `backup_safe_copy_delta` is the primary safety-optimal policy, `whole_file_bsdiff` is byte-optimal, `copy_only` is fastest, and `backup_safe_copy_delta` has the best recovery profile. The byte-optimal policy differs from the primary safety-optimal policy in this run.

Safety-optimal category: `backup_safe_copy_delta` is the unique winner under the stated criteria.
Best-recovery category: `backup_safe_copy_delta` is the unique winner under the stated criteria.

## A/B Slot Interpretation

A/B slot semantics were enabled for 864 reported row(s). Replay validity remains the byte-for-byte reconstruction contract; A/B validity is deployment-specific and can fail because of inactive-slot capacity, activation, boot health, or rollback behavior.
- A/B update validity rate across A/B rows: 0.944.
- A/B rollback-ready rate across A/B rows: 1.000.
- Boot health success rate across A/B rows: 0.944.
- Slot capacity violation rate across A/B rows: 0.056.
- Highest policy-level A/B update validity: `deployment_aware_greedy` at 0.944.

## Generated Artifacts

- `tables/per_run_metrics.csv`: every run with strategy, A-to-B success, correctness, deployment, interruption, and runtime fields.
- `tables/dataset_summary.csv`, `tables/policy_summary.csv`, `tables/safety_tradeoff.csv`, `tables/attempted_vs_reported_runs.csv`, and `tables/operation_mix.csv`: publication table CSVs with matching LaTeX outputs.
- `tables/best_policy_by_metric.csv`: best policy by metric for each matched pair/config/seed.
- `plots/*.png` and `plots/*.svg`: paper-ready figures.

## Dataset Scope

Bounded profile: all enabled reproducible pairs were included. Block sizes 16384 and 65536 were run for all enabled pairs. Block size 4096 was run only for the smaller Autoware-style perception module and YOLO-small container fixture; Alpine compressed, Alpine normalized/rootfs, and YOLO multiblock skipped 4096 because a rootfs 4096 probe took about 63 seconds for only eight policy rows under one deployment and one interruption setting. Deterministic seed 1 only. Learned methods and the 10-vehicle deployment MVP were not run.
