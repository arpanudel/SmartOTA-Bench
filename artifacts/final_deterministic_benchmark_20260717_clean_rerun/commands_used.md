# Commands Used

Clean rerun generated from temporary clone `/tmp/smartota-clean-rerun-2942cbd-copy`
at commit `2942cbde2fd344cccea23aabbe8d5bf168610c71`. Dataset artifacts and
ignored result profile files were read from
`/home/arpan/OTA/.cursor-tutor/ota_environment`.

## Broad 16 KiB/64 KiB Benchmark

```bash
/home/arpan/OTA/.cursor-tutor/ota_environment/.venv/bin/python scripts/run_benchmark_suite.py --manifests datasets/manifests/smartota-alpine-expanded.json datasets/manifests/smartota-alpine-rootfs.json datasets/manifests/smartota-automotive.json datasets/manifests/smartota-av-software.json datasets/manifests/smartota-container-layers.json datasets/manifests/smartota-debian-ubuntu-updates.json datasets/manifests/smartota-linux-images.json datasets/manifests/smartota-smoke.json datasets/manifests/smartota-synthetic-adversarial.json --manifest-base-dir /home/arpan/OTA/.cursor-tutor/ota_environment --policies full_replacement whole_file_bsdiff blockwise_bsdiff copy_only copy_delta backup_safe_copy_delta deployment_aware_greedy rsync_rolling_hash --block-sizes 16384 65536 --seeds 1 --deployment-config-file /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260505/deployment_profiles.json --interruption-setting interrupt_025=0.25 --interruption-setting interrupt_050=0.50 --interruption-setting interrupt_075=0.75 --output-dir /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260717_clean_rerun/broad_16k_64k
```

## Small-Fixture 4 KiB Benchmark

```bash
/home/arpan/OTA/.cursor-tutor/ota_environment/.venv/bin/python scripts/run_benchmark_suite.py --manifests datasets/manifests/smartota-automotive.json /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260505/bounded_manifests/smartota-container-yolo-small-only.json --manifest-base-dir /home/arpan/OTA/.cursor-tutor/ota_environment --policies full_replacement whole_file_bsdiff blockwise_bsdiff copy_only copy_delta backup_safe_copy_delta deployment_aware_greedy rsync_rolling_hash --block-sizes 4096 --seeds 1 --deployment-config-file /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260505/deployment_profiles.json --interruption-setting interrupt_025=0.25 --interruption-setting interrupt_050=0.50 --interruption-setting interrupt_075=0.75 --output-dir /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260717_clean_rerun/small_4k
```

## Paper Reports

```bash
/home/arpan/OTA/.cursor-tutor/ota_environment/.venv/bin/python scripts/make_paper_reports.py /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260717_clean_rerun/benchmark_results.jsonl --attempt-ledger /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260717_clean_rerun/attempt_ledger.jsonl --output-dir /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260717_clean_rerun/paper --dataset-scope-note "Bounded profile: all enabled reproducible pairs were included. Block sizes 16384 and 65536 were run for all enabled pairs. Block size 4096 was run only for the smaller Autoware-style perception module and YOLO-small container fixture; Alpine compressed, Alpine normalized/rootfs, and YOLO multiblock skipped 4096 because a rootfs 4096 probe took about 63 seconds for only eight policy rows under one deployment and one interruption setting. Deterministic seed 1 only. Learned methods and the 10-vehicle deployment MVP were not run."
```

## Reproducibility Audit

```bash
/home/arpan/OTA/.cursor-tutor/ota_environment/.venv/bin/python scripts/audit_result_reproducibility.py /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260717_clean_rerun/combined_benchmark_results.jsonl --expected-commit 2942cbde2fd344cccea23aabbe8d5bf168610c71
```
