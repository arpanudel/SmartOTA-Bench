# SmartOTA-Bench Clean Deterministic Rerun Archive Notes

- Result package: `results/final_deterministic_benchmark_20260717_clean_rerun/`
- Generated from clean temporary clone: `/tmp/smartota-clean-rerun-2942cbd-copy`
- Result row commit: `2942cbde2fd344cccea23aabbe8d5bf168610c71`
- Result rows: 1,152
- Replay-valid rows: 1,152
- Completed rows: 1,104
- Deployment-incomplete/error rows: 48
- `git_dirty` rows: 0

Reproducibility audit:

```bash
/home/arpan/OTA/.cursor-tutor/ota_environment/.venv/bin/python scripts/audit_result_reproducibility.py \
  /home/arpan/OTA/.cursor-tutor/ota_environment/results/final_deterministic_benchmark_20260717_clean_rerun/combined_benchmark_results.jsonl \
  --expected-commit 2942cbde2fd344cccea23aabbe8d5bf168610c71
```

Archive policy:

- Keep this `results/` package out of Git.
- Archive externally with the deterministic tar command recorded in
  `paper/REPRODUCIBILITY_PACKAGE_PLAN.md`.
- Preserve `commands_used.md`, `SHA256SUMS`, and this notes file with the
  archived result package.
