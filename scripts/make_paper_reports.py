import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deployment.semantics import AB_FLAT_METRIC_KEYS


METRIC_COLUMNS = {
    "transfer_size_bytes": "Transfer size",
    "peak_storage_bytes": "Peak storage",
    "peak_ram_bytes": "Peak RAM",
    "install_time_s": "Install time",
    "flash_write_bytes": "Flash writes",
    "runtime_s": "Runtime",
}

PAPER_POLICY_ORDER = [
    "deployment_aware_greedy",
    "backup_safe_copy_delta",
    "copy_delta",
    "copy_only",
    "rsync_rolling_hash",
    "blockwise_bsdiff",
    "whole_file_bsdiff",
    "full_replacement",
    "backup_aware_copy_delta",
    "greedy_smallest_delta",
    "copy_first",
    "sequential_mb",
    "sequential_m",
    "random",
]

PUBLICATION_OPERATIONS = [
    "copy",
    "delta",
    "raw_insert",
    "backup",
    "checkpoint",
    "verify",
    "rollback",
]

OPERATION_ALIASES = {
    "copy": "copy",
    "keep": "copy",
    "modify_copy": "copy",
    "delta": "delta",
    "bsdiff": "delta",
    "modify_delta": "delta",
    "raw_insert": "raw_insert",
    "raw": "raw_insert",
    "insert": "raw_insert",
    "backup": "backup",
    "checkpoint": "checkpoint",
    "verify": "verify",
    "rollback": "rollback",
}

COMPRESSED_SUFFIXES = (
    ".gz",
    ".tgz",
    ".zip",
    ".xz",
    ".bz2",
    ".zst",
    ".7z",
)

UNCOMPRESSED_SUFFIXES = (
    ".bin",
    ".tar",
    ".img",
    ".raw",
    ".rootfs",
)


def _truthy(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_number(value, default=0):
    if value in (None, "", "unknown", "not_recorded"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(numerator, denominator):
    denominator = _as_number(denominator, 0)
    if denominator <= 0:
        return "not_recorded"
    return _as_number(numerator, 0) / denominator


def _numeric_for_sort(value, default=math.inf):
    if value in (None, "", "unknown", "not_recorded"):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def load_results(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_attempt_ledger(path):
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def default_attempt_ledger_path(results_jsonl):
    return Path(results_jsonl).with_name("attempt_ledger.jsonl")


def _deployment(row, key, default=0):
    return row.get("deployment", {}).get(key, default)


def _ab_value(row, key, default=0):
    if key in row:
        return row.get(key, default)
    return _deployment(row, key, default)


def _has_ab_metrics(row):
    deployment = row.get("deployment", {})
    return any(key in row or key in deployment for key in AB_FLAT_METRIC_KEYS)


def _interruption(row, key, default=0):
    return row.get("interruption_summary", {}).get(key, default)


def _operation_mix(row):
    counts = row.get("operation_counts") or {}
    if not counts:
        counts = row.get("action_counts") or {}
    interesting = [
        "keep",
        "copy",
        "delta",
        "raw_insert",
        "append",
        "backup",
        "truncate",
        "checkpoint",
        "verify",
        "commit",
        "M",
        "MB",
    ]
    parts = [f"{name}:{counts[name]}" for name in interesting if counts.get(name)]
    if not parts and counts:
        parts = [f"{name}:{count}" for name, count in sorted(counts.items()) if count]
    return ", ".join(parts)


def _dominant_operation(row):
    counts = row.get("operation_counts") or row.get("action_counts") or {}
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _success(row):
    return bool(
        row.get("completed")
        and row.get("replay_validity")
        and _deployment(row, "install_state", "") == "complete"
    )


def _path_compression_status(path):
    path = str(path or "").lower()
    if not path:
        return "unknown"
    if path.endswith(COMPRESSED_SUFFIXES):
        return "compressed"
    if path.endswith(UNCOMPRESSED_SUFFIXES):
        return "uncompressed"
    return "unknown"


def infer_compression_status_details(row):
    explicit_status = str(row.get("compression_status", "") or "").strip()
    explicit_source = str(row.get("compression_status_source", "") or "").strip()
    if explicit_status:
        return explicit_status, explicit_source or "manifest_field"

    statuses = {
        _path_compression_status(row.get("old_path", row.get("old_file", ""))),
        _path_compression_status(row.get("new_path", row.get("new_file", ""))),
    }
    statuses.discard("unknown")
    if len(statuses) == 1:
        return statuses.pop(), "inferred_from_path_suffix"
    if len(statuses) > 1:
        return "mixed", "inferred_from_path_suffix"

    artifact_type = str(row.get("artifact_type", "")).lower()
    if artifact_type in {"archive", "compressed_archive"}:
        return "compressed", "inferred_from_artifact_type"
    if artifact_type in {"binary", "rootfs_tar", "raw", "image"}:
        return "uncompressed", "inferred_from_artifact_type"
    return "unknown", "not_recorded"


def infer_compression_status(row):
    return infer_compression_status_details(row)[0]


def normalize_operation_name(name):
    return OPERATION_ALIASES.get(str(name).strip(), str(name).strip())


def normalized_operation_counts(row):
    raw_counts = row.get("operation_counts") or {}
    if not raw_counts:
        raw_counts = row.get("action_counts") or {}
    normalized = {operation: 0 for operation in PUBLICATION_OPERATIONS}
    raw_numeric = {}
    for raw_name, raw_count in raw_counts.items():
        count = int(_as_number(raw_count, 0))
        raw_numeric[str(raw_name)] = raw_numeric.get(str(raw_name), 0) + count
        operation = normalize_operation_name(raw_name)
        if operation in normalized:
            normalized[operation] += count
    return normalized, raw_numeric


def flatten_rows(rows):
    flat = []
    include_ab_metrics = any(_has_ab_metrics(row) for row in rows)
    for row in rows:
        normalized_ops, raw_ops = normalized_operation_counts(row)
        compression_status, compression_source = infer_compression_status_details(row)
        payload = {
            "dataset_id": row.get("dataset_id", ""),
            "pair_id": row.get("pair_id", ""),
            "domain": row.get("domain", ""),
            "scenario": row.get("scenario", ""),
            "artifact_type": row.get("artifact_type", ""),
            "old_path": row.get("old_path", row.get("old_file", "")),
            "new_path": row.get("new_path", row.get("new_file", "")),
            "old_sha256": row.get("old_sha256", ""),
            "new_sha256": row.get("new_sha256", ""),
            "old_size_bytes": row.get("old_size_bytes", 0),
            "new_size_bytes": row.get("new_size_bytes", 0),
            "block_size_bytes": row.get("block_size_bytes", row.get("block_size", 0)),
            "compressed_uncompressed": compression_status,
            "compression_status_source": compression_source,
            "deployment_config_name": row.get("deployment_config_name", "default"),
            "interruption_setting_name": row.get("interruption_setting_name", "default"),
            "seed": row.get("seed", 0),
            "policy": row.get("policy", ""),
            "strategy": _operation_mix(row),
            "dominant_operation": _dominant_operation(row),
            "successfully_updates_a_to_b": _success(row),
            "completed": bool(row.get("completed")),
            "replay_validity": bool(row.get("replay_validity")),
            "replay_errors": row.get("replay_errors", ""),
            "rollback_ready": bool(_deployment(row, "rollback_ready", False)),
            "budget_violation_count": _deployment(row, "budget_violation_count", 0),
            "transfer_size_bytes": _deployment(row, "package_size_bytes", 0),
            "network_bytes": _deployment(row, "network_bytes", 0),
            "peak_storage_bytes": _deployment(row, "peak_persistent_storage_bytes", 0),
            "peak_ram_bytes": _deployment(row, "peak_ram_bytes", 0),
            "install_time_s": _deployment(row, "install_time_s", 0.0),
            "total_time_s": _deployment(row, "total_time_s", 0.0),
            "flash_write_bytes": _deployment(row, "flash_write_bytes", 0),
            "runtime_s": row.get("runtime_s", row.get("duration_s", 0.0)),
            "encoding_op_count": row.get(
                "encoding_op_count",
                sum(normalized_ops.values()),
            ),
            "interruption_scenario_count": _interruption(row, "scenario_count", 0),
            "checkpoint_resume_count": _interruption(row, "checkpoint_resume_count", 0),
            "rollback_success_count": _interruption(row, "rollback_success_count", 0),
            "failed_recovery_count": _interruption(row, "failed_recovery_count", 0),
            "final_replay_validity_all": bool(_interruption(row, "final_replay_validity_all", True)),
            "max_recovery_cost_operations": _interruption(row, "max_recovery_cost_operations", 0),
            "max_extra_network_bytes": _interruption(row, "max_extra_network_bytes", 0),
            "max_extra_flash_writes": _interruption(row, "max_extra_flash_writes", 0),
            "operation_counts_raw": raw_ops,
            **{f"op_{name}_count": normalized_ops[name] for name in PUBLICATION_OPERATIONS},
        }
        if include_ab_metrics:
            payload.update(
                {
                    "ab_enabled": bool(_ab_value(row, "ab_enabled", False)),
                    "ab_update_valid": bool(_ab_value(row, "ab_update_valid", False)),
                    "ab_rollback_ready": bool(_ab_value(row, "ab_rollback_ready", False)),
                    "slot_storage_bytes": _as_number(_ab_value(row, "slot_storage_bytes", 0), 0),
                    "slot_storage_violation": bool(_ab_value(row, "slot_storage_violation", False)),
                    "activation_success": bool(_ab_value(row, "activation_success", False)),
                    "boot_health_success": bool(_ab_value(row, "boot_health_success", False)),
                    "rollback_after_failed_boot": bool(_ab_value(row, "rollback_after_failed_boot", False)),
                    "rollback_success": bool(_ab_value(row, "rollback_success", False)),
                    "reboot_count": _as_number(_ab_value(row, "reboot_count", 0), 0),
                    "downtime_seconds": _as_number(_ab_value(row, "downtime_seconds", 0.0), 0.0),
                    "slot_switch_count": _as_number(_ab_value(row, "slot_switch_count", 0), 0),
                }
            )
        flat.append(payload)
    return pd.DataFrame(flat)


def policy_sort_key(policy):
    if policy in PAPER_POLICY_ORDER:
        return (PAPER_POLICY_ORDER.index(policy), policy)
    return (len(PAPER_POLICY_ORDER), policy)


def _sort_by_policy(df):
    return df.sort_values("policy", key=lambda col: col.map(policy_sort_key))


def _bool_mean(series):
    if len(series) == 0:
        return 0.0
    return float(series.map(_truthy).mean())


def _sum_series(series):
    return sum(_as_number(value, 0) for value in series)


def build_dataset_summary_table(df):
    group_cols = [
        "dataset_id",
        "domain",
        "artifact_type",
        "compressed_uncompressed",
        "compression_status_source",
        "block_size_bytes",
        "scenario",
    ]
    pair_cols = [
        "dataset_id",
        "domain",
        "artifact_type",
        "compressed_uncompressed",
        "compression_status_source",
        "block_size_bytes",
        "scenario",
        "pair_id",
        "old_path",
        "new_path",
    ]
    pairs = df.drop_duplicates(pair_cols)
    summary = pairs.groupby(group_cols, dropna=False).agg(
        pairs=("pair_id", "nunique"),
        old_size_mean_bytes=("old_size_bytes", "mean"),
        old_size_median_bytes=("old_size_bytes", "median"),
        old_size_min_bytes=("old_size_bytes", "min"),
        old_size_max_bytes=("old_size_bytes", "max"),
        new_size_mean_bytes=("new_size_bytes", "mean"),
        new_size_median_bytes=("new_size_bytes", "median"),
        new_size_min_bytes=("new_size_bytes", "min"),
        new_size_max_bytes=("new_size_bytes", "max"),
    ).reset_index()
    return summary.sort_values(group_cols).reset_index(drop=True)


def build_policy_summary_table(df):
    rows = []
    has_ab_metrics = "ab_enabled" in df.columns and df["ab_enabled"].map(_truthy).any()
    for policy, group in df.groupby("policy", dropna=False):
        scenario_count = _sum_series(group["interruption_scenario_count"])
        failed_recoveries = _sum_series(group["failed_recovery_count"])
        budget_violation_rows = group["budget_violation_count"].map(lambda value: _as_number(value, 0) > 0)
        payload = {
            "policy": policy,
            "runs": len(group),
            "success_rate": _bool_mean(group["successfully_updates_a_to_b"]),
            "replay_validity_rate": _bool_mean(group["replay_validity"]),
            "rollback_ready_rate": _bool_mean(group["rollback_ready"]),
            "mean_transfer_size_bytes": group["transfer_size_bytes"].mean(),
            "mean_peak_storage_bytes": group["peak_storage_bytes"].mean(),
            "mean_peak_ram_bytes": group["peak_ram_bytes"].mean(),
            "mean_install_time_s": group["install_time_s"].mean(),
            "mean_flash_write_bytes": group["flash_write_bytes"].mean(),
            "mean_budget_violations": group["budget_violation_count"].mean(),
            "mean_failed_recoveries": group["failed_recovery_count"].mean(),
            "mean_runtime_s": group["runtime_s"].mean(),
            "median_transfer_size_bytes": group["transfer_size_bytes"].median(),
            "mean_network_bytes": group["network_bytes"].mean(),
            "median_network_bytes": group["network_bytes"].median(),
            "median_runtime_s": group["runtime_s"].median(),
            "median_peak_ram_bytes": group["peak_ram_bytes"].median(),
            "median_peak_storage_bytes": group["peak_storage_bytes"].median(),
            "median_flash_write_bytes": group["flash_write_bytes"].median(),
            "median_install_time_s": group["install_time_s"].median(),
            "budget_violation_rate": float(budget_violation_rows.mean()) if len(group) else 0.0,
            "interruption_recovery_failure_rate": _rate(failed_recoveries, scenario_count),
        }
        if has_ab_metrics:
            ab_group = group[group["ab_enabled"].map(_truthy)]
            ab_runs = len(ab_group)
            payload.update(
                {
                    "ab_enabled_runs": ab_runs,
                    "ab_update_valid_rate": _bool_mean(ab_group["ab_update_valid"]) if ab_runs else 0.0,
                    "ab_rollback_ready_rate": _bool_mean(ab_group["ab_rollback_ready"]) if ab_runs else 0.0,
                    "activation_success_rate": _bool_mean(ab_group["activation_success"]) if ab_runs else 0.0,
                    "boot_health_success_rate": _bool_mean(ab_group["boot_health_success"]) if ab_runs else 0.0,
                    "rollback_after_failed_boot_rate": (
                        _bool_mean(ab_group["rollback_after_failed_boot"]) if ab_runs else 0.0
                    ),
                    "rollback_success_rate": _bool_mean(ab_group["rollback_success"]) if ab_runs else 0.0,
                    "slot_storage_violation_rate": (
                        _bool_mean(ab_group["slot_storage_violation"]) if ab_runs else 0.0
                    ),
                    "mean_slot_storage_bytes": ab_group["slot_storage_bytes"].mean() if ab_runs else 0.0,
                    "mean_reboot_count": ab_group["reboot_count"].mean() if ab_runs else 0.0,
                    "mean_downtime_seconds": ab_group["downtime_seconds"].mean() if ab_runs else 0.0,
                    "mean_slot_switch_count": ab_group["slot_switch_count"].mean() if ab_runs else 0.0,
                }
            )
        rows.append(payload)
    summary = pd.DataFrame(rows)
    return _sort_by_policy(summary).reset_index(drop=True)


def _winner_from_summary(summary, sort_specs):
    # sort_specs is a sequence of (column, ascending, missing_default).  This
    # makes the policy selection criteria explicit and keeps ties deterministic.
    ranked = summary.copy()
    sort_cols = []
    ascending = []
    for index, (column, asc, missing_default) in enumerate(sort_specs):
        sort_column = f"_sort_{index}_{column}"
        ranked[sort_column] = ranked[column].map(
            lambda value: _numeric_for_sort(value, missing_default)
        )
        sort_cols.append(sort_column)
        ascending.append(asc)
    ranked["_policy_order"] = ranked["policy"].map(policy_sort_key)
    ranked = ranked.sort_values(sort_cols + ["_policy_order"], ascending=ascending + [True])
    return ranked.iloc[0]


def _winners_from_summary(summary, sort_specs):
    tied = summary.copy()
    for column, asc, missing_default in sort_specs:
        scores = tied[column].map(lambda value: _numeric_for_sort(value, missing_default))
        best = scores.min() if asc else scores.max()
        tied = tied[scores == best]
    tied = _sort_by_policy(tied).reset_index(drop=True)
    return tied.iloc[0], tied


def _supporting_metrics(policy, summary):
    row = summary[summary["policy"] == policy].iloc[0]
    return {
        "rollback_ready_rate": row["rollback_ready_rate"],
        "budget_violation_rate": row["budget_violation_rate"],
        "interruption_recovery_failure_rate": row["interruption_recovery_failure_rate"],
        "replay_validity_rate": row["replay_validity_rate"],
        "median_transfer_size_bytes": row["median_transfer_size_bytes"],
        "median_runtime_s": row["median_runtime_s"],
    }


def build_safety_tradeoff_table(df, policy_summary):
    valid = df[df["replay_validity"].map(_truthy)]
    valid_summary = build_policy_summary_table(valid) if not valid.empty else policy_summary

    # Safest prioritizes rollback readiness, then absence of budget violations,
    # then interruption recovery, then replay validity.  Transfer/runtime are
    # intentionally not part of this safety-first criterion.
    safety_specs = [
        ("rollback_ready_rate", False, -math.inf),
        ("budget_violation_rate", True, math.inf),
        ("interruption_recovery_failure_rate", True, math.inf),
        ("replay_validity_rate", False, -math.inf),
    ]
    safest, safest_ties = _winners_from_summary(
        policy_summary,
        safety_specs,
    )
    # Smallest/fastest are selected among replay-valid runs only.
    smallest_specs = [
        ("median_transfer_size_bytes", True, math.inf),
        ("replay_validity_rate", False, -math.inf),
    ]
    smallest, smallest_ties = _winners_from_summary(
        valid_summary,
        smallest_specs,
    )
    fastest_specs = [
        ("median_runtime_s", True, math.inf),
        ("replay_validity_rate", False, -math.inf),
    ]
    fastest, fastest_ties = _winners_from_summary(
        valid_summary,
        fastest_specs,
    )
    # Best recovery minimizes interruption recovery failures, then prefers
    # rollback readiness for interrupted-device recovery.
    recovery_specs = [
        ("interruption_recovery_failure_rate", True, math.inf),
        ("rollback_ready_rate", False, -math.inf),
        ("replay_validity_rate", False, -math.inf),
    ]
    best_recovery, best_recovery_ties = _winners_from_summary(
        policy_summary,
        recovery_specs,
    )

    selections = [
        ("safest_policy", safest["policy"], safest_ties, "highest rollback_ready_rate, then lowest budget_violation_rate, lowest interruption_recovery_failure_rate, highest replay_validity_rate"),
        ("smallest_transfer_policy", smallest["policy"], smallest_ties, "lowest median_transfer_size_bytes among replay-valid runs"),
        ("fastest_policy", fastest["policy"], fastest_ties, "lowest median_runtime_s among replay-valid runs"),
        ("best_recovery_policy", best_recovery["policy"], best_recovery_ties, "lowest interruption_recovery_failure_rate, then highest rollback_ready_rate"),
    ]
    rows = []
    for criterion, policy, tied, rule in selections:
        rows.append(
            {
                "criterion": criterion,
                "winning_policy": policy,
                "tied_policies": ", ".join(tied["policy"]),
                "num_tied_policies": len(tied),
                "tie_breaker": "PAPER_POLICY_ORDER" if len(tied) > 1 else "none",
                "selection_rule": rule,
                **_supporting_metrics(policy, policy_summary),
            }
        )
    return pd.DataFrame(rows)


def _has_any_field(rows, field_names):
    return any(any(field in row for field in field_names) for row in rows)


def build_attempted_vs_reported_table(rows, df, attempt_ledger_rows=None):
    if attempt_ledger_rows is not None:
        statuses = [str(row.get("status", "")) for row in attempt_ledger_rows]
        status_counts = {
            status: sum(1 for value in statuses if value == status)
            for status in (
                "attempted",
                "completed",
                "timeout",
                "failed_replay",
                "invalid_replay",
                "error",
                "excluded",
            )
        }
        result_rows_written = sum(
            1
            for row in attempt_ledger_rows
            if _truthy(row.get("result_row_written"))
        )
        return pd.DataFrame(
            [
                {
                    "scope": "attempt_ledger_jsonl",
                    "attempted": len(attempt_ledger_rows),
                    "reported": len(df),
                    "completed": status_counts["completed"],
                    "timeout": status_counts["timeout"],
                    "failed_replay": status_counts["failed_replay"],
                    "invalid_replay": status_counts["invalid_replay"],
                    "error": status_counts["error"],
                    "excluded": status_counts["excluded"],
                    "replay_valid": int(df["replay_validity"].map(_truthy).sum()),
                    "result_rows_written": result_rows_written,
                    "result_rows_match_reported": result_rows_written == len(df),
                    "accounting_note": "Counts are derived from attempt_ledger.jsonl.",
                }
            ]
        )

    attempted_fields = ("attempted", "attempted_runs", "run_attempted")
    timeout_fields = ("timeout", "timeouts", "timeout_rows", "timed_out")
    excluded_fields = ("excluded", "excluded_rows", "learned_reportable")
    has_attempt_accounting = _has_any_field(rows, attempted_fields) or _has_any_field(rows, timeout_fields)
    has_timeout_accounting = _has_any_field(rows, timeout_fields)
    has_excluded_accounting = _has_any_field(rows, excluded_fields) and has_attempt_accounting

    invalid_replay = int((~df["replay_validity"].map(_truthy)).sum())
    failed_replay = int(
        (
            (~df["replay_validity"].map(_truthy))
            | df["replay_errors"].fillna("").astype(str).map(bool)
        ).sum()
    )
    row = {
        "scope": "reported_results_jsonl",
        "attempted": "not_recorded",
        "reported": len(df),
        "completed": int(df["completed"].map(_truthy).sum()),
        "timeout": "not_recorded",
        "failed_replay": failed_replay,
        "invalid_replay": invalid_replay,
        "error": "not_recorded",
        "excluded": "not_recorded",
        "replay_valid": int(df["replay_validity"].map(_truthy).sum()),
        "result_rows_written": "not_recorded",
        "result_rows_match_reported": "not_recorded",
        "accounting_note": (
            "No attempted/timeout/excluded run ledger was found; counts are limited "
            "to rows present in the reported JSONL."
        ),
    }
    if has_attempt_accounting:
        row["attempted"] = sum(int(_truthy(row.get(field))) for row in rows for field in attempted_fields)
    if has_timeout_accounting:
        row["timeout"] = sum(int(_truthy(row.get(field))) for row in rows for field in timeout_fields)
    if has_excluded_accounting:
        row["excluded"] = sum(int(not _truthy(row.get("learned_reportable"))) for row in rows)
    return pd.DataFrame([row])


def build_operation_mix_table(df):
    group_cols = ["dataset_id", "domain", "policy"]
    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        payload = dict(zip(group_cols, group_key))
        payload["runs"] = len(group)
        total_operations = int(_sum_series(group["encoding_op_count"]))
        payload["total_operation_count"] = total_operations
        displayed_count = 0
        for operation in PUBLICATION_OPERATIONS:
            count = int(_sum_series(group[f"op_{operation}_count"]))
            displayed_count += count
            payload[f"{operation}_count"] = count
            payload[f"{operation}_pct"] = count / total_operations if total_operations else 0.0
        other_count = max(0, total_operations - displayed_count)
        payload["other_count"] = other_count
        payload["other_pct"] = other_count / total_operations if total_operations else 0.0

        raw_totals = {}
        for raw_counts in group["operation_counts_raw"]:
            if not isinstance(raw_counts, dict):
                continue
            for raw_name, raw_count in raw_counts.items():
                raw_totals[raw_name] = raw_totals.get(raw_name, 0) + int(_as_number(raw_count, 0))
        for raw_name in sorted(raw_totals):
            payload[f"raw_op_{raw_name}"] = raw_totals[raw_name]
        rows.append(payload)
    operation_mix = pd.DataFrame(rows)
    return operation_mix.sort_values(group_cols).reset_index(drop=True)


def compact_table_for_latex(df, name):
    if name == "dataset_summary":
        compact = df.copy()
        compact["dataset/domain"] = compact["dataset_id"] + " / " + compact["domain"]
        compact = compact[
            [
                "dataset/domain",
                "artifact_type",
                "pairs",
                "old_size_median_bytes",
                "new_size_median_bytes",
                "compressed_uncompressed",
                "compression_status_source",
                "block_size_bytes",
                "scenario",
            ]
        ]
        return compact.rename(
            columns={
                "artifact_type": "artifact",
                "old_size_median_bytes": "old med B",
                "new_size_median_bytes": "new med B",
                "compressed_uncompressed": "compression",
                "compression_status_source": "source",
                "block_size_bytes": "block B",
            }
        )
    if name == "policy_summary":
        columns = [
            "policy",
            "runs",
            "median_network_bytes",
            "median_runtime_s",
            "median_peak_ram_bytes",
            "median_peak_storage_bytes",
            "rollback_ready_rate",
            "replay_validity_rate",
            "interruption_recovery_failure_rate",
        ]
        if "ab_update_valid_rate" in df.columns:
            columns.extend(
                [
                    "ab_enabled_runs",
                    "ab_update_valid_rate",
                    "ab_rollback_ready_rate",
                    "boot_health_success_rate",
                    "slot_storage_violation_rate",
                ]
            )
        compact = df[columns]
        return compact.rename(
            columns={
                "median_network_bytes": "med net B",
                "median_runtime_s": "med runtime s",
                "median_peak_ram_bytes": "med RAM B",
                "median_peak_storage_bytes": "med storage B",
                "rollback_ready_rate": "rollback",
                "replay_validity_rate": "replay",
                "interruption_recovery_failure_rate": "int fail",
                "ab_enabled_runs": "A/B runs",
                "ab_update_valid_rate": "A/B valid",
                "ab_rollback_ready_rate": "A/B rollback",
                "boot_health_success_rate": "A/B health",
                "slot_storage_violation_rate": "slot viol",
            }
        )
    if name == "safety_tradeoff":
        compact = df[
            [
                "criterion",
                "winning_policy",
                "tied_policies",
                "tie_breaker",
                "rollback_ready_rate",
                "budget_violation_rate",
                "interruption_recovery_failure_rate",
                "replay_validity_rate",
                "median_transfer_size_bytes",
                "median_runtime_s",
            ]
        ]
        return compact.rename(
            columns={
                "winning_policy": "primary",
                "tied_policies": "ties",
                "tie_breaker": "tie breaker",
                "rollback_ready_rate": "rollback",
                "budget_violation_rate": "budget viol",
                "interruption_recovery_failure_rate": "int fail",
                "replay_validity_rate": "replay",
                "median_transfer_size_bytes": "med net B",
                "median_runtime_s": "med runtime s",
            }
        )
    if name == "attempted_vs_reported_runs":
        return df[
            [
                "reported",
                "replay_valid",
                "invalid_replay",
                "attempted",
                "timeout",
                "error",
                "excluded",
                "result_rows_written",
            ]
        ].rename(
            columns={
                "reported": "reported rows",
                "replay_valid": "replay-valid rows",
                "invalid_replay": "invalid replay rows",
                "result_rows_written": "ledger-written rows",
            }
        )
    if name == "operation_mix":
        compact = df.copy()
        compact["dataset/domain"] = compact["dataset_id"] + " / " + compact["domain"]
        compact = compact[
            [
                "dataset/domain",
                "policy",
                "runs",
                "copy_pct",
                "delta_pct",
                "raw_insert_pct",
                "backup_pct",
                "checkpoint_pct",
                "verify_pct",
                "rollback_pct",
                "other_pct",
            ]
        ]
        return compact.rename(
            columns={
                "copy_pct": "copy",
                "delta_pct": "delta",
                "raw_insert_pct": "raw",
                "backup_pct": "backup",
                "checkpoint_pct": "ckpt",
                "verify_pct": "verify",
                "rollback_pct": "rollback",
                "other_pct": "other",
            }
        )
    return df


def write_named_table(df, tables_dir, name, caption, label):
    csv_path = tables_dir / f"{name}.csv"
    tex_path = tables_dir / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    compact = compact_table_for_latex(df, name)
    tabular = compact.to_latex(
        index=False,
        float_format="%.3f",
    )
    tex = "\n".join(
        [
            "\\begin{table}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\scriptsize",
            "\\resizebox{\\linewidth}{!}{%",
            tabular.rstrip(),
            "}",
            "\\end{table}",
            "",
        ]
    )
    tex_path.write_text(tex, encoding="utf-8")
    return csv_path, tex_path


def write_table_outputs(df, output_dir, raw_rows=None, attempt_ledger_rows=None):
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    run_table = df.sort_values(
        [
            "dataset_id",
            "pair_id",
            "deployment_config_name",
            "block_size_bytes",
            "seed",
            "policy",
        ]
    )
    run_table.to_csv(tables_dir / "per_run_metrics.csv", index=False)

    dataset_summary = build_dataset_summary_table(df)
    policy_summary = build_policy_summary_table(df)
    safety_tradeoff = build_safety_tradeoff_table(df, policy_summary)
    attempted_vs_reported = build_attempted_vs_reported_table(
        raw_rows or [],
        df,
        attempt_ledger_rows=attempt_ledger_rows,
    )
    operation_mix = build_operation_mix_table(df)

    write_named_table(
        dataset_summary,
        tables_dir,
        "dataset_summary",
        "SmartOTA-Bench dataset summary for the latest benchmark run.",
        "tab:smartota_dataset_summary",
    )
    write_named_table(
        policy_summary,
        tables_dir,
        "policy_summary",
        "SmartOTA-Bench policy summary across enabled local datasets.",
        "tab:smartota_policy_summary",
    )
    write_named_table(
        safety_tradeoff,
        tables_dir,
        "safety_tradeoff",
        "SmartOTA-Bench safety and efficiency tradeoff winners.",
        "tab:smartota_safety_tradeoff",
    )
    write_named_table(
        attempted_vs_reported,
        tables_dir,
        "attempted_vs_reported_runs",
        "SmartOTA-Bench attempted versus reported run accounting.",
        "tab:smartota_attempted_vs_reported",
    )
    write_named_table(
        operation_mix,
        tables_dir,
        "operation_mix",
        "SmartOTA-Bench operation mix by dataset, domain, and policy.",
        "tab:smartota_operation_mix",
    )

    best_rows = []
    metric_specs = [
        ("transfer_size_bytes", True),
        ("peak_storage_bytes", True),
        ("peak_ram_bytes", True),
        ("install_time_s", True),
        ("flash_write_bytes", True),
        ("runtime_s", True),
    ]
    group_cols = [
        "dataset_id",
        "pair_id",
        "deployment_config_name",
        "block_size_bytes",
        "seed",
    ]
    for group_key, group in df[df["successfully_updates_a_to_b"]].groupby(group_cols, dropna=False):
        key_payload = dict(zip(group_cols, group_key))
        for metric, lower_is_better in metric_specs:
            index = group[metric].idxmin() if lower_is_better else group[metric].idxmax()
            row = group.loc[index]
            best_rows.append(
                {
                    **key_payload,
                    "metric": metric,
                    "best_policy": row["policy"],
                    "best_value": row[metric],
                }
            )
    best = pd.DataFrame(best_rows)
    best.to_csv(tables_dir / "best_policy_by_metric.csv", index=False)
    return {
        "dataset_summary": dataset_summary,
        "policy_summary": policy_summary,
        "safety_tradeoff": safety_tradeoff,
        "attempted_vs_reported_runs": attempted_vs_reported,
        "operation_mix": operation_mix,
        "best_policy_by_metric": best,
    }


def _human_bytes(value):
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB"]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _bar_plot(summary, metric, ylabel, output_dir):
    plot_data = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(plot_data["policy"], plot_data[metric], color="#386cb0")
    ax.set_xlabel(ylabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(ylabel + " by policy")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{metric}_by_policy.{suffix}", dpi=300)
    plt.close(fig)


def _rollback_plot(summary, output_dir):
    plot_data = summary.sort_values("rollback_ready_rate")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(plot_data["policy"], plot_data["rollback_ready_rate"], color="#7fc97f")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rollback-ready run fraction")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Rollback readiness by policy")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"rollback_ready_rate_by_policy.{suffix}", dpi=300)
    plt.close(fig)


def _pareto_plot(df, output_dir):
    grouped = df.groupby("policy", dropna=False).agg(
        transfer_size_bytes=("transfer_size_bytes", "mean"),
        flash_write_bytes=("flash_write_bytes", "mean"),
        success_rate=("successfully_updates_a_to_b", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sizes = 80 + 240 * grouped["success_rate"]
    ax.scatter(
        grouped["transfer_size_bytes"],
        grouped["flash_write_bytes"],
        s=sizes,
        alpha=0.75,
        color="#fdc086",
        edgecolors="#333333",
        linewidths=0.6,
    )
    for _, row in grouped.iterrows():
        ax.annotate(row["policy"], (row["transfer_size_bytes"], row["flash_write_bytes"]), fontsize=7)
    ax.set_xlabel("Mean transfer size (bytes)")
    ax.set_ylabel("Mean flash writes (bytes)")
    ax.grid(alpha=0.25)
    ax.set_title("Transfer/flash tradeoff; marker size is success rate")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"pareto_transfer_vs_flash.{suffix}", dpi=300)
    plt.close(fig)


def _heatmap(df, output_dir):
    pivot = df.pivot_table(
        index="policy",
        columns="pair_id",
        values="successfully_updates_a_to_b",
        aggfunc="mean",
        fill_value=0,
    )
    pivot = pivot.loc[sorted(pivot.index, key=policy_sort_key)]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(pivot.columns)), 5.5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Update success rate by policy and pair")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Success rate")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"success_rate_heatmap.{suffix}", dpi=300)
    plt.close(fig)


def write_plots(df, summary, output_dir):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in METRIC_COLUMNS.items():
        _bar_plot(summary, f"mean_{metric}" if f"mean_{metric}" in summary else metric, label, plots_dir)
    _rollback_plot(summary, plots_dir)
    _pareto_plot(df, plots_dir)
    _heatmap(df, plots_dir)


def _comparison_summary(df, policy_name="deployment_aware_greedy"):
    smart = df[df["policy"] == policy_name]
    if smart.empty:
        return [f"No `{policy_name}` rows were present."]
    group_cols = [
        "dataset_id",
        "pair_id",
        "deployment_config_name",
        "block_size_bytes",
        "seed",
    ]
    wins = {metric: 0 for metric in METRIC_COLUMNS}
    ties = {metric: 0 for metric in METRIC_COLUMNS}
    losses = {metric: 0 for metric in METRIC_COLUMNS}
    success_failures = 0
    rollback_failures = 0
    budget_failure_rows = 0
    compared = 0

    for key, group in df.groupby(group_cols, dropna=False):
        smart_rows = group[group["policy"] == policy_name]
        others = group[group["policy"] != policy_name]
        if smart_rows.empty or others.empty:
            continue
        compared += 1
        smart_row = smart_rows.iloc[0]
        if not smart_row["successfully_updates_a_to_b"]:
            success_failures += 1
        if not smart_row["rollback_ready"]:
            rollback_failures += 1
        if smart_row["budget_violation_count"] > 0:
            budget_failure_rows += 1
        valid_others = others[others["successfully_updates_a_to_b"]]
        for metric in METRIC_COLUMNS:
            smart_value = smart_row[metric]
            baseline_best = valid_others[metric].min() if not valid_others.empty else others[metric].min()
            if smart_value < baseline_best:
                wins[metric] += 1
            elif smart_value == baseline_best:
                ties[metric] += 1
            else:
                losses[metric] += 1

    lines = [
        f"Compared `{policy_name}` against other policies on {compared} matched pair/config/seed groups.",
        "",
        "| Metric | Wins | Ties | Losses |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, label in METRIC_COLUMNS.items():
        lines.append(f"| {label} | {wins[metric]} | {ties[metric]} | {losses[metric]} |")
    lines.extend(
        [
            "",
            f"`{policy_name}` failed to update A to B in {success_failures} matched groups.",
            f"`{policy_name}` was not rollback-ready in {rollback_failures} matched groups.",
            f"`{policy_name}` had budget violations in {budget_failure_rows} matched groups.",
        ]
    )
    return lines


def _tradeoff_value(tradeoff, criterion, column):
    rows = tradeoff[tradeoff["criterion"] == criterion]
    if rows.empty:
        return "not_recorded"
    return rows.iloc[0][column]


def _tie_note(tradeoff, criterion, label):
    rows = tradeoff[tradeoff["criterion"] == criterion]
    if rows.empty:
        return f"{label}: not recorded."
    row = rows.iloc[0]
    if int(row["num_tied_policies"]) <= 1:
        return f"{label}: `{row['winning_policy']}` is the unique winner under the stated criteria."
    tied = ", ".join(f"`{policy.strip()}`" for policy in str(row["tied_policies"]).split(","))
    return (
        f"{label}: tied under the stated criteria among {tied}; "
        f"`{row['winning_policy']}` is shown as the primary winner by `{row['tie_breaker']}`."
    )


def write_markdown_report(df, tables, output_dir, skipped_manifest_note=""):
    summary = tables["policy_summary"]
    tradeoff = tables["safety_tradeoff"]
    attempted = tables["attempted_vs_reported_runs"]
    report_path = output_dir / "paper_report.md"
    best_transfer = summary.sort_values("mean_transfer_size_bytes").iloc[0]
    best_flash = summary.sort_values("mean_flash_write_bytes").iloc[0]
    best_replay = summary.sort_values("replay_validity_rate", ascending=False).iloc[0]
    safest_policy = _tradeoff_value(tradeoff, "safest_policy", "winning_policy")
    smallest_policy = _tradeoff_value(tradeoff, "smallest_transfer_policy", "winning_policy")
    fastest_policy = _tradeoff_value(tradeoff, "fastest_policy", "winning_policy")
    best_recovery_policy = _tradeoff_value(tradeoff, "best_recovery_policy", "winning_policy")
    byte_vs_safety = "differs from" if smallest_policy != safest_policy else "matches"
    has_ab_metrics = "ab_enabled" in df.columns and df["ab_enabled"].map(_truthy).any()
    attempted_limited = any(
        attempted.iloc[0].get(column) == "not_recorded"
        for column in ("attempted", "timeout", "excluded")
    )
    lines = [
        "# SmartOTA-Bench Paper Report",
        "",
        f"Rows analyzed: {len(df)}.",
        f"Datasets: {df['dataset_id'].nunique()}, pairs: {df['pair_id'].nunique()}, policies: {df['policy'].nunique()}, deployment configs: {df['deployment_config_name'].nunique()}, seeds: {df['seed'].nunique()}.",
        "",
        "## Headline Results",
        "",
        f"- Lowest mean transfer size: `{best_transfer['policy']}` at {_human_bytes(best_transfer['mean_transfer_size_bytes'])}.",
        f"- Lowest mean flash writes: `{best_flash['policy']}` at {_human_bytes(best_flash['mean_flash_write_bytes'])}.",
        f"- Highest reported-row replay validity rate: `{best_replay['policy']}` at {best_replay['replay_validity_rate']:.3f}.",
        "",
        "## SmartOTA Comparison",
        "",
        *_comparison_summary(df),
        "",
        "## Publication Tables",
        "",
        "Unless otherwise stated, policy-level rates are computed over reported JSONL rows, not over all attempted benchmark jobs. Interruption recovery failure rate is computed over simulated interruption scenarios.",
        "",
        "- `tables/dataset_summary.csv` and `.tex`: dataset/domain/artifact/scenario/block-size coverage with unique pair counts and artifact size distributions.",
        "- `tables/policy_summary.csv` and `.tex`: policy-level transfer, runtime, RAM, storage, flash-write, rollback, replay, budget, and interruption recovery aggregates.",
        "- `tables/safety_tradeoff.csv` and `.tex`: policy winners for safety, byte efficiency, runtime, and recovery.",
        "- `tables/attempted_vs_reported_runs.csv` and `.tex`: reported-row accounting and any available attempted/timeout/excluded accounting.",
        "- `tables/operation_mix.csv` and `.tex`: normalized operation counts and percentages by dataset, domain, and policy.",
        "",
        "Dataset compression status is inferred from artifact path suffixes or artifact type when manifest-ground-truth compression metadata is unavailable. Operation percentages are computed over all recorded operations; `other_pct` captures operations outside the displayed publication categories, such as `truncate`.",
        "",
        "Safety tradeoff criteria: safest means highest rollback readiness rate, then lowest budget violation rate, lowest interruption recovery failure rate, and highest replay validity rate. Smallest transfer and fastest runtime are selected by lowest median value among replay-valid runs. Best recovery means lowest interruption recovery failure rate, then highest rollback readiness rate.",
        "",
        f"Using deterministic tie-breaking after those criteria, `{safest_policy}` is the primary safety-optimal policy, `{smallest_policy}` is byte-optimal, `{fastest_policy}` is fastest, and `{best_recovery_policy}` has the best recovery profile. The byte-optimal policy {byte_vs_safety} the primary safety-optimal policy in this run.",
        "",
        _tie_note(tradeoff, "safest_policy", "Safety-optimal category"),
        _tie_note(tradeoff, "best_recovery_policy", "Best-recovery category"),
        "",
        "## Generated Artifacts",
        "",
        "- `tables/per_run_metrics.csv`: every run with strategy, A-to-B success, correctness, deployment, interruption, and runtime fields.",
        "- `tables/dataset_summary.csv`, `tables/policy_summary.csv`, `tables/safety_tradeoff.csv`, `tables/attempted_vs_reported_runs.csv`, and `tables/operation_mix.csv`: publication table CSVs with matching LaTeX outputs.",
        "- `tables/best_policy_by_metric.csv`: best policy by metric for each matched pair/config/seed.",
        "- `plots/*.png` and `plots/*.svg`: paper-ready figures.",
    ]
    if has_ab_metrics:
        ab_rows = df[df["ab_enabled"].map(_truthy)]
        best_ab = summary.sort_values("ab_update_valid_rate", ascending=False).iloc[0]
        generated_index = lines.index("## Generated Artifacts")
        lines[generated_index:generated_index] = [
            "## A/B Slot Interpretation",
            "",
            f"A/B slot semantics were enabled for {len(ab_rows)} reported row(s). Replay validity remains the byte-for-byte reconstruction contract; A/B validity is deployment-specific and can fail because of inactive-slot capacity, activation, boot health, or rollback behavior.",
            f"- A/B update validity rate across A/B rows: {_bool_mean(ab_rows['ab_update_valid']):.3f}.",
            f"- A/B rollback-ready rate across A/B rows: {_bool_mean(ab_rows['ab_rollback_ready']):.3f}.",
            f"- Boot health success rate across A/B rows: {_bool_mean(ab_rows['boot_health_success']):.3f}.",
            f"- Slot capacity violation rate across A/B rows: {_bool_mean(ab_rows['slot_storage_violation']):.3f}.",
            f"- Highest policy-level A/B update validity: `{best_ab['policy']}` at {best_ab['ab_update_valid_rate']:.3f}.",
            "",
        ]
    if attempted_limited:
        lines.extend(
            [
                "",
                "## Accounting Caution",
                "",
                "The latest run records reported result rows but does not include a raw attempted-run ledger with timeout and exclusion records. The attempted-vs-reported table therefore reports `not_recorded` for attempted, timeout, and excluded rows. Do not interpret the reported replay-valid rows as 100% validity over all attempted jobs.",
            ]
        )
    if skipped_manifest_note:
        lines.extend(["", "## Dataset Scope", "", skipped_manifest_note])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results(args.results_jsonl)
    if not rows:
        raise ValueError(f"no result rows found in {args.results_jsonl}")
    ledger_path = Path(args.attempt_ledger) if args.attempt_ledger else default_attempt_ledger_path(args.results_jsonl)
    attempt_ledger_rows = load_attempt_ledger(ledger_path)
    df = flatten_rows(rows)
    tables = write_table_outputs(
        df,
        output_dir,
        raw_rows=rows,
        attempt_ledger_rows=attempt_ledger_rows,
    )
    write_plots(df, tables["policy_summary"], output_dir)
    report_path = write_markdown_report(
        df,
        tables,
        output_dir,
        skipped_manifest_note=args.dataset_scope_note or "",
    )
    print(f"Wrote paper report: {report_path}")
    print(f"Wrote tables: {output_dir / 'tables'}")
    print(f"Wrote plots: {output_dir / 'plots'}")
    return {
        "report": str(report_path),
        "tables": str(output_dir / "tables"),
        "plots": str(output_dir / "plots"),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create paper-ready SmartOTA-Bench tables and plots from benchmark JSONL."
    )
    parser.add_argument("results_jsonl")
    parser.add_argument("--output-dir", default="results/paper")
    parser.add_argument(
        "--attempt-ledger",
        default=None,
        help="optional attempt_ledger.jsonl path; defaults to sibling of results_jsonl",
    )
    parser.add_argument("--dataset-scope-note", default="")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
