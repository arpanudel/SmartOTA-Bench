import tempfile
import unittest
from pathlib import Path

from scripts import make_paper_reports as reports


def result_row(
    policy="policy_a",
    pair_id="pair_a",
    dataset_id="dataset_a",
    domain="synthetic",
    artifact_type="binary",
    old_path="old.bin",
    new_path="new.bin",
    old_size=100,
    new_size=120,
    block_size=4,
    scenario="scenario_a",
    completed=True,
    replay_validity=True,
    rollback_ready=True,
    budget_violation_count=0,
    failed_recovery_count=0,
    scenario_count=2,
    transfer_size=10,
    runtime_s=1.0,
    operation_counts=None,
    ab_metrics=None,
):
    row = {
        "dataset_id": dataset_id,
        "pair_id": pair_id,
        "domain": domain,
        "artifact_type": artifact_type,
        "old_path": old_path,
        "new_path": new_path,
        "old_sha256": f"old-{pair_id}",
        "new_sha256": f"new-{pair_id}",
        "old_size_bytes": old_size,
        "new_size_bytes": new_size,
        "block_size_bytes": block_size,
        "scenario": scenario,
        "deployment_config_name": "default",
        "interruption_setting_name": "default",
        "seed": 1,
        "policy": policy,
        "completed": completed,
        "replay_validity": replay_validity,
        "replay_valid": replay_validity,
        "replay_errors": "" if replay_validity else "mismatch",
        "runtime_s": runtime_s,
        "duration_s": runtime_s,
        "encoding_op_count": sum((operation_counts or {"copy": 1}).values()),
        "operation_counts": operation_counts or {"copy": 1},
        "action_counts": {},
        "deployment": {
            "install_state": "complete" if completed else "failed",
            "rollback_ready": rollback_ready,
            "budget_violation_count": budget_violation_count,
            "package_size_bytes": transfer_size,
            "network_bytes": transfer_size,
            "peak_persistent_storage_bytes": 50,
            "peak_ram_bytes": 20,
            "install_time_s": 0.5,
            "total_time_s": 0.6,
            "flash_write_bytes": 30,
        },
        "interruption_summary": {
            "scenario_count": scenario_count,
            "failed_recovery_count": failed_recovery_count,
            "checkpoint_resume_count": 0,
            "rollback_success_count": scenario_count - failed_recovery_count,
            "final_replay_validity_all": failed_recovery_count == 0,
        },
    }
    if ab_metrics:
        row.update(ab_metrics)
        row["deployment"].update(ab_metrics)
    return row


class PaperReportTableTests(unittest.TestCase):
    def test_dataset_summary_groups_unique_pairs_and_infers_compression(self):
        rows = [
            result_row(
                policy="policy_a",
                pair_id="pair_1",
                artifact_type="archive",
                old_path="old.tar.gz",
                new_path="new.tar.gz",
                old_size=100,
                new_size=200,
            ),
            result_row(
                policy="policy_b",
                pair_id="pair_1",
                artifact_type="archive",
                old_path="old.tar.gz",
                new_path="new.tar.gz",
                old_size=100,
                new_size=200,
            ),
            result_row(
                pair_id="pair_unknown",
                artifact_type="mystery",
                old_path="",
                new_path="",
                scenario="scenario_unknown",
            ),
        ]
        table = reports.build_dataset_summary_table(reports.flatten_rows(rows))

        compressed = table[table["compressed_uncompressed"] == "compressed"].iloc[0]
        unknown = table[table["compressed_uncompressed"] == "unknown"].iloc[0]
        self.assertEqual(compressed["pairs"], 1)
        self.assertEqual(compressed["compression_status_source"], "inferred_from_path_suffix")
        self.assertEqual(compressed["old_size_mean_bytes"], 100)
        self.assertEqual(compressed["new_size_max_bytes"], 200)
        self.assertEqual(unknown["pairs"], 1)
        self.assertEqual(unknown["compression_status_source"], "not_recorded")

    def test_policy_summary_rates_and_medians(self):
        rows = [
            result_row(
                policy="policy_a",
                transfer_size=100,
                runtime_s=2.0,
                rollback_ready=True,
                replay_validity=True,
                budget_violation_count=0,
                failed_recovery_count=1,
                scenario_count=2,
            ),
            result_row(
                policy="policy_a",
                pair_id="pair_b",
                transfer_size=200,
                runtime_s=4.0,
                rollback_ready=False,
                replay_validity=False,
                budget_violation_count=1,
                failed_recovery_count=1,
                scenario_count=2,
            ),
        ]
        table = reports.build_policy_summary_table(reports.flatten_rows(rows))
        policy = table.iloc[0]

        self.assertEqual(policy["runs"], 2)
        self.assertEqual(policy["median_transfer_size_bytes"], 150)
        self.assertEqual(policy["median_runtime_s"], 3.0)
        self.assertEqual(policy["rollback_ready_rate"], 0.5)
        self.assertEqual(policy["replay_validity_rate"], 0.5)
        self.assertEqual(policy["budget_violation_rate"], 0.5)
        self.assertEqual(policy["interruption_recovery_failure_rate"], 0.5)

    def test_policy_summary_includes_ab_metrics_when_present(self):
        rows = [
            result_row(
                policy="policy_ab",
                ab_metrics={
                    "ab_enabled": True,
                    "ab_update_valid": True,
                    "ab_rollback_ready": True,
                    "slot_storage_bytes": 120,
                    "slot_storage_violation": False,
                    "activation_success": True,
                    "boot_health_success": True,
                    "rollback_after_failed_boot": False,
                    "rollback_success": False,
                    "reboot_count": 1,
                    "downtime_seconds": 2.0,
                    "slot_switch_count": 1,
                },
            ),
            result_row(
                policy="policy_ab",
                pair_id="pair_b",
                completed=False,
                ab_metrics={
                    "ab_enabled": True,
                    "ab_update_valid": False,
                    "ab_rollback_ready": True,
                    "slot_storage_bytes": 140,
                    "slot_storage_violation": False,
                    "activation_success": True,
                    "boot_health_success": False,
                    "rollback_after_failed_boot": True,
                    "rollback_success": True,
                    "reboot_count": 2,
                    "downtime_seconds": 4.0,
                    "slot_switch_count": 2,
                },
            ),
        ]
        df = reports.flatten_rows(rows)
        table = reports.build_policy_summary_table(df)
        policy = table.iloc[0]

        self.assertIn("ab_update_valid_rate", table.columns)
        self.assertEqual(policy["ab_enabled_runs"], 2)
        self.assertEqual(policy["ab_update_valid_rate"], 0.5)
        self.assertEqual(policy["ab_rollback_ready_rate"], 1.0)
        self.assertEqual(policy["boot_health_success_rate"], 0.5)
        self.assertEqual(policy["rollback_success_rate"], 0.5)
        self.assertEqual(policy["mean_reboot_count"], 1.5)
        self.assertEqual(policy["mean_downtime_seconds"], 3.0)

    def test_old_rows_without_ab_metrics_omit_ab_report_columns(self):
        df = reports.flatten_rows([result_row()])
        table = reports.build_policy_summary_table(df)

        self.assertNotIn("ab_enabled", df.columns)
        self.assertNotIn("ab_update_valid_rate", table.columns)

    def test_safety_tradeoff_uses_documented_criteria(self):
        rows = [
            result_row(policy="deployment_aware_greedy", transfer_size=1000, runtime_s=10.0, rollback_ready=True),
            result_row(policy="backup_safe_copy_delta", pair_id="pair_b", transfer_size=900, runtime_s=11.0, rollback_ready=True),
            result_row(policy="imitation_bc", pair_id="pair_c", transfer_size=800, runtime_s=12.0, rollback_ready=True),
            result_row(policy="small", transfer_size=10, runtime_s=1.0, rollback_ready=False),
        ]
        df = reports.flatten_rows(rows)
        summary = reports.build_policy_summary_table(df)
        table = reports.build_safety_tradeoff_table(df, summary)

        winners = dict(zip(table["criterion"], table["winning_policy"]))
        self.assertEqual(winners["safest_policy"], "deployment_aware_greedy")
        self.assertEqual(winners["smallest_transfer_policy"], "small")
        self.assertEqual(winners["fastest_policy"], "small")
        self.assertEqual(winners["best_recovery_policy"], "deployment_aware_greedy")
        safety = table[table["criterion"] == "safest_policy"].iloc[0]
        self.assertEqual(safety["num_tied_policies"], 3)
        self.assertEqual(safety["tie_breaker"], "PAPER_POLICY_ORDER")
        self.assertEqual(
            safety["tied_policies"],
            "deployment_aware_greedy, backup_safe_copy_delta, imitation_bc",
        )

    def test_operation_mix_normalizes_names_and_preserves_raw_counts(self):
        rows = [
            result_row(
                policy="policy_ops",
                operation_counts={
                    "keep": 2,
                    "modify_copy": 3,
                    "bsdiff": 4,
                    "raw": 1,
                    "insert": 1,
                    "backup": 2,
                    "checkpoint": 1,
                    "verify": 1,
                    "rollback": 1,
                    "truncate": 2,
                },
            )
        ]
        table = reports.build_operation_mix_table(reports.flatten_rows(rows))
        row = table.iloc[0]

        self.assertEqual(row["copy_count"], 5)
        self.assertEqual(row["delta_count"], 4)
        self.assertEqual(row["raw_insert_count"], 2)
        self.assertEqual(row["backup_count"], 2)
        self.assertEqual(row["checkpoint_count"], 1)
        self.assertEqual(row["verify_count"], 1)
        self.assertEqual(row["rollback_count"], 1)
        self.assertEqual(row["other_count"], 2)
        self.assertAlmostEqual(row["copy_pct"], 5 / 18)
        self.assertAlmostEqual(row["other_pct"], 2 / 18)
        self.assertEqual(row["raw_op_keep"], 2)
        self.assertEqual(row["raw_op_bsdiff"], 4)

    def test_attempted_vs_reported_marks_absent_attempt_fields_not_recorded(self):
        rows = [
            result_row(policy="valid", replay_validity=True),
            result_row(policy="invalid", pair_id="pair_b", replay_validity=False),
        ]
        df = reports.flatten_rows(rows)
        table = reports.build_attempted_vs_reported_table(rows, df)
        row = table.iloc[0]

        self.assertEqual(row["attempted"], "not_recorded")
        self.assertEqual(row["timeout"], "not_recorded")
        self.assertEqual(row["excluded"], "not_recorded")
        self.assertEqual(row["reported"], 2)
        self.assertEqual(row["completed"], 2)
        self.assertEqual(row["replay_valid"], 1)
        self.assertEqual(row["invalid_replay"], 1)
        self.assertEqual(row["failed_replay"], 1)
        self.assertEqual(row["error"], "not_recorded")
        self.assertEqual(row["result_rows_written"], "not_recorded")

    def test_attempted_vs_reported_uses_attempt_ledger_when_present(self):
        rows = [
            result_row(policy="valid", replay_validity=True),
            result_row(policy="invalid", pair_id="pair_b", replay_validity=False),
        ]
        ledger = [
            {
                "run_id": "1",
                "status": "completed",
                "result_row_written": True,
            },
            {
                "run_id": "2",
                "status": "invalid_replay",
                "result_row_written": True,
            },
            {
                "run_id": "3",
                "status": "timeout",
                "result_row_written": False,
            },
            {
                "run_id": "4",
                "status": "error",
                "result_row_written": False,
            },
            {
                "run_id": "5",
                "status": "excluded",
                "result_row_written": False,
            },
            {
                "run_id": "6",
                "status": "failed_replay",
                "result_row_written": False,
            },
        ]
        table = reports.build_attempted_vs_reported_table(
            rows,
            reports.flatten_rows(rows),
            attempt_ledger_rows=ledger,
        )
        row = table.iloc[0]

        self.assertEqual(row["scope"], "attempt_ledger_jsonl")
        self.assertEqual(row["attempted"], 6)
        self.assertEqual(row["reported"], 2)
        self.assertEqual(row["completed"], 1)
        self.assertEqual(row["invalid_replay"], 1)
        self.assertEqual(row["failed_replay"], 1)
        self.assertEqual(row["timeout"], 1)
        self.assertEqual(row["error"], 1)
        self.assertEqual(row["excluded"], 1)
        self.assertEqual(row["result_rows_written"], 2)
        self.assertTrue(row["result_rows_match_reported"])

    def test_write_table_outputs_emits_compact_latex(self):
        rows = [
            result_row(policy="deployment_aware_greedy", operation_counts={"keep": 1}),
            result_row(policy="backup_safe_copy_delta", pair_id="pair_b", operation_counts={"raw_insert": 1}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            reports.write_table_outputs(reports.flatten_rows(rows), Path(tmp), raw_rows=rows)
            for name in (
                "dataset_summary",
                "policy_summary",
                "safety_tradeoff",
                "attempted_vs_reported_runs",
                "operation_mix",
            ):
                tex = Path(tmp) / "tables" / f"{name}.tex"
                self.assertTrue(tex.exists(), name)
                self.assertIn("\\scriptsize", tex.read_text(encoding="utf-8"))
                self.assertIn("\\resizebox{\\linewidth}{!}{%", tex.read_text(encoding="utf-8"))

    def test_markdown_report_adds_ab_interpretation_only_when_present(self):
        ab_metrics = {
            "ab_enabled": True,
            "ab_update_valid": True,
            "ab_rollback_ready": True,
            "slot_storage_bytes": 120,
            "slot_storage_violation": False,
            "activation_success": True,
            "boot_health_success": True,
            "rollback_after_failed_boot": False,
            "rollback_success": False,
            "reboot_count": 1,
            "downtime_seconds": 2.0,
            "slot_switch_count": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ab"
            df = reports.flatten_rows([result_row(ab_metrics=ab_metrics)])
            tables = reports.write_table_outputs(df, output_dir, raw_rows=[result_row(ab_metrics=ab_metrics)])
            report_path = reports.write_markdown_report(df, tables, output_dir)
            self.assertIn("A/B Slot Interpretation", report_path.read_text(encoding="utf-8"))

            old_output_dir = Path(tmp) / "old"
            old_df = reports.flatten_rows([result_row()])
            old_tables = reports.write_table_outputs(old_df, old_output_dir, raw_rows=[result_row()])
            old_report_path = reports.write_markdown_report(old_df, old_tables, old_output_dir)
            self.assertNotIn("A/B Slot Interpretation", old_report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
