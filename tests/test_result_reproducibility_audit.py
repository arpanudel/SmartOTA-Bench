import unittest

from scripts import audit_result_reproducibility as audit


def result_row(commit="abc123", dirty=False):
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "policy": "copy_delta",
    }


class ResultReproducibilityAuditTests(unittest.TestCase):
    def test_clean_rows_for_expected_commit_pass(self):
        summary = audit.audit_rows(
            [result_row(), result_row()],
            expected_commit="abc123",
        )

        self.assertTrue(summary["valid"])
        self.assertEqual(summary["dirty_count"], 0)
        self.assertEqual(summary["commit_counts"], {"abc123": 2})
        self.assertEqual(summary["errors"], [])

    def test_dirty_rows_fail_by_default(self):
        summary = audit.audit_rows([result_row(dirty=True)])

        self.assertFalse(summary["valid"])
        self.assertEqual(summary["dirty_count"], 1)
        self.assertIn(
            "1 row(s) were generated from a dirty worktree",
            summary["errors"],
        )

    def test_dirty_rows_can_be_allowed_for_diagnostics(self):
        summary = audit.audit_rows([result_row(dirty="true")], require_clean=False)

        self.assertTrue(summary["valid"])
        self.assertEqual(summary["dirty_count"], 1)

    def test_expected_commit_mismatch_fails(self):
        summary = audit.audit_rows(
            [result_row(commit="abc123"), result_row(commit="def456")],
            expected_commit="abc123",
        )

        self.assertFalse(summary["valid"])
        self.assertIn(
            "1 row(s) do not match expected commit abc123",
            summary["errors"],
        )

    def test_missing_metadata_fails(self):
        summary = audit.audit_rows([{"policy": "copy_delta"}])

        self.assertFalse(summary["valid"])
        self.assertIn("1 row(s) are missing git_commit", summary["errors"])
        self.assertIn("1 row(s) are missing git_dirty", summary["errors"])

    def test_empty_results_fail(self):
        summary = audit.audit_rows([])

        self.assertFalse(summary["valid"])
        self.assertIn("no result rows found", summary["errors"])


if __name__ == "__main__":
    unittest.main()
