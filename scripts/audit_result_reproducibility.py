import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def audit_rows(rows, *, expected_commit=None, require_clean=True):
    commit_counts = Counter(row.get("git_commit", "") for row in rows)
    dirty_count = sum(1 for row in rows if _truthy(row.get("git_dirty", False)))
    missing_commit_count = sum(1 for row in rows if not row.get("git_commit"))
    missing_dirty_count = sum(1 for row in rows if "git_dirty" not in row)

    errors = []
    if not rows:
        errors.append("no result rows found")
    if missing_commit_count:
        errors.append(f"{missing_commit_count} row(s) are missing git_commit")
    if missing_dirty_count:
        errors.append(f"{missing_dirty_count} row(s) are missing git_dirty")
    if require_clean and dirty_count:
        errors.append(f"{dirty_count} row(s) were generated from a dirty worktree")
    if expected_commit:
        mismatches = sum(
            1 for row in rows if row.get("git_commit") != expected_commit
        )
        if mismatches:
            errors.append(
                f"{mismatches} row(s) do not match expected commit {expected_commit}"
            )

    return {
        "row_count": len(rows),
        "commit_counts": dict(sorted(commit_counts.items())),
        "dirty_count": dirty_count,
        "missing_commit_count": missing_commit_count,
        "missing_dirty_count": missing_dirty_count,
        "require_clean": require_clean,
        "expected_commit": expected_commit or "",
        "valid": not errors,
        "errors": errors,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit benchmark JSONL result rows for publication reproducibility metadata."
    )
    parser.add_argument("results_jsonl", help="Benchmark JSONL result file to audit")
    parser.add_argument("--expected-commit", default="")
    parser.add_argument(
        "--match-current-commit",
        action="store_true",
        help="Require every row's git_commit to match the current repository HEAD.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Do not fail rows whose git_dirty metadata is true.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    expected_commit = args.expected_commit.strip()
    if args.match_current_commit:
        expected_commit = current_git_commit()
        if not expected_commit:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "errors": ["could not determine current git commit"],
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
            return 1

    summary = audit_rows(
        load_jsonl(args.results_jsonl),
        expected_commit=expected_commit or None,
        require_clean=not args.allow_dirty,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
