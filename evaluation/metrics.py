import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path):
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def flatten_result(result):
    flattened = {}

    def visit(prefix, value):
        if prefix == "action_counts":
            flattened["action_count_m"] = value.get("M", 0)
            flattened["action_count_mb"] = value.get("MB", 0)
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_prefix = f"{prefix}_{child_key}" if prefix else child_key
                visit(child_prefix, child_value)
            return
        if isinstance(value, list):
            flattened[prefix] = "; ".join(str(item) for item in value)
            return
        flattened[prefix] = value

    for key, value in result.items():
        visit(key, value)
    return flattened


def write_results(output_dir, results, fieldnames=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "baseline_results.jsonl"
    csv_path = output_dir / "baseline_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")

    flat_results = [flatten_result(dict(result)) for result in results]
    inferred_fieldnames = sorted({key for result in flat_results for key in result})
    if fieldnames is None:
        fieldnames = inferred_fieldnames
    else:
        fieldnames = list(dict.fromkeys([*fieldnames, *inferred_fieldnames]))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_results)

    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
    }
