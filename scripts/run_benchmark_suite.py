import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import fields
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_manifest
from deployment.semantics import DeploymentConfig
from evaluation.baselines import (
    DEFAULT_INTERRUPTION_PERCENTAGES,
    PUBLICATION_BASELINE_NAMES,
    available_policy_specs,
    is_supported_policy_spec,
    run_policy,
)
from evaluation.metrics import flatten_result
from scripts.run_baselines import infer_artifact_type, parse_learned_policy


REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _run_git_command(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def git_metadata():
    commit = _run_git_command(["rev-parse", "HEAD"]) or "unknown"
    status = _run_git_command(["status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
    }


def dependency_versions(requirements_path=None):
    requirements_path = requirements_path or PROJECT_ROOT / "requirements.txt"
    versions = {}
    if not requirements_path.exists():
        return versions

    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = REQUIREMENT_NAME_RE.match(line)
        if not match:
            continue
        package_name = match.group(1)
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def runtime_metadata():
    return {
        "python_version": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "dependency_versions": dependency_versions(),
        **git_metadata(),
    }


def parse_json_value(value, label):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return parsed


def _deployment_config_from_dict(raw_config, label):
    valid_keys = {field.name for field in fields(DeploymentConfig)}
    unknown = sorted(set(raw_config) - valid_keys)
    if unknown:
        raise ValueError(f"{label} has unsupported DeploymentConfig key(s): {', '.join(unknown)}")
    return DeploymentConfig(**raw_config)


def parse_named_deployment_config(value):
    if "=" not in value:
        raise ValueError("--deployment-config must use NAME=JSON")
    name, raw_json = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--deployment-config name cannot be empty")
    config_dict = parse_json_value(raw_json, f"deployment config '{name}'")
    return {
        "name": name,
        "config": _deployment_config_from_dict(config_dict, f"deployment config '{name}'"),
        "config_dict": config_dict,
    }


def load_deployment_config_file(path):
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    items = []
    if isinstance(data, dict):
        if "configs" in data:
            data = data["configs"]
        else:
            data = [
                {"name": name, "config": config}
                for name, config in data.items()
            ]
    if not isinstance(data, list):
        raise ValueError("--deployment-config-file must contain a list or mapping")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"deployment config file entry {index} must be an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"deployment config file entry {index} is missing a non-empty name")
        config_dict = item.get("config", {})
        if not isinstance(config_dict, dict):
            raise ValueError(f"deployment config '{name}' config must be an object")
        items.append({
            "name": name,
            "config": _deployment_config_from_dict(config_dict, f"deployment config '{name}'"),
            "config_dict": dict(config_dict),
        })
    return items


def deployment_config_sets(args):
    configs = []
    for path in args.deployment_config_file or []:
        configs.extend(load_deployment_config_file(path))
    for value in args.deployment_config or []:
        configs.append(parse_named_deployment_config(value))
    if not configs:
        configs.append({
            "name": "default",
            "config": DeploymentConfig(),
            "config_dict": {},
        })

    seen = set()
    for item in configs:
        if item["name"] in seen:
            raise ValueError(f"duplicate deployment config name: {item['name']}")
        seen.add(item["name"])
    return configs


def parse_interruption_percentages(raw_percentages, label):
    percentages = []
    if raw_percentages.strip():
        for raw_percentage in raw_percentages.split(","):
            if raw_percentage.strip():
                percentages.append(float(raw_percentage))
    for percentage in percentages:
        if percentage < 0.0 or percentage > 1.0:
            raise ValueError(f"{label} percentages must be between 0.0 and 1.0")
    return percentages


def parse_named_interruption_setting(value):
    if "=" not in value:
        raise ValueError("--interruption-setting must use NAME=PCT[,PCT...]")
    name, raw_percentages = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--interruption-setting name cannot be empty")
    return {
        "name": name,
        "percentages": parse_interruption_percentages(raw_percentages, f"interruption setting '{name}'"),
    }


def interruption_sets(args):
    if args.no_interruption_eval:
        return [{"name": "none", "percentages": []}]
    settings = [
        parse_named_interruption_setting(value)
        for value in args.interruption_setting or []
    ]
    if not settings:
        settings.append({
            "name": "default",
            "percentages": list(DEFAULT_INTERRUPTION_PERCENTAGES),
        })

    seen = set()
    for item in settings:
        if item["name"] in seen:
            raise ValueError(f"duplicate interruption setting name: {item['name']}")
        seen.add(item["name"])
    return settings


def selected_policy_specs(args):
    specs = [(policy_name, policy_name) for policy_name in args.policies]
    specs.extend(parse_learned_policy(value) for value in args.learned_policy or [])
    return specs


def pair_metadata(manifest, pair, block_size):
    metadata = {
        "dataset_id": manifest.name,
        "manifest_path": str(manifest.path),
        "manifest_version": manifest.version,
        "pair_id": pair.id,
        "domain": pair.domain,
        "scenario": pair.scenario,
        "artifact_type": infer_artifact_type(pair),
        "source": pair.source,
        "license_notes": pair.license_notes,
        "manifest_block_size_bytes": pair.block_size_bytes,
        "block_size": block_size,
        "block_size_bytes": block_size,
        "old_path": str(pair.old_path),
        "new_path": str(pair.new_path),
        "old_size_bytes": pair.old_size_bytes,
        "new_size_bytes": pair.new_size_bytes,
        "old_sha256": pair.old_sha256,
        "new_sha256": pair.new_sha256,
        "artifact_hashes": {
            "old_sha256": pair.old_sha256,
            "new_sha256": pair.new_sha256,
        },
    }
    for key in ("tier", "compression_status", "compression_status_source"):
        if key in pair.extra:
            metadata[key] = pair.extra[key]
    return metadata


def selected_block_sizes(args, pair):
    return args.block_sizes if args.block_sizes else [pair.block_size_bytes]


def _utc_now():
    return datetime.now(timezone.utc)


def _attempt_run_id(
    manifest,
    pair,
    display_name,
    deployment_name,
    interruption_name,
    seed,
    block_size,
    index,
):
    return "::".join(
        [
            str(index),
            manifest.name,
            pair.id,
            display_name,
            deployment_name,
            interruption_name,
            str(seed),
            str(block_size),
        ]
    )


def _base_attempt_row(
    run_id,
    manifest,
    pair,
    display_name,
    deployment_name,
    seed,
    block_size,
    started_at,
):
    return {
        "run_id": run_id,
        "dataset": manifest.name,
        "pair_id": pair.id,
        "policy": display_name,
        "deployment_config": deployment_name,
        "seed": seed,
        "block_size": block_size,
        "status": "attempted",
        "start_time": started_at.isoformat(),
        "end_time": "",
        "runtime_seconds": 0.0,
        "result_row_written": False,
        "error_type": "",
        "error_message": "",
    }


def _classify_result_status(result):
    if result.get("replay_errors"):
        return "failed_replay", "", ""
    if result.get("replay_validity") is False or result.get("replay_valid") is False:
        return "invalid_replay", "", ""
    if not result.get("completed", False):
        return "error", "incomplete_run", "policy returned completed=false"
    return "completed", "", ""


def _finalize_attempt_row(row, status, started_at, error_type="", error_message=""):
    ended_at = _utc_now()
    row.update(
        {
            "status": status,
            "end_time": ended_at.isoformat(),
            "runtime_seconds": round((ended_at - started_at).total_seconds(), 6),
            "error_type": error_type or "",
            "error_message": str(error_message) if error_message else "",
        }
    )
    return row


def _public_result(result):
    return {
        key: value
        for key, value in result.items()
        if not str(key).startswith("_attempt_")
    }


def update_attempt_ledger_for_reported_results(ledger_rows, reported_results, skipped_results):
    reported_ids = {
        result.get("_attempt_run_id")
        for result in reported_results
        if result.get("_attempt_run_id")
    }
    skipped_ids = {
        result.get("_attempt_run_id")
        for result in skipped_results
        if result.get("_attempt_run_id")
    }
    for row in ledger_rows:
        if row["run_id"] in reported_ids:
            row["result_row_written"] = True
        elif row["run_id"] in skipped_ids:
            row["status"] = "excluded"
            row["result_row_written"] = False


def run_suite_with_attempt_ledger(args):
    suite_started = datetime.now(timezone.utc).isoformat()
    suite_metadata = runtime_metadata()
    deployment_configs = deployment_config_sets(args)
    interruption_configs = interruption_sets(args)
    manifests = [
        load_manifest(path, base_dir=args.manifest_base_dir)
        for path in args.manifests
    ]

    results = []
    attempt_ledger = []
    attempt_index = 0
    for manifest in manifests:
        print(f"Loaded {len(manifest.pairs)} enabled pair(s) from: {manifest.path}")
        for pair in manifest.pairs:
            for block_size in selected_block_sizes(args, pair):
                for deployment in deployment_configs:
                    for interruption in interruption_configs:
                        for seed in args.seeds:
                            for display_name, runner_policy in selected_policy_specs(args):
                                print(
                                    "Running "
                                    f"manifest={manifest.name} pair={pair.id} policy={display_name} "
                                    f"block_size={block_size} deployment={deployment['name']} "
                                    f"interruptions={interruption['name']} seed={seed}"
                                )
                                attempt_index += 1
                                started_at = _utc_now()
                                run_id = _attempt_run_id(
                                    manifest,
                                    pair,
                                    display_name,
                                    deployment["name"],
                                    interruption["name"],
                                    seed,
                                    block_size,
                                    attempt_index,
                                )
                                ledger_row = _base_attempt_row(
                                    run_id,
                                    manifest,
                                    pair,
                                    display_name,
                                    deployment["name"],
                                    seed,
                                    block_size,
                                    started_at,
                                )
                                run_started = time.perf_counter()
                                try:
                                    result = run_policy(
                                        policy_name=runner_policy,
                                        old_file=pair.old_path,
                                        new_file=pair.new_path,
                                        seed=seed,
                                        max_steps=args.max_steps,
                                        deployment_config=deployment["config"],
                                        block_size=block_size,
                                        interruption_percentages=interruption["percentages"],
                                        device=args.device,
                                    )
                                except TimeoutError as exc:
                                    _finalize_attempt_row(
                                        ledger_row,
                                        "timeout",
                                        started_at,
                                        type(exc).__name__,
                                        exc,
                                    )
                                    attempt_ledger.append(ledger_row)
                                    print(f"Timed out: {run_id}: {exc}")
                                    continue
                                except Exception as exc:
                                    _finalize_attempt_row(
                                        ledger_row,
                                        "error",
                                        started_at,
                                        type(exc).__name__,
                                        exc,
                                    )
                                    attempt_ledger.append(ledger_row)
                                    print(f"Errored: {run_id}: {exc}")
                                    continue
                                suite_runtime_s = round(time.perf_counter() - run_started, 6)
                                result["policy"] = display_name
                                result["baseline_name"] = display_name
                                if runner_policy != display_name:
                                    result["policy_runner"] = runner_policy
                                result["_attempt_run_id"] = run_id
                                result.update(pair_metadata(manifest, pair, block_size))
                                result.update(suite_metadata)
                                result.update({
                                    "suite_started_at_utc": suite_started,
                                    "suite_result_runtime_s": suite_runtime_s,
                                    "deployment_config_name": deployment["name"],
                                    "deployment_config": deployment["config_dict"],
                                    "interruption_setting_name": interruption["name"],
                                    "interruption_percentages": interruption["percentages"],
                                    "seed": seed,
                                    "policy": display_name,
                                    "baseline_name": display_name,
                                })
                                status, error_type, error_message = _classify_result_status(result)
                                _finalize_attempt_row(
                                    ledger_row,
                                    status,
                                    started_at,
                                    error_type,
                                    error_message,
                                )
                                attempt_ledger.append(ledger_row)
                                results.append(result)
    return results, attempt_ledger


def run_suite(args):
    results, _ = run_suite_with_attempt_ledger(args)
    return results


def write_jsonl(path, results):
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(_public_result(result), sort_keys=True) + "\n")


def write_csv(path, results):
    flat_results = [flatten_result(_public_result(dict(result))) for result in results]
    fieldnames = sorted({key for result in flat_results for key in result})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_results)


def write_attempt_ledger(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_learned_result(result):
    return (
        result.get("policy_family") == "imitation"
        or str(result.get("policy_runner", "")).startswith("imitation:")
    )


def _comparison_key(result):
    return (
        result.get("dataset_id"),
        result.get("pair_id"),
        result.get("block_size_bytes"),
        result.get("deployment_config_name"),
        result.get("interruption_setting_name"),
        result.get("seed"),
    )


def _deployment_value(result, key, default=0):
    return result.get("deployment", {}).get(key, default)


def learned_comparison_reason(learned_result, baseline_results):
    if not baseline_results:
        return ""
    if not learned_result.get("completed") or not learned_result.get("replay_validity"):
        return ""

    if any(not row.get("completed") or not row.get("replay_validity") for row in baseline_results):
        return "valid_replay"

    lower_is_better = [
        ("package_size_bytes", lambda row: _deployment_value(row, "package_size_bytes")),
        ("flash_write_bytes", lambda row: _deployment_value(row, "flash_write_bytes")),
        ("budget_violation_count", lambda row: _deployment_value(row, "budget_violation_count")),
        ("runtime_s", lambda row: row.get("runtime_s", 0.0)),
        ("peak_ram_bytes", lambda row: _deployment_value(row, "peak_ram_bytes")),
        (
            "peak_persistent_storage_bytes",
            lambda row: _deployment_value(row, "peak_persistent_storage_bytes"),
        ),
    ]
    for metric_name, getter in lower_is_better:
        learned_value = getter(learned_result)
        baseline_best = min(getter(row) for row in baseline_results)
        if learned_value < baseline_best:
            return f"improves_{metric_name}"

    if (
        learned_result.get("deployment", {}).get("rollback_ready")
        and any(not row.get("deployment", {}).get("rollback_ready") for row in baseline_results)
    ):
        return "complements_rollback_ready"
    return ""


def filter_reported_learned_results(results, report_all_learned=False):
    deterministic_by_key = defaultdict(list)
    for result in results:
        if not _is_learned_result(result):
            deterministic_by_key[_comparison_key(result)].append(result)

    filtered = []
    skipped = []
    for result in results:
        if not _is_learned_result(result):
            result["learned_reportable"] = False
            filtered.append(result)
            continue
        reason = learned_comparison_reason(
            result,
            deterministic_by_key.get(_comparison_key(result), []),
        )
        result["learned_reportable"] = bool(reason)
        result["learned_comparison_reason"] = reason
        if report_all_learned or reason:
            filtered.append(result)
        else:
            skipped.append(result)
    return filtered, skipped


def markdown_summary(results):
    grouped = defaultdict(list)
    for result in results:
        key = (
            result["policy"],
            result["block_size_bytes"],
            result["deployment_config_name"],
            result["interruption_setting_name"],
        )
        grouped[key].append(result)

    lines = [
        "# SmartOTA-Bench Suite Summary",
        "",
        "| Policy | Block Size | Deployment Config | Interruption Setting | Runs | Replay Valid | Complete | Avg Runtime (s) | Avg Network Bytes | Avg Flash Writes | Max Peak RAM | Max Peak Storage | Failed Recoveries |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(grouped):
        policy, block_size, deployment_name, interruption_name = key
        rows = grouped[key]
        valid_count = sum(1 for row in rows if row.get("replay_validity"))
        completed_count = sum(1 for row in rows if row.get("completed"))
        avg_runtime = _mean(row.get("runtime_s", 0.0) for row in rows)
        avg_network = _mean(row.get("deployment", {}).get("network_bytes", 0) for row in rows)
        avg_flash = _mean(row.get("deployment", {}).get("flash_write_bytes", 0) for row in rows)
        max_peak_ram = max(row.get("deployment", {}).get("peak_ram_bytes", 0) for row in rows)
        max_peak_storage = max(row.get("deployment", {}).get("peak_persistent_storage_bytes", 0) for row in rows)
        failed_recoveries = sum(
            row.get("interruption_summary", {}).get("failed_recovery_count", 0)
            for row in rows
        )
        lines.append(
            f"| {policy} | {block_size} | {deployment_name} | {interruption_name} | "
            f"{len(rows)} | {valid_count} | {completed_count} | {avg_runtime:.6f} | "
            f"{avg_network:.1f} | {avg_flash:.1f} | {max_peak_ram} | {max_peak_storage} | "
            f"{failed_recoveries} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(output_dir, results, attempt_ledger_rows=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": output_dir / "benchmark_results.jsonl",
        "csv": output_dir / "benchmark_results.csv",
        "markdown": output_dir / "benchmark_summary.md",
        "attempt_ledger": output_dir / "attempt_ledger.jsonl",
    }
    write_jsonl(paths["jsonl"], results)
    write_csv(paths["csv"], results)
    paths["markdown"].write_text(markdown_summary(results), encoding="utf-8")
    write_attempt_ledger(paths["attempt_ledger"], attempt_ledger_rows or [])
    return {key: str(value) for key, value in paths.items()}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SmartOTA-Bench matrix suites across manifests, policies, configs, and seeds."
    )
    parser.add_argument("--manifests", nargs="+", required=True, help="dataset manifest paths to run")
    parser.add_argument(
        "--manifest-base-dir",
        default=None,
        help="override base_dir for all selected manifests",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=PUBLICATION_BASELINE_NAMES,
    )
    parser.add_argument(
        "--learned-policy",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="evaluate a behavior-cloned policy checkpoint as NAME",
    )
    parser.add_argument(
        "--report-all-learned",
        action="store_true",
        help="write learned-policy rows even when they do not beat or complement deterministic baselines",
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        default=None,
        help="block sizes to run; defaults to each pair's manifest block_size_bytes",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for learned-policy inference: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="results/benchmark_suite")
    parser.add_argument(
        "--deployment-config",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help='repeatable deployment config, e.g. tight=\'{"ram_budget_bytes": 262144}\'',
    )
    parser.add_argument(
        "--deployment-config-file",
        action="append",
        default=[],
        help="JSON file containing a list of {name, config} objects or a name-to-config mapping",
    )
    parser.add_argument(
        "--interruption-setting",
        action="append",
        default=[],
        metavar="NAME=PCT[,PCT...]",
        help="repeatable interruption percentage set, e.g. default=0.25,0.5,0.75",
    )
    parser.add_argument(
        "--no-interruption-eval",
        action="store_true",
        help="run with one interruption setting named 'none' and no interruption scenarios",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_args(args):
    for policy_name in args.policies:
        if not is_supported_policy_spec(policy_name):
            raise ValueError(
                f"unknown policy '{policy_name}'. Available: {available_policy_specs()} "
                "or use --learned-policy NAME=PATH"
            )
    for value in args.learned_policy or []:
        parse_learned_policy(value)
    if args.block_sizes:
        for block_size in args.block_sizes:
            if block_size <= 0:
                raise ValueError("--block-sizes values must be positive")
    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    deployment_config_sets(args)
    interruption_sets(args)


def main(args):
    results, attempt_ledger = run_suite_with_attempt_ledger(args)
    results, skipped_learned = filter_reported_learned_results(
        results,
        report_all_learned=args.report_all_learned,
    )
    update_attempt_ledger_for_reported_results(attempt_ledger, results, skipped_learned)
    if skipped_learned:
        print(
            "Skipped "
            f"{len(skipped_learned)} learned-policy result(s) that did not beat or "
            "complement deterministic baselines. Use --report-all-learned to keep diagnostics."
        )
    paths = write_outputs(args.output_dir, results, attempt_ledger_rows=attempt_ledger)
    print(f"Wrote JSONL results: {paths['jsonl']}")
    print(f"Wrote CSV results: {paths['csv']}")
    print(f"Wrote Markdown summary: {paths['markdown']}")
    print(f"Wrote attempt ledger: {paths['attempt_ledger']}")
    return {
        "paths": paths,
        "results": results,
        "attempt_ledger": attempt_ledger,
    }


if __name__ == "__main__":
    main(parse_args())
