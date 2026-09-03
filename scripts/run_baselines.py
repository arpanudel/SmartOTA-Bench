import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_manifest
from deployment.semantics import DeploymentConfig
from evaluation.baselines import (
    DEFAULT_BLOCK_SIZE_BYTES,
    DEFAULT_INTERRUPTION_PERCENTAGES,
    PUBLICATION_BASELINE_NAMES,
    available_policy_specs,
    is_supported_policy_spec,
    run_policy,
)
from evaluation.metrics import write_results


MANIFEST_CSV_FIELDNAMES = [
    "dataset_id",
    "pair_id",
    "domain",
    "scenario",
    "artifact_type",
    "old_path",
    "new_path",
    "old_size_bytes",
    "new_size_bytes",
    "old_sha256",
    "new_sha256",
    "block_size",
    "block_size_bytes",
    "baseline_name",
    "policy",
    "encoding_cost",
    "memory_cost",
    "replay_validity",
    "replay_valid",
    "duration_s",
    "runtime",
    "runtime_s",
    "seed",
    "deployment_install_state",
    "deployment_package_size_bytes",
    "deployment_network_bytes",
    "deployment_peak_ram_bytes",
    "deployment_peak_persistent_storage_bytes",
    "deployment_flash_write_bytes",
    "deployment_total_time_s",
    "deployment_rollback_ready",
    "deployment_budget_violation_count",
    "interruption_summary_scenario_count",
    "interruption_summary_checkpoint_resume_count",
    "interruption_summary_rollback_success_count",
    "interruption_summary_failed_recovery_count",
    "interruption_summary_final_replay_validity_all",
    "interruption_summary_max_recovery_cost_operations",
    "interruption_summary_max_extra_network_bytes",
    "interruption_summary_max_extra_flash_writes",
]


def parse_learned_policy(value):
    if "=" not in value:
        raise ValueError("--learned-policy must use NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError("--learned-policy name cannot be empty")
    if not path:
        raise ValueError("--learned-policy path cannot be empty")
    return name, f"imitation:{path}"


def selected_policy_specs(args):
    specs = [(policy_name, policy_name) for policy_name in args.policies]
    specs.extend(parse_learned_policy(value) for value in args.learned_policy or [])
    return specs


def build_deployment_config(args):
    return DeploymentConfig(
        bandwidth_bytes_per_s=args.bandwidth_bytes_per_s,
        flash_write_bytes_per_s=args.flash_write_bytes_per_s,
        patch_apply_bytes_per_s=args.patch_apply_bytes_per_s,
        storage_budget_bytes=args.storage_budget_bytes,
        ram_budget_bytes=args.ram_budget_bytes,
        require_backup_for_overwrite=not args.allow_unsafe_overwrite,
        checkpoint_interval_ops=args.checkpoint_interval_ops,
        staging_strategy=args.staging_strategy,
        enable_ab_slots=args.enable_ab_slots,
        slot_capacity_bytes=args.slot_capacity_bytes,
        max_boot_attempts=args.max_boot_attempts,
        reboot_downtime_seconds=args.reboot_downtime_seconds,
        health_check_mode=args.health_check_mode,
        require_inactive_slot_install=not args.allow_active_slot_install,
    )


def infer_artifact_type(pair):
    explicit = pair.extra.get("artifact_type")
    if explicit:
        return explicit

    suffixes = "".join(pair.old_path.suffixes).lower()
    if suffixes.endswith((".tar.gz", ".tgz", ".zip", ".tar")):
        return "archive"
    if suffixes.endswith((".qcow2", ".img", ".iso")):
        return "image"
    if suffixes.endswith(".bin"):
        return "binary"
    return ""


def manifest_pair_metadata(manifest, pair):
    metadata = {
        "dataset_id": manifest.name,
        "pair_id": pair.id,
        "domain": pair.domain,
        "scenario": pair.scenario,
        "artifact_type": infer_artifact_type(pair),
        "manifest_path": str(manifest.path),
        "manifest_version": manifest.version,
        "old_path": str(pair.old_path),
        "new_path": str(pair.new_path),
        "old_size_bytes": pair.old_size_bytes,
        "new_size_bytes": pair.new_size_bytes,
        "old_sha256": pair.old_sha256,
        "new_sha256": pair.new_sha256,
        "block_size": pair.block_size_bytes,
        "block_size_bytes": pair.block_size_bytes,
        "source": pair.source,
        "license_notes": pair.license_notes,
    }
    for key in ("tier", "compression_status", "compression_status_source"):
        if key in pair.extra:
            metadata[key] = pair.extra[key]
    return metadata


def run_pair_baselines(args, old_file, new_file, block_size, deployment_config, metadata=None):
    results = []
    pair_label = f" on pair: {metadata['pair_id']}" if metadata else ""
    interruption_percentages = [] if args.no_interruption_eval else args.interruption_percentages
    for display_name, runner_policy in selected_policy_specs(args):
        print(f"Running baseline: {display_name}{pair_label}")
        result = run_policy(
            policy_name=runner_policy,
            old_file=old_file,
            new_file=new_file,
            seed=args.seed,
            max_steps=args.max_steps,
            deployment_config=deployment_config,
            block_size=block_size,
            interruption_percentages=interruption_percentages,
            device=args.device,
        )
        result["policy"] = display_name
        result["baseline_name"] = display_name
        if runner_policy != display_name:
            result["policy_runner"] = runner_policy
        if metadata:
            result.update(metadata)
        results.append(result)
    return results


def run_manifest_baselines(args, deployment_config):
    manifest = load_manifest(args.manifest, base_dir=getattr(args, "manifest_base_dir", None))
    if not manifest.pairs:
        print(f"No enabled dataset pairs found in manifest: {manifest.path}")
        return []

    results = []
    print(f"Loaded {len(manifest.pairs)} enabled dataset pair(s) from: {manifest.path}")
    for pair in manifest.pairs:
        results.extend(
            run_pair_baselines(
                args=args,
                old_file=pair.old_path,
                new_file=pair.new_path,
                block_size=pair.block_size_bytes,
                deployment_config=deployment_config,
                metadata=manifest_pair_metadata(manifest, pair),
            )
        )
    return results


def validate_args(args):
    for percentage in getattr(args, "interruption_percentages", []) or []:
        if percentage < 0.0 or percentage > 1.0:
            raise ValueError("--interruption-percentages values must be between 0.0 and 1.0")
    for policy_name in getattr(args, "policies", []) or []:
        if not is_supported_policy_spec(policy_name):
            raise ValueError(
                f"unknown policy '{policy_name}'. Available: {available_policy_specs()} "
                "or use --learned-policy NAME=PATH"
            )
    for value in getattr(args, "learned_policy", []) or []:
        parse_learned_policy(value)
    manifest = getattr(args, "manifest", None)
    old_file = getattr(args, "old_file", None)
    new_file = getattr(args, "new_file", None)
    if manifest:
        if old_file or new_file:
            raise ValueError("--manifest cannot be combined with --old-file or --new-file")
        return
    if not old_file or not new_file:
        raise ValueError("single-pair mode requires --old-file and --new-file, or use --manifest")


def main(args):
    validate_args(args)
    deployment_config = build_deployment_config(args)

    if args.manifest:
        results = run_manifest_baselines(args, deployment_config)
        fieldnames = MANIFEST_CSV_FIELDNAMES
    else:
        results = run_pair_baselines(
            args=args,
            old_file=args.old_file,
            new_file=args.new_file,
            block_size=DEFAULT_BLOCK_SIZE_BYTES,
            deployment_config=deployment_config,
        )
        fieldnames = None

    paths = write_results(args.output_dir, results, fieldnames=fieldnames)
    print(f"Wrote JSONL results: {paths['jsonl']}")
    print(f"Wrote CSV summary: {paths['csv']}")
    return {
        "paths": paths,
        "results": results,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--old-file")
    parser.add_argument("-n", "--new-file")
    parser.add_argument("--manifest", help="run baselines for every enabled pair in a dataset manifest")
    parser.add_argument(
        "--manifest-base-dir",
        default=None,
        help="override the manifest base_dir used to resolve relative artifact paths",
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for learned-policy inference: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument(
        "--interruption-percentages",
        nargs="*",
        type=float,
        default=list(DEFAULT_INTERRUPTION_PERCENTAGES),
        help="operation percentages at which to simulate interrupted updates",
    )
    parser.add_argument(
        "--no-interruption-eval",
        action="store_true",
        help="skip interrupted-update recovery evaluation",
    )
    parser.add_argument("--bandwidth-bytes-per-s", type=float, default=1_000_000.0)
    parser.add_argument("--flash-write-bytes-per-s", type=float, default=20_000_000.0)
    parser.add_argument("--patch-apply-bytes-per-s", type=float, default=10_000_000.0)
    parser.add_argument("--storage-budget-bytes", type=int, default=None)
    parser.add_argument("--ram-budget-bytes", type=int, default=None)
    parser.add_argument("--checkpoint-interval-ops", type=int, default=None)
    parser.add_argument("--staging-strategy", choices=["streaming", "full_package"], default="streaming")
    parser.add_argument("--allow-unsafe-overwrite", action="store_true")
    parser.add_argument("--enable-ab-slots", action="store_true")
    parser.add_argument("--slot-capacity-bytes", type=int, default=None)
    parser.add_argument("--max-boot-attempts", type=int, default=1)
    parser.add_argument("--reboot-downtime-seconds", type=float, default=0.0)
    parser.add_argument("--health-check-mode", choices=["always_pass", "forced_fail"], default="always_pass")
    parser.add_argument("--allow-active-slot-install", action="store_true")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    main(parse_args())
