import hashlib
import math
import random
import time
from dataclasses import dataclass

import bsdiff4

from deployment.semantics import AB_FLAT_METRIC_KEYS, DeploymentSimulator, simulate_update
from encoding.operations import (
    DeltaCodec,
    make_backup_operation,
    make_copy_operation,
    make_delta_operation,
    make_keep_operation,
    make_raw_insert_operation,
    make_truncate_operation,
)
from encoding.replay import blocks_to_bytes, read_blocks, replay_operations, validate_update
from env.ota_env import OTAEnv
from evaluation.metrics import file_metadata


ACTION_M = 0
ACTION_MB = 1


def _first_remaining(env):
    return env._blocks_remaining[0]


def sequential_m(env, rng):
    return ACTION_M, _first_remaining(env)


def sequential_mb(env, rng):
    return ACTION_MB, _first_remaining(env)


def random_policy(env, rng):
    return rng.choice([ACTION_M, ACTION_MB]), rng.choice(env._blocks_remaining)


def copy_first(env, rng):
    old_md5s = set(env._old_block_md5s)
    for block_index in env._blocks_remaining:
        if env._new_block_md5s[block_index] in old_md5s:
            return ACTION_M, block_index
    return ACTION_M, _first_remaining(env)


def _target_needs_backup(env, block_index):
    snapshot_blocks = env._deployment.state.snapshot_blocks
    if block_index >= len(snapshot_blocks):
        return False
    return env._new_blocks[block_index] != snapshot_blocks[block_index]


def _action_for_target(env, block_index):
    if _target_needs_backup(env, block_index):
        return ACTION_MB
    return ACTION_M


def _candidate_source_blocks(env, block_index):
    seen = set()
    for candidate in env.get_similarity_candidates(block_index):
        source_index = candidate["source_index"]
        if source_index in seen:
            continue
        if 0 <= source_index < len(env._old_blocks):
            seen.add(source_index)
            yield env._old_blocks[source_index]
    if not seen and env._old_blocks:
        yield env._old_blocks[min(block_index, len(env._old_blocks) - 1)]


def backup_aware_copy_delta(env, rng):
    for block_index in env._blocks_remaining:
        if not _target_needs_backup(env, block_index):
            return ACTION_M, block_index

    old_md5s = set(env._old_block_md5s)
    for block_index in env._blocks_remaining:
        if env._new_block_md5s[block_index] in old_md5s:
            return _action_for_target(env, block_index), block_index

    best_block = _first_remaining(env)
    best_patch_size = float("inf")
    for block_index in env._blocks_remaining:
        new_block = env._new_blocks[block_index]
        for old_block in _candidate_source_blocks(env, block_index):
            patch_size = len(bsdiff4.diff(old_block, new_block))
            if patch_size < best_patch_size:
                best_patch_size = patch_size
                best_block = block_index

    return _action_for_target(env, best_block), best_block


def greedy_smallest_delta(env, rng):
    best_block = _first_remaining(env)
    best_patch_size = float("inf")

    for block_index in env._blocks_remaining:
        new_block = env._new_blocks[block_index]
        for old_block in _candidate_source_blocks(env, block_index):
            patch_size = len(bsdiff4.diff(old_block, new_block))
            if patch_size < best_patch_size:
                best_patch_size = patch_size
                best_block = block_index

    return ACTION_M, best_block


ACTION_POLICIES = {
    "sequential_m": sequential_m,
    "sequential_mb": sequential_mb,
    "random": random_policy,
    "copy_first": copy_first,
    "backup_aware_copy_delta": backup_aware_copy_delta,
    "greedy_smallest_delta": greedy_smallest_delta,
}

PUBLICATION_BASELINE_NAMES = [
    "full_replacement",
    "whole_file_bsdiff",
    "blockwise_bsdiff",
    "copy_only",
    "copy_delta",
    "backup_safe_copy_delta",
    "deployment_aware_greedy",
    "rsync_rolling_hash",
]

DEFAULT_BLOCK_SIZE_BYTES = 4096 * 16
DEFAULT_INTERRUPTION_PERCENTAGES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class SourceRef:
    block: bytes
    source: int
    source_area: str = "installed"
    source_backup_id: str | None = None

    def operation_kwargs(self):
        kwargs = {
            "source": self.source,
            "source_area": self.source_area,
        }
        if self.source_backup_id is not None:
            kwargs["source_backup_id"] = self.source_backup_id
        return kwargs


def _write_block(blocks, target, block):
    while len(blocks) < target:
        blocks.append(b"")
    if target == len(blocks):
        blocks.append(block)
    else:
        blocks[target] = block


def _truncate_blocks_if_needed(blocks, new_size_bytes, block_size, operations):
    if len(blocks_to_bytes(blocks)) <= new_size_bytes:
        return
    operations.append(make_truncate_operation(new_size_bytes=new_size_bytes))
    data = blocks_to_bytes(blocks)[:new_size_bytes]
    blocks[:] = [data[index:index + block_size] for index in range(0, len(data), block_size)]


def _backup_id(block_index):
    return f"old-block-{block_index}"


def _backup_block(blocks, backups, operations, block_index):
    if block_index < 0 or block_index >= len(blocks) or block_index in backups:
        return
    backup_id = _backup_id(block_index)
    operations.append(make_backup_operation(source=block_index, backup_id=backup_id))
    backups[block_index] = SourceRef(
        block=blocks[block_index],
        source=block_index,
        source_area="backup",
        source_backup_id=backup_id,
    )


def _installed_sources(blocks):
    return [
        SourceRef(block=block, source=index)
        for index, block in enumerate(blocks)
    ]


def _source_candidates(blocks, backups=None):
    candidates = _installed_sources(blocks)
    if backups:
        candidates.extend(backups[index] for index in sorted(backups))
    return candidates


def _bounded_delta_candidates(target_index, blocks, backups=None):
    candidates = []
    if target_index < len(blocks):
        candidates.append(SourceRef(block=blocks[target_index], source=target_index))
    if backups and target_index in backups:
        candidates.append(backups[target_index])
    if not candidates and blocks:
        fallback_index = min(target_index, len(blocks) - 1)
        candidates.append(SourceRef(block=blocks[fallback_index], source=fallback_index))
    return candidates


def _first_exact_source(target_block, candidates, target_index=None):
    if target_index is not None:
        for candidate in candidates:
            if (
                candidate.source_area == "installed"
                and candidate.source == target_index
                and candidate.block == target_block
            ):
                return candidate
    for candidate in candidates:
        if candidate.block == target_block:
            return candidate
    return None


def _best_delta_source(target_block, candidates):
    if not candidates:
        return None, b""

    best_source = None
    best_patch = None
    for candidate in candidates:
        patch = bsdiff4.diff(candidate.block, target_block)
        if (
            best_patch is None
            or len(patch) < len(best_patch)
            or (
                len(patch) == len(best_patch)
                and _source_sort_key(candidate) < _source_sort_key(best_source)
            )
        ):
            best_source = candidate
            best_patch = patch
    return best_source, best_patch


def _source_sort_key(candidate):
    area_rank = 0 if candidate.source_area == "installed" else 1
    return area_rank, candidate.source, candidate.source_backup_id or ""


def _raw_insert(target_index, target_block, blocks, operations):
    operations.append(make_raw_insert_operation(target=target_index, data=target_block))
    _write_block(blocks, target_index, target_block)


def _copy_or_keep(target_index, target_block, source, blocks, operations):
    if source.source_area == "installed" and source.source == target_index:
        operations.append(make_keep_operation(target=target_index, target_block=target_block))
    else:
        operations.append(
            make_copy_operation(
                target=target_index,
                target_block=target_block,
                **source.operation_kwargs(),
            )
        )
        _write_block(blocks, target_index, target_block)


def _delta_or_raw(target_index, target_block, candidates, blocks, operations):
    source, patch = _best_delta_source(target_block, candidates)
    if source is None:
        _raw_insert(target_index, target_block, blocks, operations)
        return
    raw_operation = make_raw_insert_operation(target=target_index, data=target_block)
    delta_operation = make_delta_operation(
        target=target_index,
        patch_bytes=patch,
        target_block=target_block,
        codec=DeltaCodec.BSDIFF4,
        **source.operation_kwargs(),
    )
    if (
        _operation_payload_preference(delta_operation)
        <= _operation_payload_preference(raw_operation)
    ):
        operations.append(delta_operation)
    else:
        operations.append(raw_operation)
    _write_block(blocks, target_index, target_block)


def _operation_payload_preference(operation):
    if operation["op"] == "delta":
        return len(operation["patch_b64"])
    if operation["op"] == "raw_insert":
        return len(operation["data_b64"])
    return 0


def _build_full_replacement(old_blocks, new_blocks, block_size):
    operations = []
    current_blocks = list(old_blocks)
    new_bytes = blocks_to_bytes(new_blocks)
    if new_bytes:
        _raw_insert(0, new_bytes, current_blocks, operations)
    _truncate_blocks_if_needed(current_blocks, len(new_bytes), block_size, operations)
    return operations


def _build_whole_file_bsdiff(old_blocks, new_blocks, block_size):
    operations = []
    current_blocks = list(old_blocks)
    old_bytes = blocks_to_bytes(old_blocks)
    new_bytes = blocks_to_bytes(new_blocks)
    patch = bsdiff4.diff(old_bytes, new_bytes)
    operations.append(
        make_delta_operation(
            source=0,
            source_area="image",
            target=0,
            patch_bytes=patch,
            target_block=new_bytes,
            codec=DeltaCodec.BSDIFF4,
        )
    )
    _write_block(current_blocks, 0, new_bytes)
    _truncate_blocks_if_needed(current_blocks, len(new_bytes), block_size, operations)
    return operations


def _build_blockwise_bsdiff(old_blocks, new_blocks, block_size):
    operations = []
    current_blocks = list(old_blocks)
    for target_index, target_block in enumerate(new_blocks):
        if target_index < len(current_blocks) and current_blocks[target_index] == target_block:
            operations.append(make_keep_operation(target=target_index, target_block=target_block))
            continue
        if target_index < len(current_blocks):
            patch = bsdiff4.diff(current_blocks[target_index], target_block)
            operations.append(
                make_delta_operation(
                    source=target_index,
                    target=target_index,
                    patch_bytes=patch,
                    target_block=target_block,
                    codec=DeltaCodec.BSDIFF4,
                )
            )
            _write_block(current_blocks, target_index, target_block)
        else:
            _raw_insert(target_index, target_block, current_blocks, operations)
    _truncate_blocks_if_needed(
        current_blocks,
        len(blocks_to_bytes(new_blocks)),
        block_size,
        operations,
    )
    return operations


def _build_copy_only(old_blocks, new_blocks, block_size):
    operations = []
    current_blocks = list(old_blocks)
    for target_index, target_block in enumerate(new_blocks):
        source = _first_exact_source(target_block, _installed_sources(current_blocks), target_index)
        if source is None:
            _raw_insert(target_index, target_block, current_blocks, operations)
        else:
            _copy_or_keep(target_index, target_block, source, current_blocks, operations)
    _truncate_blocks_if_needed(
        current_blocks,
        len(blocks_to_bytes(new_blocks)),
        block_size,
        operations,
    )
    return operations


def _build_copy_delta(old_blocks, new_blocks, block_size):
    operations = []
    current_blocks = list(old_blocks)
    for target_index, target_block in enumerate(new_blocks):
        candidates = _installed_sources(current_blocks)
        source = _first_exact_source(target_block, candidates, target_index)
        if source is not None:
            _copy_or_keep(target_index, target_block, source, current_blocks, operations)
        else:
            _delta_or_raw(
                target_index,
                target_block,
                _bounded_delta_candidates(target_index, current_blocks),
                current_blocks,
                operations,
            )
    _truncate_blocks_if_needed(
        current_blocks,
        len(blocks_to_bytes(new_blocks)),
        block_size,
        operations,
    )
    return operations


def _build_backup_safe_copy_delta(old_blocks, new_blocks, block_size):
    operations = []
    current_blocks = list(old_blocks)
    backups = {}
    old_size_bytes = len(blocks_to_bytes(old_blocks))
    new_size_bytes = len(blocks_to_bytes(new_blocks))
    old_block_count_after_truncate = (
        (new_size_bytes + block_size - 1) // block_size
        if new_size_bytes
        else 0
    )

    for block_index in range(old_block_count_after_truncate, len(old_blocks)):
        _backup_block(current_blocks, backups, operations, block_index)

    for target_index, target_block in enumerate(new_blocks):
        if (
            target_index < len(old_blocks)
            and target_index < len(current_blocks)
            and current_blocks[target_index] != target_block
        ):
            _backup_block(current_blocks, backups, operations, target_index)

        exact_candidates = _source_candidates(current_blocks, backups)
        source = _first_exact_source(target_block, exact_candidates, target_index)
        if source is not None:
            _copy_or_keep(target_index, target_block, source, current_blocks, operations)
        else:
            _delta_or_raw(
                target_index,
                target_block,
                _bounded_delta_candidates(target_index, current_blocks, backups),
                current_blocks,
                operations,
            )

    if new_size_bytes < old_size_bytes:
        _truncate_blocks_if_needed(current_blocks, new_size_bytes, block_size, operations)
    return operations


def _backup_refs_from_simulator(simulator):
    return {
        record.index: SourceRef(
            block=record.block,
            source=record.index,
            source_area="backup",
            source_backup_id=backup_id,
        )
        for backup_id, record in sorted(simulator.state.backup_area.items())
    }


def _has_backup_for_index(simulator, block_index):
    return any(record.index == block_index for record in simulator.state.backup_area.values())


def _backup_prefix_if_needed(simulator, target_index, target_block):
    if not simulator.config.require_backup_for_overwrite:
        return []
    if target_index >= len(simulator.state.snapshot_blocks):
        return []
    if target_block == simulator.state.snapshot_blocks[target_index]:
        return []
    if target_index >= len(simulator.state.installed_blocks):
        return []
    if _has_backup_for_index(simulator, target_index):
        return []
    return [make_backup_operation(source=target_index, backup_id=_backup_id(target_index))]


def _with_backup_variants(prefix, operation):
    if not prefix:
        return [[operation]]
    return [list(prefix) + [operation], [operation]]


def _deployment_target_candidates(simulator, target_index, target_block):
    current_blocks = list(simulator.state.installed_blocks)
    if target_index < len(current_blocks) and current_blocks[target_index] == target_block:
        return [[make_keep_operation(target=target_index, target_block=target_block)]]

    backups = _backup_refs_from_simulator(simulator)
    prefix = _backup_prefix_if_needed(simulator, target_index, target_block)
    candidates = []

    for source in _source_candidates(current_blocks, backups):
        if source.block != target_block:
            continue
        operation = make_copy_operation(
            target=target_index,
            target_block=target_block,
            **source.operation_kwargs(),
        )
        candidates.extend(_with_backup_variants(prefix, operation))

    for source in _bounded_delta_candidates(target_index, current_blocks, backups):
        patch = bsdiff4.diff(source.block, target_block)
        operation = make_delta_operation(
            target=target_index,
            patch_bytes=patch,
            target_block=target_block,
            codec=DeltaCodec.BSDIFF4,
            **source.operation_kwargs(),
        )
        candidates.extend(_with_backup_variants(prefix, operation))

    raw_operation = make_raw_insert_operation(target=target_index, data=target_block)
    candidates.extend(_with_backup_variants(prefix, raw_operation))
    return candidates


def _score_deployment_candidate(simulator, candidate):
    trial = simulator.clone()
    starting_violations = set(simulator.metrics()["budget_violations"])
    try:
        for operation in candidate:
            trial.apply_operation(operation)
    except Exception:
        return None, None

    metrics = trial.metrics()
    new_violations = set(metrics["budget_violations"]) - starting_violations
    rollback_penalty = 0 if metrics["rollback_ready"] else 1
    score = (
        len(new_violations),
        rollback_penalty,
        metrics["budget_violation_count"],
        metrics["package_size_bytes"],
        metrics["peak_persistent_storage_bytes"],
        metrics["peak_ram_bytes"],
        metrics["flash_write_bytes"],
        metrics["total_time_s"],
        len(candidate),
    )
    return score, trial


def _choose_deployment_candidate(simulator, candidates):
    best_candidate = None
    best_score = None
    for candidate in candidates:
        score, _ = _score_deployment_candidate(simulator, candidate)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_candidate = candidate
            best_score = score
    if best_candidate is None:
        raise RuntimeError("no valid deployment-aware baseline candidate was available")
    return best_candidate


def _apply_candidate(simulator, operations, candidate):
    for operation in candidate:
        simulator.apply_operation(operation)
        operations.append(operation)


def _truncate_backup_indices(simulator, new_size_bytes, block_size):
    current_blocks = list(simulator.state.installed_blocks)
    current_bytes = blocks_to_bytes(current_blocks)
    if len(current_bytes) <= new_size_bytes:
        return []

    truncated_data = current_bytes[:new_size_bytes]
    truncated_blocks = [
        truncated_data[index:index + block_size]
        for index in range(0, len(truncated_data), block_size)
    ]
    snapshot_blocks = simulator.state.snapshot_blocks
    affected = set()

    for index in range(len(truncated_blocks), len(current_blocks)):
        if index < len(snapshot_blocks):
            affected.add(index)

    if truncated_blocks:
        final_index = len(truncated_blocks) - 1
        if (
            final_index < len(snapshot_blocks)
            and truncated_blocks[final_index] != snapshot_blocks[final_index]
        ):
            affected.add(final_index)

    return [
        index
        for index in sorted(affected)
        if index < len(current_blocks) and not _has_backup_for_index(simulator, index)
    ]


def _deployment_truncate_candidates(simulator, new_size_bytes, block_size):
    backup_ops = [
        make_backup_operation(source=index, backup_id=_backup_id(index))
        for index in _truncate_backup_indices(simulator, new_size_bytes, block_size)
    ]
    truncate_op = make_truncate_operation(new_size_bytes=new_size_bytes)
    if backup_ops:
        return [backup_ops + [truncate_op], [truncate_op]]
    return [[truncate_op]]


def _build_deployment_aware_greedy(old_blocks, new_blocks, block_size, deployment_config=None):
    operations = []
    simulator = DeploymentSimulator(old_blocks, block_size, config=deployment_config)

    for target_index, target_block in enumerate(new_blocks):
        candidate = _choose_deployment_candidate(
            simulator,
            _deployment_target_candidates(simulator, target_index, target_block),
        )
        _apply_candidate(simulator, operations, candidate)

    new_size_bytes = len(blocks_to_bytes(new_blocks))
    if len(blocks_to_bytes(simulator.state.installed_blocks)) > new_size_bytes:
        candidate = _choose_deployment_candidate(
            simulator,
            _deployment_truncate_candidates(simulator, new_size_bytes, block_size),
        )
        _apply_candidate(simulator, operations, candidate)

    return operations


def _rolling_weak_checksum(data):
    modulus = 65521
    a = 0
    b = 0
    length = len(data)
    for index, value in enumerate(data):
        a = (a + value) % modulus
        b = (b + (length - index) * value) % modulus
    return (b << 16) | a


def _strong_checksum(data):
    return hashlib.sha256(data).digest()


def _build_rsync_rolling_hash(old_blocks, new_blocks, block_size):
    operations = []
    current_blocks = list(old_blocks)
    signatures = {}
    for source_index, source_block in enumerate(old_blocks):
        key = (_rolling_weak_checksum(source_block), _strong_checksum(source_block))
        signatures.setdefault(key, []).append(source_index)

    for target_index, target_block in enumerate(new_blocks):
        if target_index < len(current_blocks) and current_blocks[target_index] == target_block:
            operations.append(make_keep_operation(target=target_index, target_block=target_block))
            continue

        key = (_rolling_weak_checksum(target_block), _strong_checksum(target_block))
        source = None
        for source_index in signatures.get(key, []):
            if source_index < len(current_blocks) and current_blocks[source_index] == target_block:
                source = SourceRef(block=target_block, source=source_index)
                break
        if source is None:
            _raw_insert(target_index, target_block, current_blocks, operations)
        else:
            _copy_or_keep(target_index, target_block, source, current_blocks, operations)
    _truncate_blocks_if_needed(
        current_blocks,
        len(blocks_to_bytes(new_blocks)),
        block_size,
        operations,
    )
    return operations


OPERATION_BASELINES = {
    "full_replacement": _build_full_replacement,
    "whole_file_bsdiff": _build_whole_file_bsdiff,
    "blockwise_bsdiff": _build_blockwise_bsdiff,
    "copy_only": _build_copy_only,
    "copy_delta": _build_copy_delta,
    "backup_safe_copy_delta": _build_backup_safe_copy_delta,
    "deployment_aware_greedy": _build_deployment_aware_greedy,
    "rsync_rolling_hash": _build_rsync_rolling_hash,
}

POLICIES = {**ACTION_POLICIES, **OPERATION_BASELINES}
LEARNED_POLICY_PREFIXES = ("imitation:", "bc:")


def is_learned_policy_spec(policy_name):
    return any(str(policy_name).startswith(prefix) for prefix in LEARNED_POLICY_PREFIXES)


def is_supported_policy_spec(policy_name):
    return policy_name in POLICIES or is_learned_policy_spec(policy_name)


def available_policy_specs():
    return sorted(POLICIES)


def operation_counts(operations):
    counts = {}
    for operation in operations:
        op_name = operation.get("op", "unknown")
        counts[op_name] = counts.get(op_name, 0) + 1
    return counts


def flat_ab_result_metrics(deployment_metrics):
    if not deployment_metrics.get("ab_enabled"):
        return {}
    return {
        key: deployment_metrics.get(key)
        for key in AB_FLAT_METRIC_KEYS
    }


def deployment_update_completed(deployment_metrics):
    if deployment_metrics.get("ab_enabled"):
        return (
            deployment_metrics.get("install_state") == "complete"
            and bool(deployment_metrics.get("ab_update_valid"))
        )
    return deployment_metrics.get("install_state") == "complete"


def _normalise_interruption_percentages(percentages):
    if percentages is None:
        return tuple(DEFAULT_INTERRUPTION_PERCENTAGES)
    return tuple(float(percentage) for percentage in percentages)


def _interruption_operation_index(operation_count, percentage):
    if operation_count <= 0:
        return 0
    clamped = min(1.0, max(0.0, float(percentage)))
    if operation_count == 1:
        return 0
    return min(operation_count - 1, max(0, int(math.ceil(operation_count * clamped))))


def _apply_operations(simulator, operations):
    errors = []
    for operation in operations:
        try:
            simulator.apply_operation(operation)
        except Exception as exc:
            errors.append(str(exc))
            break
    return errors


def _final_bytes_match(simulator, expected_blocks):
    return blocks_to_bytes(simulator.state.installed_blocks) == blocks_to_bytes(expected_blocks)


def _checkpoint_replay_valid(old_file, new_file, operations, block_size, checkpoint_index):
    checkpoint_replay = replay_operations(
        read_blocks(old_file, block_size),
        operations[:checkpoint_index],
        block_size=block_size,
    )
    if checkpoint_replay.errors:
        return False, "; ".join(checkpoint_replay.errors)
    result = validate_update(
        old_file,
        new_file,
        operations[checkpoint_index:],
        block_size,
        state=checkpoint_replay.state,
    )
    return result.valid, "; ".join(result.errors)


def _rollback_replay_valid(old_file, new_file, operations, block_size):
    result = validate_update(old_file, new_file, operations, block_size)
    return result.valid, "; ".join(result.errors)


def _empty_interruption_summary():
    return {
        "scenario_count": 0,
        "checkpoint_resume_count": 0,
        "rollback_success_count": 0,
        "failed_recovery_count": 0,
        "final_replay_validity_all": True,
        "max_recovery_cost_operations": 0,
        "max_extra_network_bytes": 0,
        "max_extra_flash_writes": 0,
    }


def summarize_interruption_results(results):
    if not results:
        return _empty_interruption_summary()
    return {
        "scenario_count": len(results),
        "checkpoint_resume_count": sum(1 for result in results if result["resumed_from_checkpoint"]),
        "rollback_success_count": sum(1 for result in results if result["rollback_success"]),
        "failed_recovery_count": sum(1 for result in results if not result["recovery_success"]),
        "final_replay_validity_all": all(result["final_replay_validity"] for result in results),
        "max_recovery_cost_operations": max(
            result["recovery_cost"]["replayed_operation_count"]
            for result in results
        ),
        "max_extra_network_bytes": max(result["extra_network_bytes"] for result in results),
        "max_extra_flash_writes": max(result["extra_flash_writes"] for result in results),
    }


def simulate_interrupted_update(
    old_file,
    new_file,
    operations,
    block_size=DEFAULT_BLOCK_SIZE_BYTES,
    config=None,
    interruption_percentages=DEFAULT_INTERRUPTION_PERCENTAGES,
    baseline_deployment_metrics=None,
):
    percentages = _normalise_interruption_percentages(interruption_percentages)
    if not percentages:
        return []
    operation_count = len(operations)
    if operation_count == 0:
        return []

    initial_blocks = read_blocks(old_file, block_size)
    expected_blocks = read_blocks(new_file, block_size)
    if baseline_deployment_metrics is None:
        baseline_deployment_metrics = simulate_update(
            old_file,
            new_file,
            operations,
            block_size,
            config=config,
        ).metrics

    results = []
    for percentage in percentages:
        interrupt_after = _interruption_operation_index(operation_count, percentage)
        interrupted = DeploymentSimulator(
            initial_blocks,
            block_size,
            config=config,
            expected_blocks=expected_blocks,
        )
        preinterrupt_errors = _apply_operations(interrupted, operations[:interrupt_after])
        if not preinterrupt_errors:
            interrupted.interrupt()

        interrupted_metrics = interrupted.metrics()
        checkpoint = interrupted.latest_checkpoint()
        checkpoint_index = checkpoint.operation_index if checkpoint is not None else 0
        checkpoint_network_bytes = checkpoint.network_bytes if checkpoint is not None else 0
        checkpoint_flash_write_bytes = checkpoint.flash_write_bytes if checkpoint is not None else 0
        rollback_success = False
        recovery_success = False
        final_replay_validity = False
        replay_errors = ""
        recovery_errors = list(preinterrupt_errors)
        recovery_strategy = "failed"
        recovered_metrics = {}
        recovery_network_bytes = 0
        recovery_flash_write_bytes = 0
        recovery_operation_count = 0
        total_network_bytes = interrupted_metrics["network_bytes"]
        total_flash_write_bytes = interrupted_metrics["flash_write_bytes"]

        if not preinterrupt_errors and interrupted.state.rollback_ready:
            rollback_trial = DeploymentSimulator(
                [],
                block_size,
                config=config,
                state=interrupted.state,
                expected_blocks=expected_blocks,
            )
            try:
                rollback_trial.rollback()
                rollback_success = blocks_to_bytes(rollback_trial.state.installed_blocks) == blocks_to_bytes(initial_blocks)
            except Exception:
                rollback_success = False

        if checkpoint is not None and not preinterrupt_errors:
            recovery_strategy = "checkpoint_resume"
            resumed = DeploymentSimulator(
                [],
                block_size,
                config=config,
                state=interrupted.state,
                expected_blocks=expected_blocks,
            )
            try:
                resumed.restore_checkpoint(checkpoint)
                recovery_errors.extend(_apply_operations(resumed, operations[checkpoint_index:]))
                if not recovery_errors:
                    resumed.complete(expected_blocks=expected_blocks)
                recovered_metrics = resumed.metrics()
                recovery_network_bytes = max(0, recovered_metrics["network_bytes"] - checkpoint_network_bytes)
                recovery_flash_write_bytes = max(0, recovered_metrics["flash_write_bytes"] - checkpoint_flash_write_bytes)
                recovery_operation_count = max(0, operation_count - checkpoint_index)
                total_network_bytes = interrupted_metrics["network_bytes"] + recovery_network_bytes
                total_flash_write_bytes = interrupted_metrics["flash_write_bytes"] + recovery_flash_write_bytes
                recovery_success = (
                    not recovery_errors
                    and recovered_metrics["install_state"] == "complete"
                    and _final_bytes_match(resumed, expected_blocks)
                )
                final_replay_validity, replay_errors = _checkpoint_replay_valid(
                    old_file,
                    new_file,
                    operations,
                    block_size,
                    checkpoint_index,
                )
            except Exception as exc:
                recovery_errors.append(str(exc))
        elif rollback_success and not preinterrupt_errors:
            recovery_strategy = "rollback_reinstall"
            recovered = DeploymentSimulator(
                [],
                block_size,
                config=config,
                state=interrupted.state,
                expected_blocks=expected_blocks,
            )
            try:
                recovered.rollback()
                recovery_start_metrics = recovered.metrics()
                recovery_errors.extend(_apply_operations(recovered, operations))
                if not recovery_errors:
                    recovered.complete(expected_blocks=expected_blocks)
                recovered_metrics = recovered.metrics()
                recovery_network_bytes = max(0, recovered_metrics["network_bytes"] - recovery_start_metrics["network_bytes"])
                recovery_flash_write_bytes = max(
                    0,
                    recovered_metrics["flash_write_bytes"] - recovery_start_metrics["flash_write_bytes"],
                )
                recovery_operation_count = operation_count
                total_network_bytes = recovered_metrics["network_bytes"]
                total_flash_write_bytes = recovered_metrics["flash_write_bytes"]
                recovery_success = (
                    not recovery_errors
                    and recovered_metrics["install_state"] == "complete"
                    and _final_bytes_match(recovered, expected_blocks)
                )
                final_replay_validity, replay_errors = _rollback_replay_valid(
                    old_file,
                    new_file,
                    operations,
                    block_size,
                )
            except Exception as exc:
                recovery_errors.append(str(exc))
        elif not preinterrupt_errors:
            recovery_errors.append("no checkpoint and rollback is not ready")

        extra_network_bytes = max(
            0,
            total_network_bytes - baseline_deployment_metrics["network_bytes"],
        )
        extra_flash_writes = max(
            0,
            total_flash_write_bytes - baseline_deployment_metrics["flash_write_bytes"],
        )
        discarded_operation_count = max(0, interrupt_after - checkpoint_index)

        results.append({
            "interruption_percentage": percentage,
            "interrupted_after_operations": interrupt_after,
            "operation_count": operation_count,
            "checkpoint_available": checkpoint is not None,
            "checkpoint_operation_index": checkpoint_index,
            "resumed_from_checkpoint": recovery_strategy == "checkpoint_resume",
            "recovery_strategy": recovery_strategy,
            "recovery_success": recovery_success,
            "rollback_success": rollback_success,
            "rollback_ready_at_interruption": interrupted_metrics["rollback_ready"],
            "final_replay_validity": final_replay_validity and recovery_success,
            "replay_errors": replay_errors,
            "recovery_errors": "; ".join(recovery_errors),
            "extra_network_bytes": extra_network_bytes,
            "extra_flash_writes": extra_flash_writes,
            "total_network_bytes": total_network_bytes,
            "total_flash_write_bytes": total_flash_write_bytes,
            "interrupted_metrics": interrupted_metrics,
            "recovered_metrics": recovered_metrics,
            "recovery_cost": {
                "discarded_operation_count": discarded_operation_count,
                "replayed_operation_count": recovery_operation_count,
                "network_bytes": recovery_network_bytes,
                "flash_write_bytes": recovery_flash_write_bytes,
            },
        })

    return results


def run_policy(
    policy_name,
    old_file,
    new_file,
    seed=1,
    max_steps=None,
    deployment_config=None,
    block_size=DEFAULT_BLOCK_SIZE_BYTES,
    interruption_percentages=DEFAULT_INTERRUPTION_PERCENTAGES,
    device="auto",
):
    if is_learned_policy_spec(policy_name):
        model_path = str(policy_name).split(":", 1)[1]
        from learning.imitation import run_imitation_policy

        return run_imitation_policy(
            model_path=model_path,
            old_file=old_file,
            new_file=new_file,
            seed=seed,
            max_steps=max_steps,
            deployment_config=deployment_config,
            block_size=block_size,
            interruption_percentages=interruption_percentages,
            device=device,
        )

    if policy_name not in POLICIES:
        raise ValueError(
            f"Unknown policy '{policy_name}'. Available: {available_policy_specs()} "
            "or imitation:/path/to/model.pt"
        )
    if policy_name in OPERATION_BASELINES:
        return _run_operation_baseline(
            policy_name=policy_name,
            old_file=old_file,
            new_file=new_file,
            seed=seed,
            max_steps=max_steps,
            deployment_config=deployment_config,
            block_size=block_size,
            interruption_percentages=interruption_percentages,
        )

    rng = random.Random(seed)
    env = OTAEnv(old_file, new_file, block_size=block_size, deployment_config=deployment_config)
    _, _ = env.reset()
    start_time = time.perf_counter()

    done = env.get_block_num() == 0
    step_limit = max_steps if max_steps is not None else env.get_block_num()
    for _ in range(step_limit):
        action, next_block = ACTION_POLICIES[policy_name](env, rng)
        _, _, done, _ = env.step(action, next_block)
        if done:
            break

    duration_s = time.perf_counter() - start_time
    metrics = env.get_metrics()
    replay_result = env.validate_encoding()
    operations = env.get_encoding_ops()
    encoding_op_count = len(operations)
    op_counts = operation_counts(operations)
    env.close()

    old_meta = file_metadata(old_file)
    new_meta = file_metadata(new_file)

    runtime_s = round(duration_s, 6)
    deployment_metrics = metrics["deployment"]
    completed = done and deployment_update_completed(deployment_metrics)
    interruption_results = simulate_interrupted_update(
        old_file,
        new_file,
        operations,
        block_size=block_size,
        config=deployment_config,
        interruption_percentages=interruption_percentages,
        baseline_deployment_metrics=deployment_metrics,
    )

    return {
        "policy": policy_name,
        "baseline_name": policy_name,
        "seed": seed,
        "old_file": old_meta["path"],
        "old_path": old_meta["path"],
        "old_size_bytes": old_meta["size_bytes"],
        "old_sha256": old_meta["sha256"],
        "new_file": new_meta["path"],
        "new_path": new_meta["path"],
        "new_size_bytes": new_meta["size_bytes"],
        "new_sha256": new_meta["sha256"],
        "block_size": block_size,
        "block_size_bytes": block_size,
        "duration_s": runtime_s,
        "runtime": runtime_s,
        "runtime_s": runtime_s,
        "completed": completed,
        "encoding_op_count": encoding_op_count,
        "operation_counts": op_counts,
        "replay_valid": replay_result.valid,
        "replay_validity": replay_result.valid,
        "replay_errors": "; ".join(replay_result.errors),
        "interruption_results": interruption_results,
        "interruption_summary": summarize_interruption_results(interruption_results),
        **metrics,
        **flat_ab_result_metrics(deployment_metrics),
    }


def _run_operation_baseline(
    policy_name,
    old_file,
    new_file,
    seed=1,
    max_steps=None,
    deployment_config=None,
    block_size=DEFAULT_BLOCK_SIZE_BYTES,
    interruption_percentages=DEFAULT_INTERRUPTION_PERCENTAGES,
):
    old_blocks = read_blocks(old_file, block_size)
    new_blocks = read_blocks(new_file, block_size)

    start_time = time.perf_counter()
    if policy_name == "deployment_aware_greedy":
        operations = OPERATION_BASELINES[policy_name](
            old_blocks,
            new_blocks,
            block_size,
            deployment_config=deployment_config,
        )
    else:
        operations = OPERATION_BASELINES[policy_name](old_blocks, new_blocks, block_size)
    if max_steps is not None:
        operations = operations[:max_steps]
    deployment_result = simulate_update(
        old_file,
        new_file,
        operations,
        block_size,
        config=deployment_config,
    )
    duration_s = time.perf_counter() - start_time

    replay_result = validate_update(old_file, new_file, operations, block_size)
    old_meta = file_metadata(old_file)
    new_meta = file_metadata(new_file)
    deployment_metrics = deployment_result.metrics
    op_counts = operation_counts(operations)
    interruption_results = simulate_interrupted_update(
        old_file,
        new_file,
        operations,
        block_size=block_size,
        config=deployment_config,
        interruption_percentages=interruption_percentages,
        baseline_deployment_metrics=deployment_metrics,
    )
    runtime_s = round(duration_s, 6)
    completed = replay_result.valid and deployment_result.valid

    return {
        "policy": policy_name,
        "baseline_name": policy_name,
        "seed": seed,
        "old_file": old_meta["path"],
        "old_path": old_meta["path"],
        "old_size_bytes": old_meta["size_bytes"],
        "old_sha256": old_meta["sha256"],
        "new_file": new_meta["path"],
        "new_path": new_meta["path"],
        "new_size_bytes": new_meta["size_bytes"],
        "new_sha256": new_meta["sha256"],
        "block_size": block_size,
        "block_size_bytes": block_size,
        "duration_s": runtime_s,
        "runtime": runtime_s,
        "runtime_s": runtime_s,
        "completed": completed,
        "encoding_op_count": len(operations),
        "operation_counts": op_counts,
        "replay_valid": replay_result.valid,
        "replay_validity": replay_result.valid,
        "replay_errors": "; ".join(replay_result.errors),
        "interruption_results": interruption_results,
        "interruption_summary": summarize_interruption_results(interruption_results),
        "steps": len(operations),
        "blocks_total": len(new_blocks),
        "blocks_remaining": 0 if replay_result.valid else len(new_blocks),
        "memory_cost": deployment_metrics["peak_ram_bytes"],
        "encoding_cost": deployment_metrics["package_size_bytes"],
        "reward": 0.0,
        "action_counts": {
            "M": sum(1 for operation in operations if operation.get("op") != "backup"),
            "MB": sum(1 for operation in operations if operation.get("op") == "backup"),
        },
        "transformation_valid": replay_result.valid,
        "deployment": deployment_metrics,
        **flat_ab_result_metrics(deployment_metrics),
    }
