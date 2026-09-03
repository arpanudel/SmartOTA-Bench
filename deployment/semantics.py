import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import bsdiff4

from encoding.operations import DeltaCodec, OperationType
from encoding.replay import block_sha256, blocks_to_bytes, bytes_sha256, bytes_to_blocks, decode_bytes, read_blocks


class DeploymentError(Exception):
    """Raised when a deployment operation violates the install model."""


@dataclass
class DeploymentConfig:
    bandwidth_bytes_per_s: float = 1_000_000.0
    flash_write_bytes_per_s: float = 20_000_000.0
    patch_apply_bytes_per_s: float = 10_000_000.0
    storage_budget_bytes: int | None = None
    ram_budget_bytes: int | None = None
    require_backup_for_overwrite: bool = True
    checkpoint_interval_ops: int | None = None
    staging_strategy: str = "streaming"
    metadata_overhead_bytes: int = 64
    enable_ab_slots: bool = False
    slot_capacity_bytes: int | None = None
    max_boot_attempts: int = 1
    reboot_downtime_seconds: float = 0.0
    health_check_mode: str = "always_pass"
    require_inactive_slot_install: bool = True

    def __post_init__(self):
        for field_name in (
            "bandwidth_bytes_per_s",
            "flash_write_bytes_per_s",
            "patch_apply_bytes_per_s",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.slot_capacity_bytes is not None and self.slot_capacity_bytes < 0:
            raise ValueError("slot_capacity_bytes cannot be negative")
        if self.max_boot_attempts < 1:
            raise ValueError("max_boot_attempts must be at least 1")
        if self.reboot_downtime_seconds < 0:
            raise ValueError("reboot_downtime_seconds cannot be negative")
        if self.health_check_mode not in {"always_pass", "forced_fail"}:
            raise ValueError(
                "health_check_mode must be one of: always_pass, forced_fail"
            )


AB_FLAT_METRIC_KEYS = (
    "ab_enabled",
    "ab_update_valid",
    "ab_rollback_ready",
    "slot_storage_bytes",
    "slot_storage_violation",
    "activation_success",
    "boot_health_success",
    "rollback_after_failed_boot",
    "rollback_success",
    "reboot_count",
    "downtime_seconds",
    "slot_switch_count",
)


@dataclass
class BackupRecord:
    index: int
    block: bytes


@dataclass
class DeploymentCheckpoint:
    operation_index: int
    installed_blocks: list
    backup_area: dict
    package_size_bytes: int
    network_bytes: int
    flash_write_bytes: int
    install_time_s: float
    download_time_s: float
    peak_ram_bytes: int = 0
    peak_persistent_storage_bytes: int = 0
    rollback_ready: bool = True
    unsafe_overwrites: list = field(default_factory=list)
    budget_violations: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def clone(self):
        return DeploymentCheckpoint(
            operation_index=self.operation_index,
            installed_blocks=list(self.installed_blocks),
            backup_area=copy.deepcopy(self.backup_area),
            package_size_bytes=self.package_size_bytes,
            network_bytes=self.network_bytes,
            flash_write_bytes=self.flash_write_bytes,
            install_time_s=self.install_time_s,
            download_time_s=self.download_time_s,
            peak_ram_bytes=self.peak_ram_bytes,
            peak_persistent_storage_bytes=self.peak_persistent_storage_bytes,
            rollback_ready=self.rollback_ready,
            unsafe_overwrites=list(self.unsafe_overwrites),
            budget_violations=list(self.budget_violations),
            errors=list(self.errors),
        )


@dataclass
class DeploymentState:
    installed_blocks: list
    snapshot_blocks: list
    backup_area: dict = field(default_factory=dict)
    staging_area_bytes: int = 0
    package_size_bytes: int = 0
    network_bytes: int = 0
    flash_write_bytes: int = 0
    applied_operations: int = 0
    install_state: str = "ready"
    peak_ram_bytes: int = 0
    current_ram_bytes: int = 0
    peak_persistent_storage_bytes: int = 0
    download_time_s: float = 0.0
    install_time_s: float = 0.0
    checkpoints: list = field(default_factory=list)
    rollback_ready: bool = True
    unsafe_overwrites: list = field(default_factory=list)
    budget_violations: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    active_slot: str = "A"
    inactive_slot: str = "B"
    active_version: str = ""
    inactive_version: str = ""
    slot_capacity_bytes: int | None = None
    slot_storage_bytes: int = 0
    slot_storage_violation: bool = False
    boot_attempt_counter: int = 0
    max_boot_attempts: int = 1
    pending_activation: bool = False
    health_verdict: str = "not_run"
    rollback_performed: bool = False
    ab_rollback_ready: bool = False
    ab_update_valid: bool = False
    ab_finalized: bool = False
    activation_success: bool = False
    boot_health_success: bool = False
    rollback_after_failed_boot: bool = False
    rollback_success: bool = False
    reboot_count: int = 0
    downtime_seconds: float = 0.0
    slot_switch_count: int = 0

    def clone(self):
        return DeploymentState(
            installed_blocks=list(self.installed_blocks),
            snapshot_blocks=list(self.snapshot_blocks),
            backup_area=copy.deepcopy(self.backup_area),
            staging_area_bytes=self.staging_area_bytes,
            package_size_bytes=self.package_size_bytes,
            network_bytes=self.network_bytes,
            flash_write_bytes=self.flash_write_bytes,
            applied_operations=self.applied_operations,
            install_state=self.install_state,
            peak_ram_bytes=self.peak_ram_bytes,
            current_ram_bytes=self.current_ram_bytes,
            peak_persistent_storage_bytes=self.peak_persistent_storage_bytes,
            download_time_s=self.download_time_s,
            install_time_s=self.install_time_s,
            checkpoints=[checkpoint.clone() for checkpoint in self.checkpoints],
            rollback_ready=self.rollback_ready,
            unsafe_overwrites=list(self.unsafe_overwrites),
            budget_violations=list(self.budget_violations),
            errors=list(self.errors),
            active_slot=self.active_slot,
            inactive_slot=self.inactive_slot,
            active_version=self.active_version,
            inactive_version=self.inactive_version,
            slot_capacity_bytes=self.slot_capacity_bytes,
            slot_storage_bytes=self.slot_storage_bytes,
            slot_storage_violation=self.slot_storage_violation,
            boot_attempt_counter=self.boot_attempt_counter,
            max_boot_attempts=self.max_boot_attempts,
            pending_activation=self.pending_activation,
            health_verdict=self.health_verdict,
            rollback_performed=self.rollback_performed,
            ab_rollback_ready=self.ab_rollback_ready,
            ab_update_valid=self.ab_update_valid,
            ab_finalized=self.ab_finalized,
            activation_success=self.activation_success,
            boot_health_success=self.boot_health_success,
            rollback_after_failed_boot=self.rollback_after_failed_boot,
            rollback_success=self.rollback_success,
            reboot_count=self.reboot_count,
            downtime_seconds=self.downtime_seconds,
            slot_switch_count=self.slot_switch_count,
        )


@dataclass
class DeploymentResult:
    state: DeploymentState
    completed: bool
    interrupted: bool
    valid: bool
    errors: list
    metrics: dict


def operation_payload_size(operation):
    metadata = json.dumps(operation, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_size = len(metadata)
    if operation.get("op") == "delta":
        payload_size += len(decode_bytes(operation["patch_b64"]))
    elif operation.get("op") in {"append", "raw_insert"}:
        payload_size += len(decode_bytes(operation["data_b64"]))
    return payload_size


def _sum_blocks(blocks):
    return sum(len(block) for block in blocks)


def _blocks_version(blocks):
    return bytes_sha256(blocks_to_bytes(blocks))


class DeploymentSimulator:
    def __init__(self, initial_blocks, block_size, config=None, state=None, expected_blocks=None):
        self.block_size = block_size
        self.config = config or DeploymentConfig()
        self.expected_blocks = list(expected_blocks) if expected_blocks is not None else None
        if state is None:
            blocks = list(initial_blocks)
            version = _blocks_version(blocks)
            self.state = DeploymentState(
                installed_blocks=blocks,
                snapshot_blocks=list(blocks),
                active_version=version,
                inactive_version=version,
                slot_capacity_bytes=self.config.slot_capacity_bytes,
                slot_storage_bytes=_sum_blocks(blocks),
                max_boot_attempts=self.config.max_boot_attempts,
                ab_rollback_ready=bool(self.config.enable_ab_slots),
            )
            self._refresh_peak_storage()
        else:
            self.state = state.clone()
            if self.state.slot_capacity_bytes is None:
                self.state.slot_capacity_bytes = self.config.slot_capacity_bytes
            self.state.max_boot_attempts = self.config.max_boot_attempts

    def clone(self):
        return DeploymentSimulator(
            initial_blocks=[],
            block_size=self.block_size,
            config=copy.deepcopy(self.config),
            state=self.state,
            expected_blocks=self.expected_blocks,
        )

    def latest_checkpoint(self):
        if not self.state.checkpoints:
            return None
        return self.state.checkpoints[-1].clone()

    def restore_checkpoint(self, checkpoint):
        self.state.installed_blocks = list(checkpoint.installed_blocks)
        self.state.backup_area = copy.deepcopy(checkpoint.backup_area)
        self.state.package_size_bytes = checkpoint.package_size_bytes
        self.state.network_bytes = checkpoint.network_bytes
        self.state.flash_write_bytes = checkpoint.flash_write_bytes
        self.state.install_time_s = checkpoint.install_time_s
        self.state.download_time_s = checkpoint.download_time_s
        self.state.applied_operations = checkpoint.operation_index
        self.state.install_state = "ready"
        self.state.staging_area_bytes = 0
        self.state.current_ram_bytes = 0
        self.state.peak_ram_bytes = checkpoint.peak_ram_bytes
        self.state.peak_persistent_storage_bytes = checkpoint.peak_persistent_storage_bytes
        self.state.rollback_ready = checkpoint.rollback_ready
        self.state.unsafe_overwrites = list(checkpoint.unsafe_overwrites)
        self.state.budget_violations = list(checkpoint.budget_violations)
        self.state.errors = list(checkpoint.errors)
        restored_checkpoints = [
            stored_checkpoint.clone()
            for stored_checkpoint in self.state.checkpoints
            if stored_checkpoint.operation_index <= checkpoint.operation_index
        ]
        if not any(
            stored_checkpoint.operation_index == checkpoint.operation_index
            for stored_checkpoint in restored_checkpoints
        ):
            restored_checkpoints.append(checkpoint.clone())
        self.state.checkpoints = restored_checkpoints
        self._refresh_peak_storage()

    def metrics(self):
        current_storage = self.current_persistent_storage_bytes()
        metrics = {
            "install_state": self.state.install_state,
            "package_size_bytes": self.state.package_size_bytes,
            "network_bytes": self.state.network_bytes,
            "staging_area_bytes": self.state.staging_area_bytes,
            "backup_area_bytes": self.backup_area_bytes(),
            "installed_image_bytes": _sum_blocks(self.state.installed_blocks),
            "current_persistent_storage_bytes": current_storage,
            "peak_persistent_storage_bytes": self.state.peak_persistent_storage_bytes,
            "peak_ram_bytes": self.state.peak_ram_bytes,
            "flash_write_bytes": self.state.flash_write_bytes,
            "download_time_s": round(self.state.download_time_s, 6),
            "install_time_s": round(self.state.install_time_s, 6),
            "total_time_s": round(self.state.download_time_s + self.state.install_time_s, 6),
            "checkpoint_count": len(self.state.checkpoints),
            "latest_checkpoint_operation": (
                self.state.checkpoints[-1].operation_index if self.state.checkpoints else 0
            ),
            "rollback_ready": self.state.rollback_ready,
            "unsafe_overwrite_count": len(self.state.unsafe_overwrites),
            "budget_violation_count": len(self.state.budget_violations),
            "budget_violations": list(self.state.budget_violations),
        }
        if self.config.enable_ab_slots:
            metrics.update(self._ab_metrics())
        return metrics

    def _ab_metrics(self):
        return {
            "ab_enabled": True,
            "active_slot": self.state.active_slot,
            "inactive_slot": self.state.inactive_slot,
            "active_version": self.state.active_version,
            "inactive_version": self.state.inactive_version,
            "slot_capacity_bytes": self.state.slot_capacity_bytes,
            "boot_attempt_counter": self.state.boot_attempt_counter,
            "max_boot_attempts": self.state.max_boot_attempts,
            "pending_activation": self.state.pending_activation,
            "health_verdict": self.state.health_verdict,
            "rollback_performed": self.state.rollback_performed,
            "ab_update_valid": self.state.ab_update_valid,
            "ab_rollback_ready": self.state.ab_rollback_ready,
            "slot_storage_bytes": self.state.slot_storage_bytes,
            "slot_storage_violation": self.state.slot_storage_violation,
            "activation_success": self.state.activation_success,
            "boot_health_success": self.state.boot_health_success,
            "rollback_after_failed_boot": self.state.rollback_after_failed_boot,
            "rollback_success": self.state.rollback_success,
            "reboot_count": self.state.reboot_count,
            "downtime_seconds": round(self.state.downtime_seconds, 6),
            "slot_switch_count": self.state.slot_switch_count,
        }

    def current_persistent_storage_bytes(self):
        return (
            _sum_blocks(self.state.installed_blocks)
            + self.backup_area_bytes()
            + self.state.staging_area_bytes
        )

    def backup_area_bytes(self):
        return sum(len(record.block) for record in self.state.backup_area.values())

    def apply_operation(self, operation):
        if self.state.install_state in {"failed", "interrupted"}:
            raise DeploymentError(f"cannot apply operation while state is {self.state.install_state}")

        try:
            self.state.install_state = "installing"
            self._download_operation(operation)
            self._apply_operation(operation)
            self.state.applied_operations += 1
            self._maybe_checkpoint(operation)
            if self.config.staging_strategy == "streaming":
                self.state.staging_area_bytes = 0
            self._refresh_peak_storage()
            self._check_budgets()
        except Exception as exc:
            self.state.install_state = "failed"
            self.state.errors.append(str(exc))
            raise

    def interrupt(self):
        self.state.install_state = "interrupted"

    def complete(self, expected_blocks=None):
        if (
            self.state.errors
            and not (
                self.config.enable_ab_slots
                and self.state.ab_finalized
                and self.state.install_state == "rolled_back"
            )
        ):
            self.state.install_state = "failed"
        elif self.state.install_state != "rolled_back":
            self.state.install_state = "complete"
        self.state.staging_area_bytes = 0
        self._refresh_peak_storage()
        if (
            self.config.enable_ab_slots
            and self.state.install_state == "complete"
            and not self.state.ab_finalized
        ):
            self._finalize_ab_update(expected_blocks=expected_blocks)
            self._refresh_peak_storage()

    def rollback(self, backup_id=None):
        if backup_id is None:
            self.state.installed_blocks = list(self.state.snapshot_blocks)
        else:
            if backup_id not in self.state.backup_area:
                raise DeploymentError(f"backup '{backup_id}' is not available for rollback")
            backup = self.state.backup_area[backup_id]
            self._write_block(backup.index, backup.block)
        self.state.install_state = "rolled_back"
        self.state.staging_area_bytes = 0
        self._refresh_peak_storage()

    def _download_operation(self, operation):
        op_bytes = operation_payload_size(operation) + self.config.metadata_overhead_bytes
        self.state.package_size_bytes += op_bytes
        self.state.network_bytes += op_bytes
        self.state.download_time_s += op_bytes / self.config.bandwidth_bytes_per_s
        if self.config.staging_strategy == "full_package":
            self.state.staging_area_bytes += op_bytes
        elif self.config.staging_strategy == "streaming":
            self.state.staging_area_bytes = op_bytes
        else:
            raise DeploymentError(f"unknown staging strategy '{self.config.staging_strategy}'")
        self._refresh_peak_storage()
        self._check_budgets()

    def _apply_operation(self, operation):
        op = operation.get("op")
        if op in {OperationType.KEEP.value, OperationType.SKIP.value}:
            target = operation["target"]
            self._require_index(target, f"{op} target")
            self._assert_target_hash(operation, self.state.installed_blocks[target])

        elif op == "backup":
            source = operation["source"]
            self._require_index(source, "backup source")
            block = self.state.installed_blocks[source]
            backup_id = operation.get("backup_id", f"backup-{source}")
            self.state.backup_area[backup_id] = BackupRecord(source, block)
            self._record_ram(len(block))
            self._record_flash_write(len(block))

        elif op == "copy":
            target = operation["target"]
            block = self._resolve_source_block(operation, "copy source")
            self._assert_target_hash(operation, block)
            self._mark_overwrite_if_unprotected(target, block)
            self._record_ram(len(block))
            self._write_block(target, block)

        elif op == "delta":
            codec = operation.get("codec", DeltaCodec.BSDIFF4.value)
            if codec != DeltaCodec.BSDIFF4.value:
                raise DeploymentError(f"delta codec '{codec}' is metadata-only; deployment supports bsdiff4")
            target = operation["target"]
            patch_bytes = decode_bytes(operation["patch_b64"])
            source_block = self._resolve_source_block(operation, "delta source")
            block = bsdiff4.patch(source_block, patch_bytes)
            self._assert_target_hash(operation, block)
            self._mark_overwrite_if_unprotected(target, block)
            self._record_ram(len(source_block) + len(patch_bytes) + len(block))
            self.state.install_time_s += (
                len(source_block) + len(patch_bytes)
            ) / self.config.patch_apply_bytes_per_s
            self._write_block(target, block)

        elif op == "raw_insert":
            target = operation["target"]
            block = decode_bytes(operation["data_b64"])
            self._assert_target_hash(operation, block)
            self._mark_overwrite_if_unprotected(target, block)
            self._record_ram(len(block))
            self._write_block(target, block)

        elif op == "append":
            block = decode_bytes(operation["data_b64"])
            target = operation.get("target", len(self.state.installed_blocks))
            if target != len(self.state.installed_blocks):
                raise DeploymentError(
                    f"append target {target} must equal current block count {len(self.state.installed_blocks)}"
                )
            self._assert_target_hash(operation, block)
            self._record_ram(len(block))
            self._write_block(target, block)

        elif op == "delete":
            target = operation["target"]
            count = operation.get("count", 1)
            if not isinstance(count, int) or count <= 0:
                raise DeploymentError("delete count must be a positive integer")
            self._require_index(target, "delete target")
            self._require_index(target + count - 1, "delete target")
            for deleted_index in range(target, target + count):
                self._mark_overwrite_if_unprotected(deleted_index)
            del self.state.installed_blocks[target:target + count]

        elif op == "truncate":
            if "new_size_bytes" in operation:
                new_size_bytes = operation["new_size_bytes"]
                if new_size_bytes < 0:
                    raise DeploymentError("truncate new_size_bytes cannot be negative")
                old_block_count = len(self.state.installed_blocks)
                data = blocks_to_bytes(self.state.installed_blocks)[:new_size_bytes]
                old_blocks = list(self.state.installed_blocks)
                self.state.installed_blocks = bytes_to_blocks(data, self.block_size)
                if (
                    self.state.installed_blocks
                    and len(self.state.installed_blocks) <= old_block_count
                    and self.state.installed_blocks[-1] != old_blocks[len(self.state.installed_blocks) - 1]
                ):
                    self._mark_overwrite_if_unprotected(len(self.state.installed_blocks) - 1)
                for target in range(len(self.state.installed_blocks), old_block_count):
                    self._mark_overwrite_if_unprotected(target)
            else:
                length = operation["length"]
                if length < 0:
                    raise DeploymentError("truncate length cannot be negative")
                for target in range(length, len(self.state.installed_blocks)):
                    self._mark_overwrite_if_unprotected(target)
                self.state.installed_blocks = self.state.installed_blocks[:length]

        elif op == "verify":
            size_bytes = operation.get("size_bytes")
            actual = bytes_sha256(blocks_to_bytes(self.state.installed_blocks, final_size=size_bytes))
            expected = operation.get("sha256") or operation.get("expected_sha256")
            if expected is not None and actual != expected:
                raise DeploymentError("verify sha256 mismatch")

        elif op == "rollback":
            self.rollback(operation.get("backup_id"))

        elif op == "checkpoint":
            self._create_checkpoint()

        elif op == "commit":
            self.complete()

        else:
            raise DeploymentError(f"unsupported operation '{op}'")

    def _write_block(self, target, block):
        if target < 0:
            raise DeploymentError(f"target index {target} cannot be negative")
        while len(self.state.installed_blocks) < target:
            self.state.installed_blocks.append(b"")
            self._record_flash_write(self.block_size)
        if target == len(self.state.installed_blocks):
            self.state.installed_blocks.append(block)
        else:
            self.state.installed_blocks[target] = block
        self._record_flash_write(len(block))
        self._refresh_peak_storage()

    def _record_ram(self, transient_bytes):
        self.state.current_ram_bytes = transient_bytes
        self.state.peak_ram_bytes = max(self.state.peak_ram_bytes, transient_bytes)
        self._check_budgets()
        self.state.current_ram_bytes = 0

    def _record_flash_write(self, bytes_written):
        self.state.flash_write_bytes += bytes_written
        self.state.install_time_s += bytes_written / self.config.flash_write_bytes_per_s

    def _create_checkpoint(self, operation_index=None):
        if operation_index is None:
            operation_index = self.state.applied_operations + 1
        self.state.checkpoints.append(
            DeploymentCheckpoint(
                operation_index=operation_index,
                installed_blocks=list(self.state.installed_blocks),
                backup_area=copy.deepcopy(self.state.backup_area),
                package_size_bytes=self.state.package_size_bytes,
                network_bytes=self.state.network_bytes,
                flash_write_bytes=self.state.flash_write_bytes,
                install_time_s=self.state.install_time_s,
                download_time_s=self.state.download_time_s,
                peak_ram_bytes=self.state.peak_ram_bytes,
                peak_persistent_storage_bytes=self.state.peak_persistent_storage_bytes,
                rollback_ready=self.state.rollback_ready,
                unsafe_overwrites=list(self.state.unsafe_overwrites),
                budget_violations=list(self.state.budget_violations),
                errors=list(self.state.errors),
            )
        )

    def _maybe_checkpoint(self, operation):
        if operation.get("op") == "checkpoint":
            return
        interval = self.config.checkpoint_interval_ops
        if interval and self.state.applied_operations > 0 and self.state.applied_operations % interval == 0:
            self._create_checkpoint(operation_index=self.state.applied_operations)

    def _refresh_peak_storage(self):
        self.state.peak_persistent_storage_bytes = max(
            self.state.peak_persistent_storage_bytes,
            self.current_persistent_storage_bytes(),
        )

    def _check_budgets(self):
        if (
            self.config.storage_budget_bytes is not None
            and self.current_persistent_storage_bytes() > self.config.storage_budget_bytes
        ):
            self._add_budget_violation("storage_budget_exceeded")
        if (
            self.config.ram_budget_bytes is not None
            and self.state.current_ram_bytes > self.config.ram_budget_bytes
        ):
            self._add_budget_violation("ram_budget_exceeded")

    def _add_budget_violation(self, violation):
        if violation not in self.state.budget_violations:
            self.state.budget_violations.append(violation)

    def _target_blocks_for_ab(self, expected_blocks):
        if expected_blocks is not None:
            return list(expected_blocks)
        if self.expected_blocks is not None:
            return list(self.expected_blocks)
        return list(self.state.installed_blocks)

    def _ab_health_verdict(self):
        if self.config.health_check_mode == "always_pass":
            return "pass"
        if self.config.health_check_mode == "forced_fail":
            return "fail"
        raise DeploymentError(
            f"unknown A/B health check mode '{self.config.health_check_mode}'"
        )

    def _fail_ab_before_activation(self, message, previous_active_blocks, previous_active_version):
        self.state.errors.append(message)
        self.state.installed_blocks = list(previous_active_blocks)
        self.state.active_version = previous_active_version
        self.state.pending_activation = False
        self.state.ab_update_valid = False
        self.state.ab_rollback_ready = True
        self.state.install_state = "failed"

    def _finalize_ab_update(self, expected_blocks=None):
        self.state.ab_finalized = True
        previous_active_slot = self.state.active_slot
        previous_inactive_slot = self.state.inactive_slot
        previous_active_blocks = list(self.state.snapshot_blocks)
        previous_active_version = self.state.active_version or _blocks_version(previous_active_blocks)
        candidate_blocks = list(self.state.installed_blocks)
        candidate_bytes = blocks_to_bytes(candidate_blocks)
        expected_target_blocks = self._target_blocks_for_ab(expected_blocks)
        expected_bytes = blocks_to_bytes(expected_target_blocks)
        candidate_version = bytes_sha256(candidate_bytes)

        self.state.slot_capacity_bytes = self.config.slot_capacity_bytes
        self.state.max_boot_attempts = self.config.max_boot_attempts
        self.state.slot_storage_bytes = len(candidate_bytes)
        self.state.inactive_version = candidate_version

        if not self.config.require_inactive_slot_install:
            self._finalize_active_slot_update(
                candidate_blocks=candidate_blocks,
                candidate_bytes=candidate_bytes,
                expected_bytes=expected_bytes,
                candidate_version=candidate_version,
                previous_active_blocks=previous_active_blocks,
                previous_active_version=previous_active_version,
            )
            return

        if (
            self.config.slot_capacity_bytes is not None
            and len(candidate_bytes) > self.config.slot_capacity_bytes
        ):
            self.state.slot_storage_violation = True
            self._add_budget_violation("slot_capacity_exceeded")
            self._fail_ab_before_activation(
                "A/B inactive slot capacity exceeded",
                previous_active_blocks,
                previous_active_version,
            )
            return

        if candidate_bytes != expected_bytes:
            self._fail_ab_before_activation(
                "A/B inactive slot content mismatch",
                previous_active_blocks,
                previous_active_version,
            )
            return

        self.state.pending_activation = True
        self.state.activation_success = True
        self.state.active_slot = previous_inactive_slot
        self.state.slot_switch_count += 1

        verdict = "not_run"
        self.state.boot_attempt_counter = 0
        for _ in range(self.config.max_boot_attempts):
            self.state.boot_attempt_counter += 1
            self.state.reboot_count += 1
            self.state.downtime_seconds += self.config.reboot_downtime_seconds
            verdict = self._ab_health_verdict()
            self.state.health_verdict = verdict
            if verdict == "pass":
                break

        if verdict == "pass":
            self.state.pending_activation = False
            self.state.boot_health_success = True
            self.state.ab_update_valid = True
            self.state.ab_rollback_ready = True
            self.state.active_slot = previous_inactive_slot
            self.state.inactive_slot = previous_active_slot
            self.state.active_version = candidate_version
            self.state.inactive_version = previous_active_version
            self.state.installed_blocks = candidate_blocks
            self.state.install_state = "complete"
            return

        self.state.pending_activation = False
        self.state.boot_health_success = False
        self.state.rollback_after_failed_boot = True
        self.state.rollback_performed = True
        self.state.active_slot = previous_active_slot
        self.state.inactive_slot = previous_inactive_slot
        self.state.active_version = previous_active_version
        self.state.inactive_version = candidate_version
        self.state.installed_blocks = previous_active_blocks
        self.state.slot_switch_count += 1
        self.state.rollback_success = (
            blocks_to_bytes(self.state.installed_blocks) == blocks_to_bytes(previous_active_blocks)
        )
        self.state.ab_rollback_ready = self.state.rollback_success
        self.state.ab_update_valid = False
        self.state.install_state = "rolled_back" if self.state.rollback_success else "failed"
        self.state.errors.append("A/B boot health check failed")

    def _finalize_active_slot_update(
        self,
        candidate_blocks,
        candidate_bytes,
        expected_bytes,
        candidate_version,
        previous_active_blocks,
        previous_active_version,
    ):
        self.state.inactive_version = previous_active_version
        if candidate_bytes != expected_bytes:
            self._fail_ab_before_activation(
                "active-slot content mismatch",
                previous_active_blocks,
                previous_active_version,
            )
            return

        self.state.activation_success = True
        verdict = "not_run"
        self.state.boot_attempt_counter = 0
        for _ in range(self.config.max_boot_attempts):
            self.state.boot_attempt_counter += 1
            self.state.reboot_count += 1
            self.state.downtime_seconds += self.config.reboot_downtime_seconds
            verdict = self._ab_health_verdict()
            self.state.health_verdict = verdict
            if verdict == "pass":
                break

        if verdict == "pass":
            self.state.boot_health_success = True
            self.state.ab_update_valid = True
            self.state.ab_rollback_ready = True
            self.state.active_version = candidate_version
            self.state.installed_blocks = candidate_blocks
            self.state.install_state = "complete"
            return

        self.state.boot_health_success = False
        self.state.rollback_after_failed_boot = True
        self.state.rollback_performed = True
        self.state.active_version = previous_active_version
        self.state.installed_blocks = previous_active_blocks
        self.state.rollback_success = (
            blocks_to_bytes(self.state.installed_blocks) == blocks_to_bytes(previous_active_blocks)
        )
        self.state.ab_rollback_ready = self.state.rollback_success
        self.state.ab_update_valid = False
        self.state.install_state = "rolled_back" if self.state.rollback_success else "failed"
        self.state.errors.append("A/B boot health check failed")

    def _mark_overwrite_if_unprotected(self, target, replacement_block=None):
        if not self.config.require_backup_for_overwrite:
            return
        if target >= len(self.state.snapshot_blocks):
            return
        if replacement_block == self.state.snapshot_blocks[target]:
            return
        if any(record.index == target for record in self.state.backup_area.values()):
            return
        self.state.rollback_ready = False
        marker = {
            "operation_index": self.state.applied_operations + 1,
            "target": target,
        }
        if marker not in self.state.unsafe_overwrites:
            self.state.unsafe_overwrites.append(marker)

    def _require_index(self, index, label):
        if index < 0 or index >= len(self.state.installed_blocks):
            raise DeploymentError(
                f"{label} index {index} is out of range for {len(self.state.installed_blocks)} blocks"
            )

    def _resolve_source_block(self, operation, label):
        source_area = operation.get("source_area", "installed")
        if source_area == "installed":
            source = operation["source"]
            self._require_index(source, label)
            return self.state.installed_blocks[source]
        if source_area == "backup":
            backup_id = operation.get("source_backup_id", operation["source"])
            if backup_id not in self.state.backup_area:
                raise DeploymentError(f"{label} backup '{backup_id}' is not available")
            return self.state.backup_area[backup_id].block
        if source_area == "image":
            return blocks_to_bytes(self.state.installed_blocks)
        raise DeploymentError(f"{label} source area '{source_area}' is not supported")

    def _assert_target_hash(self, operation, block):
        expected = operation.get("target_sha256")
        if expected and block_sha256(block) != expected:
            raise DeploymentError(
                f"{operation['op']} target hash mismatch for target {operation.get('target')}"
            )


def simulate_deployment(initial_blocks, operations, block_size, expected_blocks=None, config=None, stop_after=None):
    simulator = DeploymentSimulator(
        initial_blocks,
        block_size,
        config=config,
        expected_blocks=expected_blocks,
    )
    operations_to_apply = operations if stop_after is None else operations[:stop_after]
    interrupted = stop_after is not None and stop_after < len(operations)

    for operation in operations_to_apply:
        try:
            simulator.apply_operation(operation)
        except Exception:
            break

    if interrupted and simulator.state.install_state != "failed":
        simulator.interrupt()
    elif simulator.state.install_state != "failed":
        simulator.complete(expected_blocks=expected_blocks)

    expected_bytes = blocks_to_bytes(expected_blocks) if expected_blocks is not None else None
    final_bytes = blocks_to_bytes(simulator.state.installed_blocks)
    valid = (
        expected_bytes is not None
        and not simulator.state.errors
        and not interrupted
        and final_bytes == expected_bytes
    )
    return DeploymentResult(
        state=simulator.state,
        completed=not interrupted and not simulator.state.errors,
        interrupted=interrupted,
        valid=valid,
        errors=list(simulator.state.errors),
        metrics=simulator.metrics(),
    )


def simulate_update(old_file, new_file, operations, block_size, config=None, stop_after=None):
    initial_blocks = read_blocks(old_file, block_size)
    expected_blocks = read_blocks(new_file, block_size)
    return simulate_deployment(
        initial_blocks=initial_blocks,
        operations=operations,
        block_size=block_size,
        expected_blocks=expected_blocks,
        config=config,
        stop_after=stop_after,
    )
