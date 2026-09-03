import copy
from dataclasses import dataclass, field
from pathlib import Path

import bsdiff4

from encoding.operations import (
    DeltaCodec,
    OperationType,
    block_sha256,
    bytes_sha256,
    decode_bytes,
    encode_bytes,
    make_append_operation,
    make_backup_operation,
    make_checkpoint_operation,
    make_commit_operation,
    make_copy_operation,
    make_delete_operation,
    make_delta_operation,
    make_keep_operation,
    make_raw_insert_operation,
    make_rollback_operation,
    make_skip_operation,
    make_truncate_operation,
    make_verify_operation,
)


class ReplayError(Exception):
    """Raised when an OTA operation cannot be replayed safely."""


@dataclass
class BackupRecord:
    index: int
    block: bytes


@dataclass
class ReplayState:
    blocks: list
    snapshot_blocks: list
    block_size: int | None = None
    backups: dict = field(default_factory=dict)
    applied_count: int = 0
    history: list = field(default_factory=list)
    peak_blocks: int = 0
    rolled_back: bool = False
    committed: bool = False
    verified_count: int = 0

    def __post_init__(self):
        if self.peak_blocks == 0:
            self.peak_blocks = len(self.blocks)

    def clone(self):
        return ReplayState(
            blocks=list(self.blocks),
            snapshot_blocks=list(self.snapshot_blocks),
            block_size=self.block_size,
            backups=copy.deepcopy(self.backups),
            applied_count=self.applied_count,
            history=list(self.history),
            peak_blocks=self.peak_blocks,
            rolled_back=self.rolled_back,
            committed=self.committed,
            verified_count=self.verified_count,
        )


@dataclass
class ReplayResult:
    state: ReplayState
    errors: list
    completed: bool
    interrupted: bool
    applied_operations: int
    total_operations: int
    valid: bool = False
    final_size_bytes: int = 0
    expected_size_bytes: int = 0
    final_sha256: str = ""
    expected_sha256: str = ""

    @property
    def backup_count(self):
        return len(self.state.backups)

    @property
    def rolled_back(self):
        return self.state.rolled_back

    @property
    def committed(self):
        return self.state.committed


def read_blocks(path, block_size):
    blocks = []
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            blocks.append(block)
    return blocks


def blocks_to_bytes(blocks, final_size=None):
    data = b"".join(blocks)
    if final_size is not None:
        return data[:final_size]
    return data


def bytes_to_blocks(data, block_size):
    if block_size is None or block_size <= 0:
        raise ReplayError("byte-size truncate requires a positive block_size")
    return [data[index:index + block_size] for index in range(0, len(data), block_size)]


def _require_index(blocks, index, label):
    if index < 0 or index >= len(blocks):
        raise ReplayError(f"{label} index {index} is out of range for {len(blocks)} blocks")


def _write_block(state, target, block):
    if target < 0:
        raise ReplayError(f"target index {target} cannot be negative")
    while len(state.blocks) < target:
        state.blocks.append(b"")
    if target == len(state.blocks):
        state.blocks.append(block)
    else:
        state.blocks[target] = block
    state.peak_blocks = max(state.peak_blocks, len(state.blocks))


def _validate_target_hash(operation, block):
    expected = operation.get("target_sha256")
    if expected and block_sha256(block) != expected:
        raise ReplayError(
            f"{operation['op']} target hash mismatch for target {operation.get('target')}"
        )


def _require_positive_count(count, label):
    if not isinstance(count, int) or count <= 0:
        raise ReplayError(f"{label} count must be a positive integer")


def _resolve_source_block(state, operation, label):
    source_area = operation.get("source_area", "installed")
    if source_area == "installed":
        source = operation["source"]
        _require_index(state.blocks, source, label)
        return state.blocks[source]
    if source_area == "backup":
        backup_id = operation.get("source_backup_id", operation["source"])
        if backup_id not in state.backups:
            raise ReplayError(f"{label} backup '{backup_id}' does not exist")
        return state.backups[backup_id].block
    if source_area == "image":
        return blocks_to_bytes(state.blocks)
    raise ReplayError(f"{label} source area '{source_area}' is not supported")


def apply_operation(state, operation):
    op = operation.get("op")
    if not op:
        raise ReplayError("operation is missing required 'op' field")

    if op in {OperationType.KEEP.value, OperationType.SKIP.value}:
        target = operation["target"]
        _require_index(state.blocks, target, f"{op} target")
        _validate_target_hash(operation, state.blocks[target])

    elif op == "backup":
        source = operation["source"]
        _require_index(state.blocks, source, "backup source")
        backup_id = operation.get("backup_id", f"backup-{source}")
        state.backups[backup_id] = BackupRecord(source, state.blocks[source])

    elif op == "copy":
        target = operation["target"]
        block = _resolve_source_block(state, operation, "copy source")
        _validate_target_hash(operation, block)
        _write_block(state, target, block)

    elif op == "delta":
        codec = operation.get("codec", DeltaCodec.BSDIFF4.value)
        if codec != DeltaCodec.BSDIFF4.value:
            raise ReplayError(f"delta codec '{codec}' is metadata-only; replay supports bsdiff4")
        target = operation["target"]
        patch_bytes = decode_bytes(operation["patch_b64"])
        source_block = _resolve_source_block(state, operation, "delta source")
        block = bsdiff4.patch(source_block, patch_bytes)
        _validate_target_hash(operation, block)
        _write_block(state, target, block)

    elif op == "raw_insert":
        target = operation["target"]
        block = decode_bytes(operation["data_b64"])
        _validate_target_hash(operation, block)
        _write_block(state, target, block)

    elif op == "append":
        target = operation.get("target")
        if target is not None and target != len(state.blocks):
            raise ReplayError(
                f"append target {target} must equal current block count {len(state.blocks)}"
            )
        block = decode_bytes(operation["data_b64"])
        _validate_target_hash(operation, block)
        state.blocks.append(block)
        state.peak_blocks = max(state.peak_blocks, len(state.blocks))

    elif op == "delete":
        target = operation["target"]
        count = operation.get("count", 1)
        _require_positive_count(count, "delete")
        _require_index(state.blocks, target, "delete target")
        _require_index(state.blocks, target + count - 1, "delete target")
        del state.blocks[target:target + count]

    elif op == "truncate":
        if "new_size_bytes" in operation:
            new_size_bytes = operation["new_size_bytes"]
            if new_size_bytes < 0:
                raise ReplayError("truncate new_size_bytes cannot be negative")
            data = blocks_to_bytes(state.blocks)[:new_size_bytes]
            state.blocks = bytes_to_blocks(data, state.block_size)
        else:
            length = operation["length"]
            if length < 0:
                raise ReplayError("truncate length cannot be negative")
            state.blocks = state.blocks[:length]

    elif op == "rollback":
        backup_id = operation.get("backup_id")
        if backup_id is None:
            state.blocks = list(state.snapshot_blocks)
            state.rolled_back = True
        else:
            if backup_id not in state.backups:
                raise ReplayError(f"rollback backup '{backup_id}' does not exist")
            backup = state.backups[backup_id]
            _write_block(state, backup.index, backup.block)
            state.rolled_back = True

    elif op == "verify":
        size_bytes = operation.get("size_bytes")
        actual = bytes_sha256(blocks_to_bytes(state.blocks, final_size=size_bytes))
        expected = operation.get("sha256") or operation.get("expected_sha256")
        if expected is not None and actual != expected:
            raise ReplayError("verify sha256 mismatch")
        state.verified_count += 1

    elif op == "checkpoint":
        pass

    elif op == "commit":
        state.committed = True

    else:
        raise ReplayError(f"unsupported operation '{op}'")

    state.applied_count += 1
    state.history.append(operation)
    return state


def replay_operations(initial_blocks, operations, state=None, stop_after=None, strict=True, block_size=None):
    if state is None:
        blocks = list(initial_blocks)
        state = ReplayState(blocks=blocks, snapshot_blocks=list(blocks), block_size=block_size)
    else:
        state = state.clone()
        if block_size is not None:
            state.block_size = block_size

    errors = []
    operations_to_apply = operations
    if stop_after is not None:
        operations_to_apply = operations[:stop_after]

    for operation in operations_to_apply:
        try:
            apply_operation(state, operation)
        except Exception as exc:
            errors.append(str(exc))
            if strict:
                break

    interrupted = stop_after is not None and stop_after < len(operations)
    completed = not errors and not interrupted and len(operations_to_apply) == len(operations)
    return ReplayResult(
        state=state,
        errors=errors,
        completed=completed,
        interrupted=interrupted,
        applied_operations=state.applied_count,
        total_operations=state.applied_count + max(0, len(operations) - len(operations_to_apply)),
        final_size_bytes=len(blocks_to_bytes(state.blocks)),
        final_sha256=bytes_sha256(blocks_to_bytes(state.blocks)),
    )


def validate_update(old_file, new_file, operations, block_size, stop_after=None, state=None):
    expected_bytes = Path(new_file).read_bytes()
    if state is None:
        initial_blocks = read_blocks(old_file, block_size)
    else:
        initial_blocks = []

    result = replay_operations(
        initial_blocks=initial_blocks,
        operations=operations,
        state=state,
        stop_after=stop_after,
        block_size=block_size,
    )
    final_bytes = blocks_to_bytes(result.state.blocks)
    result.final_size_bytes = len(final_bytes)
    result.expected_size_bytes = len(expected_bytes)
    result.final_sha256 = bytes_sha256(final_bytes)
    result.expected_sha256 = bytes_sha256(expected_bytes)
    result.valid = not result.errors and result.completed and final_bytes == expected_bytes
    return result
