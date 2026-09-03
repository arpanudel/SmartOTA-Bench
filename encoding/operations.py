"""Structured OTA operation names and dictionary builders.

The helpers in this module intentionally return plain dictionaries so they can
be serialized directly and consumed by ``encoding.replay``.
"""

import base64
import hashlib
from dataclasses import dataclass
from enum import Enum


class OperationType(str, Enum):
    KEEP = "keep"
    SKIP = "skip"
    COPY = "copy"
    DELTA = "delta"
    RAW_INSERT = "raw_insert"
    BACKUP = "backup"
    APPEND = "append"
    DELETE = "delete"
    TRUNCATE = "truncate"
    VERIFY = "verify"
    CHECKPOINT = "checkpoint"
    COMMIT = "commit"
    ROLLBACK = "rollback"


class DeltaCodec(str, Enum):
    BSDIFF4 = "bsdiff4"
    XDELTA3 = "xdelta3"
    ZSTD = "zstd"


@dataclass(frozen=True)
class OperationSupport:
    replayable: bool
    deployment_supported: bool
    control_plane: bool = False
    notes: str = ""


OPERATION_SUPPORT = {
    OperationType.KEEP.value: OperationSupport(True, True, notes="No payload; validates an existing target block."),
    OperationType.SKIP.value: OperationSupport(True, True, notes="Alias-style no-op for already satisfied targets."),
    OperationType.COPY.value: OperationSupport(True, True, notes="Copies a block from installed or backup storage."),
    OperationType.DELTA.value: OperationSupport(True, True, notes="Replayable for bsdiff4 codec only."),
    OperationType.RAW_INSERT.value: OperationSupport(True, True, notes="Writes a full target block payload."),
    OperationType.BACKUP.value: OperationSupport(True, True, notes="Preserves rollback data."),
    OperationType.APPEND.value: OperationSupport(True, True, notes="Legacy grow-at-end raw block insertion."),
    OperationType.DELETE.value: OperationSupport(True, True, notes="Removes one or more full blocks."),
    OperationType.TRUNCATE.value: OperationSupport(True, True, notes="Shrinks by block count or exact byte size."),
    OperationType.VERIFY.value: OperationSupport(True, True, control_plane=True, notes="SHA-256 validation is enforced."),
    OperationType.CHECKPOINT.value: OperationSupport(True, True, control_plane=True, notes="Replay marker and deployment checkpoint."),
    OperationType.COMMIT.value: OperationSupport(True, True, control_plane=True, notes="Marks a validated update as finalized."),
    OperationType.ROLLBACK.value: OperationSupport(True, True, notes="Restores a backup or the full pre-update snapshot."),
}


def operation_value(operation):
    if isinstance(operation, OperationType):
        return operation.value
    return str(operation)


def codec_value(codec):
    if isinstance(codec, DeltaCodec):
        return codec.value
    return str(codec)


def encode_bytes(data):
    return base64.b64encode(data).decode("ascii")


def decode_bytes(data):
    return base64.b64decode(data.encode("ascii"))


def block_sha256(block):
    return hashlib.sha256(block).hexdigest()


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


def _with_target_hash(operation, target_block):
    if target_block is not None:
        operation["target_sha256"] = block_sha256(target_block)
    return operation


def make_keep_operation(target, target_block=None):
    operation = {
        "op": OperationType.KEEP.value,
        "target": target,
    }
    return _with_target_hash(operation, target_block)


def make_skip_operation(target, target_block=None):
    operation = {
        "op": OperationType.SKIP.value,
        "target": target,
    }
    return _with_target_hash(operation, target_block)


def make_backup_operation(source, backup_id=None):
    backup_id = backup_id or f"backup-{source}"
    return {
        "op": OperationType.BACKUP.value,
        "source": source,
        "backup_id": backup_id,
    }


def make_copy_operation(source, target, target_block=None, source_area="installed", source_backup_id=None):
    operation = {
        "op": OperationType.COPY.value,
        "source": source,
        "target": target,
    }
    if source_area != "installed":
        operation["source_area"] = source_area
    if source_backup_id is not None:
        operation["source_backup_id"] = source_backup_id
    return _with_target_hash(operation, target_block)


def make_delta_operation(
    source,
    target,
    patch_bytes,
    target_block=None,
    source_area="installed",
    source_backup_id=None,
    codec=DeltaCodec.BSDIFF4,
    codec_options=None,
):
    operation = {
        "op": OperationType.DELTA.value,
        "source": source,
        "target": target,
        "codec": codec_value(codec),
        "patch_b64": encode_bytes(patch_bytes),
    }
    if codec_options:
        operation["codec_options"] = dict(codec_options)
    if source_area != "installed":
        operation["source_area"] = source_area
    if source_backup_id is not None:
        operation["source_backup_id"] = source_backup_id
    return _with_target_hash(operation, target_block)


def make_raw_insert_operation(target, data):
    return {
        "op": OperationType.RAW_INSERT.value,
        "target": target,
        "data_b64": encode_bytes(data),
        "target_sha256": block_sha256(data),
    }


def make_append_operation(data, target=None):
    operation = {
        "op": OperationType.APPEND.value,
        "data_b64": encode_bytes(data),
        "target_sha256": block_sha256(data),
    }
    if target is not None:
        operation["target"] = target
    return operation


def make_delete_operation(target, count=1):
    operation = {
        "op": OperationType.DELETE.value,
        "target": target,
    }
    if count != 1:
        operation["count"] = count
    return operation


def make_truncate_operation(length=None, new_size_bytes=None):
    if length is None and new_size_bytes is None:
        raise ValueError("truncate requires length or new_size_bytes")
    operation = {"op": OperationType.TRUNCATE.value}
    if length is not None:
        operation["length"] = length
    if new_size_bytes is not None:
        operation["new_size_bytes"] = new_size_bytes
    return operation


def make_verify_operation(sha256=None, size_bytes=None, signature=None, metadata=None):
    operation = {"op": OperationType.VERIFY.value}
    if sha256 is not None:
        operation["sha256"] = sha256
    if size_bytes is not None:
        operation["size_bytes"] = size_bytes
    if signature is not None:
        operation["signature"] = signature
    if metadata:
        operation["metadata"] = dict(metadata)
    return operation


def make_checkpoint_operation(step_id=None):
    operation = {"op": OperationType.CHECKPOINT.value}
    if step_id is not None:
        operation["step_id"] = step_id
    return operation


def make_commit_operation():
    return {"op": OperationType.COMMIT.value}


def make_rollback_operation(backup_id=None):
    operation = {"op": OperationType.ROLLBACK.value}
    if backup_id is not None:
        operation["backup_id"] = backup_id
    return operation
