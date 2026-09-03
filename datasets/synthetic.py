import json
import os
import random
import zlib
from pathlib import Path

from .manifest import compute_file_metadata


SYNTHETIC_CASES = (
    "growing",
    "shrinking",
    "reordered",
    "repeated",
    "random",
    "compressed_like",
    "adversarial_reorder",
    "adversarial_grow",
    "adversarial_shrink",
    "adversarial_corruption",
)


SCENARIOS = {
    "growing": "target image appends new blocks to an otherwise stable source image",
    "shrinking": "target image removes trailing and middle content from the source image",
    "reordered": "target image reorders existing blocks without changing total size",
    "repeated": "target image contains repeated source blocks and duplicate targets",
    "random": "target image changes high-entropy blocks with weak source similarity",
    "compressed_like": "target image resembles compressed binary payload drift",
    "adversarial_reorder": "target image heavily reorders repeated and near-duplicate source blocks",
    "adversarial_grow": "target image inserts new blocks at head, middle, and tail while preserving reusable regions",
    "adversarial_shrink": "target image removes repeated regions and truncates the final block",
    "adversarial_corruption": "target image contains deterministic byte corruptions across sparse source blocks",
}


def generate_synthetic_dataset(
    output_dir,
    manifest_path=None,
    *,
    name="smartota-smoke",
    block_size_bytes=1024,
    block_count=12,
    seed=20260501,
):
    if block_size_bytes <= 0:
        raise ValueError("block_size_bytes must be positive")
    if block_count < 6:
        raise ValueError("block_count must be at least 6")

    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else output_dir / f"{name}.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for case_name in SYNTHETIC_CASES:
        old_bytes, new_bytes = _build_case(case_name, block_size_bytes, block_count, seed)
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        old_path = case_dir / "old.bin"
        new_path = case_dir / "new.bin"
        old_path.write_bytes(old_bytes)
        new_path.write_bytes(new_bytes)
        pairs.append(
            _pair_record(
                case_name,
                old_path,
                new_path,
                output_dir,
                block_size_bytes,
                seed,
                block_count,
            )
        )

    manifest = {
        "name": name,
        "version": 1,
        "base_dir": _posix_relpath(output_dir, manifest_path.parent),
        "pairs": pairs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _build_case(case_name, block_size_bytes, block_count, seed):
    if case_name == "growing":
        blocks = _named_blocks("growing", block_size_bytes, block_count + 4, seed)
        return b"".join(blocks[:block_count]), b"".join(blocks[: block_count + 4])

    if case_name == "shrinking":
        blocks = _named_blocks("shrinking", block_size_bytes, block_count + 3, seed)
        old_blocks = blocks[: block_count + 3]
        new_blocks = old_blocks[:2] + old_blocks[4:block_count] + old_blocks[-2:-1]
        return b"".join(old_blocks), b"".join(new_blocks)

    if case_name == "reordered":
        blocks = _named_blocks("reordered", block_size_bytes, block_count, seed)
        order = list(range(block_count))
        order = order[2::3] + order[0::3] + order[1::3]
        return b"".join(blocks), b"".join(blocks[index] for index in order)

    if case_name == "repeated":
        blocks = _named_blocks("repeated", block_size_bytes, 4, seed)
        old_indexes = [0, 1, 0, 2, 0, 1, 3, 2, 0, 1, 3, 3]
        new_indexes = [0, 0, 1, 2, 1, 0, 3, 3, 2, 0, 1, 0]
        return (
            b"".join(blocks[index] for index in old_indexes),
            b"".join(blocks[index] for index in new_indexes),
        )

    if case_name == "random":
        rng = random.Random(seed)
        old_blocks = [_random_bytes(rng, block_size_bytes) for _ in range(block_count)]
        new_blocks = list(old_blocks)
        for index in range(1, block_count, 3):
            new_blocks[index] = _random_bytes(rng, block_size_bytes)
        new_blocks[-1] = _flip_byte(new_blocks[-1], block_size_bytes // 2)
        return b"".join(old_blocks), b"".join(new_blocks)

    if case_name == "compressed_like":
        old_payload = _structured_payload("compressed-old", block_size_bytes, block_count, seed)
        new_payload = _structured_payload("compressed-new", block_size_bytes, block_count + 1, seed)
        return zlib.compress(old_payload, level=9), zlib.compress(new_payload, level=9)

    if case_name == "adversarial_reorder":
        base_blocks = _named_blocks("adversarial-reorder", block_size_bytes, max(6, block_count // 2), seed)
        old_indexes = list(range(len(base_blocks))) + list(reversed(range(len(base_blocks))))
        while len(old_indexes) < block_count:
            old_indexes.append(len(old_indexes) % len(base_blocks))
        old_indexes = old_indexes[:block_count]
        new_indexes = old_indexes[1::2] + old_indexes[::2]
        new_blocks = [base_blocks[index] for index in new_indexes[:block_count]]
        for index in range(2, len(new_blocks), 5):
            new_blocks[index] = _flip_byte(new_blocks[index], index)
        return (
            b"".join(base_blocks[index] for index in old_indexes),
            b"".join(new_blocks),
        )

    if case_name == "adversarial_grow":
        stable_blocks = _named_blocks("adversarial-grow-stable", block_size_bytes, block_count, seed)
        inserted_blocks = _named_blocks("adversarial-grow-insert", block_size_bytes, 4, seed)
        old_blocks = stable_blocks[:block_count]
        midpoint = max(1, block_count // 2)
        new_blocks = (
            inserted_blocks[:1]
            + old_blocks[:midpoint]
            + inserted_blocks[1:3]
            + old_blocks[midpoint:]
            + inserted_blocks[3:]
        )
        return b"".join(old_blocks), b"".join(new_blocks)

    if case_name == "adversarial_shrink":
        blocks = _named_blocks("adversarial-shrink", block_size_bytes, block_count + 4, seed)
        old_blocks = blocks[:2] + blocks[2:6] * 2 + blocks[6:block_count + 2]
        new_blocks = old_blocks[:2] + old_blocks[5:8] + old_blocks[-3:-1]
        new_bytes = b"".join(new_blocks)
        return b"".join(old_blocks), new_bytes[: max(1, len(new_bytes) - block_size_bytes // 3)]

    if case_name == "adversarial_corruption":
        old_blocks = _named_blocks("adversarial-corruption", block_size_bytes, block_count, seed)
        new_blocks = list(old_blocks)
        for index in range(0, block_count, 2):
            block = new_blocks[index]
            for offset in (0, len(block) // 3, (2 * len(block)) // 3):
                block = _flip_byte(block, offset)
            new_blocks[index] = block
        return b"".join(old_blocks), b"".join(new_blocks)

    raise ValueError(f"unknown synthetic case: {case_name}")


def _pair_record(case_name, old_path, new_path, base_dir, block_size_bytes, seed, block_count):
    old_metadata = compute_file_metadata(old_path)
    new_metadata = compute_file_metadata(new_path)
    return {
        "id": f"synthetic_{case_name}",
        "enabled": True,
        "domain": "synthetic",
        "scenario": SCENARIOS[case_name],
        "old_file": _posix_relpath(old_path, base_dir),
        "new_file": _posix_relpath(new_path, base_dir),
        "old_size_bytes": old_metadata.size_bytes,
        "new_size_bytes": new_metadata.size_bytes,
        "old_sha256": old_metadata.sha256,
        "new_sha256": new_metadata.sha256,
        "block_size_bytes": block_size_bytes,
        "source": f"deterministic synthetic generator seed={seed}",
        "license_notes": "generated test data; no third-party artifacts",
        "synthetic_case": case_name,
        "generator": "datasets.synthetic.generate_synthetic_dataset",
        "seed": seed,
        "block_count": block_count,
    }


def _named_blocks(case_name, block_size_bytes, count, seed):
    return [
        _expand_digest(f"{seed}:{case_name}:{index}".encode("utf-8"), block_size_bytes)
        for index in range(count)
    ]


def _structured_payload(label, block_size_bytes, block_count, seed):
    rows = []
    for index in range(block_count * 8):
        prefix = f"{seed}:{label}:{index % 17:02d}:".encode("utf-8")
        rows.append((prefix + _expand_digest(prefix, 64))[:96])
    payload = b"\n".join(rows)
    return payload[: block_size_bytes * block_count]


def _expand_digest(seed_bytes, size):
    import hashlib

    output = bytearray()
    counter = 0
    while len(output) < size:
        output.extend(hashlib.sha256(seed_bytes + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(output[:size])


def _random_bytes(rng, size):
    return bytes(rng.getrandbits(8) for _ in range(size))


def _flip_byte(data, index):
    if not data:
        return data
    index = min(max(index, 0), len(data) - 1)
    mutated = bytearray(data)
    mutated[index] ^= 0xFF
    return bytes(mutated)


def _posix_relpath(path, start):
    return Path(os.path.relpath(Path(path), Path(start))).as_posix()
