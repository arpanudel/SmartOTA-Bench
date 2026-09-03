import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import generate_synthetic_dataset


def main(args):
    manifest_path = generate_synthetic_dataset(
        args.output_dir,
        manifest_path=args.manifest,
        name=args.name,
        block_size_bytes=args.block_size_bytes,
        block_count=args.block_count,
        seed=args.seed,
    )
    print(f"Wrote synthetic dataset manifest: {manifest_path}")
    print(f"Wrote synthetic artifacts under: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate deterministic SmartOTA-Bench synthetic update pairs."
    )
    parser.add_argument("--output-dir", default="data/processed/synthetic")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--name", default="smartota-smoke")
    parser.add_argument("--block-size-bytes", type=int, default=1024)
    parser.add_argument("--block-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260501)
    main(parser.parse_args())
