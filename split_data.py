"""
split_data.py – Split train.jsonl → train.jsonl (80%) + dev.jsonl (20%)
========================================================================

Finds every ``train.jsonl`` under ``benchmarks/*/`` and splits it in-place.
The original file is overwritten with the train portion; ``dev.jsonl`` is
written alongside it.

Usage
-----
# Split all benchmarks (default)
python split_data.py

# Split a specific benchmark only
python split_data.py --benchmark Cypherbench

# Custom ratio (e.g. 90/10)
python split_data.py --train-ratio 0.9

# Fix the random seed for reproducibility
python split_data.py --seed 42
"""

import json
import random
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split train.jsonl into train (80%) and dev (20%) sets"
    )
    parser.add_argument(
        "--base-dir",
        default=r"d:\text2cypher_distillation_draft",
        help="Project root directory",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help=(
            "Benchmark to split (e.g. Cypherbench). "
            "If omitted, all benchmarks under benchmarks/ are processed."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data to keep as training set (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling (preserve original order)",
    )
    return parser.parse_args()


def split_file(train_jsonl: Path, train_ratio: float, seed: int, shuffle: bool):
    """Read train.jsonl, split, and write train.jsonl + dev.jsonl in-place."""
    lines = train_jsonl.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]  # drop blank lines

    if not lines:
        print(f"  [SKIP] {train_jsonl} is empty.")
        return

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(lines)

    split_idx = max(1, int(len(lines) * train_ratio))
    train_lines = lines[:split_idx]
    dev_lines = lines[split_idx:]

    dev_jsonl = train_jsonl.parent / "dev.jsonl"

    # Overwrite train.jsonl with the train portion
    train_jsonl.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    # Write dev.jsonl
    dev_jsonl.write_text("\n".join(dev_lines) + "\n", encoding="utf-8")

    print(
        f"  [OK] {train_jsonl.parent.name}: "
        f"{len(train_lines)} train / {len(dev_lines)} dev "
        f"(total {len(lines)})"
    )


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    benchmarks_dir = base_dir / "benchmarks"

    if args.benchmark:
        benchmark_dirs = [benchmarks_dir / args.benchmark]
    else:
        benchmark_dirs = sorted(
            p for p in benchmarks_dir.iterdir() if p.is_dir()
        )

    print(
        f"Split ratio: {args.train_ratio:.0%} train / {1 - args.train_ratio:.0%} dev  |  "
        f"seed={args.seed}  |  shuffle={'no' if args.no_shuffle else 'yes'}\n"
    )

    found_any = False
    for bdir in benchmark_dirs:
        train_jsonl = bdir / "train.jsonl"
        if not train_jsonl.exists():
            print(f"  [SKIP] {bdir.name}: train.jsonl not found (run format_data.py first)")
            continue
        found_any = True
        print(f"Processing {bdir.name} ...")
        split_file(
            train_jsonl=train_jsonl,
            train_ratio=args.train_ratio,
            seed=args.seed,
            shuffle=not args.no_shuffle,
        )

    if not found_any:
        print(
            "\nNo train.jsonl files found. "
            "Run format_data.py first to generate them."
        )
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
