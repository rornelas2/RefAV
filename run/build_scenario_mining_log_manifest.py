#!/usr/bin/env python3
"""Build a unique log-id manifest for selective AV2 scenario-mining downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-prompt-pairs",
        type=Path,
        default=None,
        help="Path to log_prompt_pairs_<split>.json",
    )
    parser.add_argument(
        "--annotations-feather",
        type=Path,
        default=None,
        help="Path to scenario_mining_<split>_annotations.feather",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output .txt manifest with one log_id per line.",
    )
    args = parser.parse_args()
    if args.log_prompt_pairs is None and args.annotations_feather is None:
        raise ValueError("Pass at least one of --log-prompt-pairs or --annotations-feather.")
    return args


def load_log_ids_from_json(path: Path) -> set[str]:
    with open(path, "r") as f:
        data = json.load(f)
    return {str(log_id) for log_id in data.keys()}


def load_log_ids_from_feather(path: Path) -> set[str]:
    df = pd.read_feather(path, columns=["log_id"])
    return {str(log_id) for log_id in df["log_id"].unique().tolist()}


def main() -> None:
    args = parse_args()
    log_ids: set[str] = set()

    if args.log_prompt_pairs is not None:
        log_ids |= load_log_ids_from_json(args.log_prompt_pairs)

    if args.annotations_feather is not None:
        log_ids |= load_log_ids_from_feather(args.annotations_feather)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for log_id in sorted(log_ids):
            f.write(f"{log_id}\n")

    print(f"Wrote {len(log_ids)} unique log IDs to {args.output}")


if __name__ == "__main__":
    main()
