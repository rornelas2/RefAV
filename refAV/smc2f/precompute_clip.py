"""One-shot CLI to precompute SMc2f CLIP features for a log list.

Usage:
    python -m refAV.smc2f.precompute_clip \
        --log-prompt-pairs /path/to/log_prompt_pairs_val.json \
        --split val \
        --av2-root /home/rornelas5/scratch/argoverse_data/sensor \
        [--device cuda] [--limit 30]

Idempotent: logs with an existing `meta.json` under the cache dir are
skipped. The cache location comes from `SMc2fConfig.CLIP_CACHE_DIR`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clip_coarse_filter import CLIPFeatureExtractor
from config import SMc2fConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SMc2f CLIP features")
    parser.add_argument(
        "--log-prompt-pairs",
        type=Path,
        required=True,
        help="Path to log_prompt_pairs_<split>.json — only the log_ids are used.",
    )
    parser.add_argument("--split", type=str, required=True, help="train | val | test")
    parser.add_argument(
        "--av2-root",
        type=Path,
        required=True,
        help="Root containing <split>/<log_id>/sensors/cameras/...",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of logs to process (for smoke tests / probes).",
    )
    args = parser.parse_args()

    with open(args.log_prompt_pairs, "r") as f:
        lpp = json.load(f)

    log_ids = list(lpp.keys())
    if args.limit is not None:
        log_ids = log_ids[: args.limit]

    print(f"[precompute_clip] target logs: {len(log_ids)}")
    print(f"[precompute_clip] cache dir:   {SMc2fConfig.CLIP_CACHE_DIR}")
    print(f"[precompute_clip] av2 root:    {args.av2_root}")
    print(f"[precompute_clip] split:       {args.split}")

    extractor = CLIPFeatureExtractor(device=args.device)
    extractor.extract_dataset(args.split, args.av2_root, log_ids)
    print("[precompute_clip] done.")


if __name__ == "__main__":
    main()
