#!/usr/bin/env python3
"""Convert raw AV2 sensor logs into a RefAV-ready layout.

This script upgrades `annotations.feather` files so RefAV can treat the ego
vehicle like any other track. For each timestamp present in annotations, it
adds a synthetic `EGO_VEHICLE` row derived from `city_SE3_egovehicle.feather`.

Usage examples:

Create a separate RefAV-ready tree with symlinks to the original data:
    python run/convert_av2_sensor_logs_to_refav.py \
        /path/to/sensor/val \
        --output-root /path/to/refav_sensor/val

Upgrade a single log in place:
    python run/convert_av2_sensor_logs_to_refav.py \
        /path/to/sensor/val/<log_id> \
        --in-place
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


EGO_TRACK_UUID = "ego"
EGO_CATEGORY = "EGO_VEHICLE"
EGO_LENGTH_M = 4.877
EGO_WIDTH_M = 2.0
EGO_HEIGHT_M = 1.473


def is_log_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "annotations.feather").exists()
        and (path / "city_SE3_egovehicle.feather").exists()
    )


def discover_log_dirs(input_path: Path) -> list[Path]:
    if is_log_dir(input_path):
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_path}")

    log_dirs = sorted(path for path in input_path.iterdir() if is_log_dir(path))
    if not log_dirs:
        raise FileNotFoundError(
            f"No AV2 log directories found under {input_path}. "
            "Expected child directories with both annotations.feather and "
            "city_SE3_egovehicle.feather."
        )
    return log_dirs


def build_ego_rows(log_dir: Path, annotations_df: pd.DataFrame) -> pd.DataFrame:
    ego_df = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    ego_df = ego_df.copy()
    ego_df["log_id"] = log_dir.name
    ego_df["track_uuid"] = EGO_TRACK_UUID
    ego_df["category"] = EGO_CATEGORY
    ego_df["length_m"] = EGO_LENGTH_M
    ego_df["width_m"] = EGO_WIDTH_M
    ego_df["height_m"] = EGO_HEIGHT_M
    ego_df["qw"] = 1.0
    ego_df["qx"] = 0.0
    ego_df["qy"] = 0.0
    ego_df["qz"] = 0.0
    ego_df["tx_m"] = 0.0
    ego_df["ty_m"] = 0.0
    ego_df["tz_m"] = 0.0
    ego_df["num_interior_pts"] = 1

    if "score" in annotations_df.columns:
        ego_df["score"] = 1.0

    synced_timestamps = annotations_df["timestamp_ns"].unique()
    ego_df = ego_df[ego_df["timestamp_ns"].isin(synced_timestamps)].copy()
    return ego_df


def make_refav_annotations(log_dir: Path, force: bool = False) -> tuple[pd.DataFrame, bool]:
    annotations_df = pd.read_feather(log_dir / "annotations.feather")
    has_ego = (annotations_df["category"] == EGO_CATEGORY).any()
    if has_ego and not force:
        return annotations_df, False

    base_df = annotations_df[annotations_df["category"] != EGO_CATEGORY].copy()
    ego_df = build_ego_rows(log_dir, base_df)
    combined_df = pd.concat([base_df, ego_df], ignore_index=True, sort=False)
    combined_df = combined_df.sort_values(["timestamp_ns", "track_uuid"], kind="stable").reset_index(drop=True)

    # Preserve the original column order, appending any ego-only columns at the end.
    ordered_columns = list(annotations_df.columns)
    for column in combined_df.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)
    combined_df = combined_df[ordered_columns]
    return combined_df, True


def ensure_symlink_tree(src_log_dir: Path, dst_log_dir: Path) -> None:
    dst_log_dir.mkdir(parents=True, exist_ok=True)
    for child in src_log_dir.iterdir():
        if child.name == "annotations.feather":
            continue
        dst_child = dst_log_dir / child.name
        if dst_child.exists() or dst_child.is_symlink():
            continue
        os.symlink(child.resolve(), dst_child)


def write_converted_log(
    src_log_dir: Path,
    input_root: Path,
    output_root: Path | None,
    in_place: bool,
    force: bool,
) -> tuple[str, str]:
    converted_df, changed = make_refav_annotations(src_log_dir, force=force)
    if not changed:
        return src_log_dir.name, "already_refav_ready"

    if in_place:
        out_log_dir = src_log_dir
    else:
        relative_parent = src_log_dir.parent.relative_to(input_root) if input_root != src_log_dir else Path()
        out_log_dir = output_root / relative_parent / src_log_dir.name
        ensure_symlink_tree(src_log_dir, out_log_dir)

    out_log_dir.mkdir(parents=True, exist_ok=True)
    converted_df.to_feather(out_log_dir / "annotations.feather")
    return src_log_dir.name, f"wrote:{out_log_dir}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a single AV2 log directory or a directory containing many logs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional destination root for converted logs. When provided, the script "
        "creates a RefAV-ready tree with symlinks to the original non-annotation files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite annotations.feather in the source logs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild ego rows even if the log already contains EGO_VEHICLE entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.in_place and args.output_root is not None:
        raise ValueError("Choose either --in-place or --output-root, not both.")
    if not args.in_place and args.output_root is None:
        raise ValueError("Pass either --in-place or --output-root.")

    input_path = args.input_path.resolve()
    log_dirs = discover_log_dirs(input_path)
    input_root = input_path if input_path != log_dirs[0] else log_dirs[0]

    if args.output_root is not None:
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = None

    converted = 0
    skipped = 0
    for log_dir in log_dirs:
        log_id, status = write_converted_log(
            log_dir,
            input_root=input_root,
            output_root=output_root,
            in_place=args.in_place,
            force=args.force,
        )
        print(f"{log_id}: {status}")
        if status == "already_refav_ready":
            skipped += 1
        else:
            converted += 1

    print(f"Finished. Converted {converted} log(s), skipped {skipped} already-ready log(s).")


if __name__ == "__main__":
    main()
