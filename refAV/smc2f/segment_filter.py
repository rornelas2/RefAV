"""Utilities for applying SMc2f CLIP temporal segments to RefAV scenario dicts.

A RefAV scenario_dict is a nested tree:
    {track_uuid: child} where child is either
      * `list[int]`  — flat timestamp list                        (leaf)
      * `dict`       — related-object dict, recursing into `list[int]`

The CLIP coarse filter produces a list of allowed `(start_ts, end_ts)` windows.
`filter_scenario_by_segments` drops any timestamp outside those windows and
prunes tracks / subtrees that become empty.
"""

from __future__ import annotations

from typing import Iterable


def _ts_in_segments(ts: int, segments: list[tuple[int, int]]) -> bool:
    for start, end in segments:
        if start <= ts <= end:
            return True
    return False


def _filter_value(value, segments: list[tuple[int, int]]):
    """Recursively filter a scenario_dict value. Returns filtered value or None
    if it becomes empty.
    """
    if isinstance(value, dict):
        new_dict = {}
        for k, v in value.items():
            if isinstance(k, int) and not _ts_in_segments(k, segments):
                # k is a timestamp key outside all segments → drop
                continue
            filtered = _filter_value(v, segments)
            if filtered is None:
                continue
            new_dict[k] = filtered
        return new_dict if new_dict else None

    if isinstance(value, (list, tuple)):
        new_list = [
            ts for ts in value if not isinstance(ts, int) or _ts_in_segments(ts, segments)
        ]
        if not new_list:
            return None
        return type(value)(new_list) if isinstance(value, tuple) else new_list

    # Non-timestamp scalar (shouldn't normally appear) — preserve.
    return value


def filter_scenario_by_segments(
    scenario_dict: dict, segments: Iterable[tuple[int, int]]
) -> dict:
    """Return a filtered copy of `scenario_dict` keeping only timestamps
    that fall inside any of the given `(start_ts, end_ts)` segments.

    If `segments` is empty/None the original dict is returned unchanged
    (no-op), matching upstream SMc2f behavior.
    """
    segs = list(segments) if segments is not None else []
    if not segs:
        return scenario_dict

    filtered: dict = {}
    for track_uuid, child in scenario_dict.items():
        new_child = _filter_value(child, segs)
        if new_child is None:
            continue
        filtered[track_uuid] = new_child

    return filtered
