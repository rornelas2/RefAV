from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from refAV.agentic_st_config import AgenticSearchConfig
from refAV.utils import (
    get_related_objects,
    reconstruct_relationship_dict,
    reconstruct_track_dict,
    swap_keys_and_listed_values,
)


@dataclass
class RewardBreakdown:
    total: float
    referred_track_f1: float
    related_track_f1: float
    referred_timestamp_f1: float
    related_timestamp_f1: float
    runtime_penalty: float
    feedback: str


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _build_role_maps_from_gt(gt_df: pd.DataFrame) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    referred_df = gt_df[gt_df["mining_category"] == "REFERRED_OBJECT"]
    related_df = gt_df[gt_df["mining_category"] == "RELATED_OBJECT"]
    referred = {
        int(timestamp): set(group["track_uuid"].astype(str).tolist())
        for timestamp, group in referred_df.groupby("timestamp_ns")
    }
    related = {
        int(timestamp): set(group["track_uuid"].astype(str).tolist())
        for timestamp, group in related_df.groupby("timestamp_ns")
    }
    return referred, related


def _build_role_maps_from_scenario(scenario_dict: dict) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    track_dict = reconstruct_track_dict(scenario_dict)
    relationship_dict = reconstruct_relationship_dict(scenario_dict)
    related_dict = get_related_objects(relationship_dict)
    referred = {
        int(timestamp): set(map(str, track_uuids))
        for timestamp, track_uuids in swap_keys_and_listed_values(track_dict).items()
    }
    related = {
        int(timestamp): set(map(str, track_uuids))
        for timestamp, track_uuids in swap_keys_and_listed_values(related_dict).items()
    }
    return referred, related


def _track_level_f1(timestamp_to_ids_pred: dict[int, set[str]], timestamp_to_ids_gt: dict[int, set[str]]) -> float:
    pred_ids = set().union(*timestamp_to_ids_pred.values()) if timestamp_to_ids_pred else set()
    gt_ids = set().union(*timestamp_to_ids_gt.values()) if timestamp_to_ids_gt else set()
    tp = len(pred_ids & gt_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)
    return _safe_f1(tp, fp, fn)


def _timestamp_level_f1(timestamp_to_ids_pred: dict[int, set[str]], timestamp_to_ids_gt: dict[int, set[str]]) -> float:
    timestamps = sorted(set(timestamp_to_ids_pred.keys()) | set(timestamp_to_ids_gt.keys()))
    if not timestamps:
        return 0.0
    f1_scores = []
    for timestamp in timestamps:
        pred = timestamp_to_ids_pred.get(timestamp, set())
        gt = timestamp_to_ids_gt.get(timestamp, set())
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        f1_scores.append(_safe_f1(tp, fp, fn))
    return float(np.mean(f1_scores))


def score_scenario_dict(
    scenario_dict: dict,
    gt_prompt_df: Optional[pd.DataFrame],
    runtime_sec: float,
    config: AgenticSearchConfig,
) -> RewardBreakdown:
    if gt_prompt_df is None or gt_prompt_df.empty:
        non_empty = float(bool(scenario_dict))
        runtime_penalty = min(runtime_sec / max(config.runtime_timeout_sec, 1), 1.0) * config.reward_runtime_penalty_weight
        total = non_empty * (0.2 + config.reward_empty_prediction_bias) - runtime_penalty
        feedback = "No GT provided; used unsupervised prior favoring concise non-empty programs."
        return RewardBreakdown(
            total=total,
            referred_track_f1=0.0,
            related_track_f1=0.0,
            referred_timestamp_f1=0.0,
            related_timestamp_f1=0.0,
            runtime_penalty=runtime_penalty,
            feedback=feedback,
        )

    gt_referred, gt_related = _build_role_maps_from_gt(gt_prompt_df)
    pred_referred, pred_related = _build_role_maps_from_scenario(scenario_dict or {})

    referred_track_f1 = _track_level_f1(pred_referred, gt_referred)
    related_track_f1 = _track_level_f1(pred_related, gt_related)
    referred_timestamp_f1 = _timestamp_level_f1(pred_referred, gt_referred)
    related_timestamp_f1 = _timestamp_level_f1(pred_related, gt_related)

    # --- Concrete diff for the planner to reflect on (agentic-loop signal) ---
    pred_referred_ids = set().union(*pred_referred.values()) if pred_referred else set()
    gt_referred_ids = set().union(*gt_referred.values()) if gt_referred else set()
    missed_tracks = sorted(gt_referred_ids - pred_referred_ids)[:3]
    spurious_tracks = sorted(pred_referred_ids - gt_referred_ids)[:3]
    pred_ts = sorted(pred_referred.keys())
    gt_ts = sorted(gt_referred.keys())
    diff_lines = []
    if not pred_referred_ids:
        diff_lines.append("DIFF: prediction is EMPTY — relax filters or check category mention")
    else:
        if missed_tracks:
            diff_lines.append(f"DIFF: missed_referred_track_uuids={missed_tracks}")
        if spurious_tracks:
            diff_lines.append(f"DIFF: spurious_referred_track_uuids={spurious_tracks}")
        if gt_ts and pred_ts:
            diff_lines.append(
                f"DIFF: gt_timestamp_window=[{min(gt_ts)},{max(gt_ts)}] (n={len(gt_ts)}); "
                f"pred_timestamp_window=[{min(pred_ts)},{max(pred_ts)}] (n={len(pred_ts)})"
            )

    runtime_penalty = min(runtime_sec / max(config.runtime_timeout_sec, 1), 1.0) * config.reward_runtime_penalty_weight
    # Weights aligned with competition HOTA metric emphasis:
    # HOTA-Temporal and Timestamp BA are heavily timestamp-dependent
    total = (
        0.30 * referred_track_f1
        + 0.10 * related_track_f1
        + 0.40 * referred_timestamp_f1
        + 0.20 * related_timestamp_f1
        - runtime_penalty
    )

    feedback_parts = [
        f"referred_track_f1={referred_track_f1:.3f}",
        f"related_track_f1={related_track_f1:.3f}",
        f"referred_timestamp_f1={referred_timestamp_f1:.3f}",
        f"related_timestamp_f1={related_timestamp_f1:.3f}",
    ]
    # Actionable guidance for the LLM planner
    if referred_track_f1 < 0.5:
        feedback_parts.append("ACTION: wrong referred tracks — check category and primary filters")
    elif referred_timestamp_f1 < referred_track_f1 - 0.2:
        feedback_parts.append("ACTION: right tracks but wrong timestamps — add temporal filters (turning, accelerating, etc.)")
    if related_track_f1 < 0.3 and related_timestamp_f1 < 0.3:
        feedback_parts.append("ACTION: related objects missing — check relational filter (direction, distance)")
    feedback_parts.extend(diff_lines)
    feedback = ", ".join(feedback_parts)
    return RewardBreakdown(
        total=total,
        referred_track_f1=referred_track_f1,
        related_track_f1=related_track_f1,
        referred_timestamp_f1=referred_timestamp_f1,
        related_timestamp_f1=related_timestamp_f1,
        runtime_penalty=runtime_penalty,
        feedback=feedback,
    )
