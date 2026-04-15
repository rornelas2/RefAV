from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class OperationSpec:
    name: str
    kind: str
    description: str
    required_params: tuple[str, ...] = ()
    optional_params: dict[str, Any] = field(default_factory=dict)
    allows_related_source: bool = False


COMMON_CATEGORY_ALIASES: dict[str, str] = {
    "ego vehicle": "EGO_VEHICLE",
    "vehicle": "REGULAR_VEHICLE",
    "car": "REGULAR_VEHICLE",
    "pedestrian": "PEDESTRIAN",
    "cyclist": "BICYCLIST",
    "bicyclist": "BICYCLIST",
    "bicycle": "BICYCLE",
    "motorcyclist": "MOTORCYCLIST",
    "motorcycle": "MOTORCYCLE",
    "bus": "BUS",
    "truck": "TRUCK",
    "sign": "SIGN",
    "stop sign": "STOP_SIGN",
    "bollard": "BOLLARD",
    "trailer": "VEHICULAR_TRAILER",
    "large vehicle": "LARGE_VEHICLE",
    "construction barrel": "CONSTRUCTION_BARREL",
    "construction cone": "CONSTRUCTION_CONE",
}


DEFAULT_OPERATION_SPECS: dict[str, OperationSpec] = {
    "get_objects_of_category": OperationSpec(
        name="get_objects_of_category",
        kind="source",
        description="Create a scenario dict from a category.",
        required_params=("category",),
    ),
    "turning": OperationSpec(
        name="turning",
        kind="unary",
        description="Filter tracks that are turning in a direction.",
        optional_params={"direction": None},
    ),
    "changing_lanes": OperationSpec(
        name="changing_lanes",
        kind="unary",
        description="Filter tracks changing lanes.",
        optional_params={"direction": None},
    ),
    "accelerating": OperationSpec(
        name="accelerating",
        kind="unary",
        description="Filter tracks by longitudinal acceleration.",
        optional_params={"min_accel": 0.0, "max_accel": float("inf")},
    ),
    "has_velocity": OperationSpec(
        name="has_velocity",
        kind="unary",
        description="Filter tracks by speed.",
        optional_params={"min_velocity": 0.0, "max_velocity": float("inf")},
    ),
    "stationary": OperationSpec(
        name="stationary",
        kind="unary",
        description="Filter stationary tracks.",
    ),
    "near_intersection": OperationSpec(
        name="near_intersection",
        kind="unary",
        description="Filter tracks near an intersection.",
    ),
    "on_intersection": OperationSpec(
        name="on_intersection",
        kind="unary",
        description="Filter tracks inside an intersection.",
    ),
    "at_stop_sign": OperationSpec(
        name="at_stop_sign",
        kind="unary",
        description="Filter tracks at a stop sign.",
    ),
    "at_pedestrian_crossing": OperationSpec(
        name="at_pedestrian_crossing",
        kind="unary",
        description="Filter tracks at a crosswalk.",
    ),
    "on_road": OperationSpec(
        name="on_road",
        kind="unary",
        description="Filter tracks on the road.",
    ),
    "in_drivable_area": OperationSpec(
        name="in_drivable_area",
        kind="unary",
        description="Filter tracks in drivable area.",
    ),
    "within_camera_view": OperationSpec(
        name="within_camera_view",
        kind="unary",
        description="Filter tracks visible in a camera.",
        required_params=("camera_name",),
    ),
    "is_color": OperationSpec(
        name="is_color",
        kind="unary",
        description="Filter tracks by coarse visual color when color caches are available.",
        required_params=("color",),
    ),
    "is_snowy_scene": OperationSpec(
        name="is_snowy_scene",
        kind="unary",
        description="Filter timestamps that visually resemble a snowy scene from camera imagery.",
        optional_params={"camera_name": "ring_front_center", "snow_score_thresh": 0.58},
    ),
    "within_time_window": OperationSpec(
        name="within_time_window",
        kind="unary",
        description=(
            "Keep tracks whose upstream-filtered timestamps form at least min_events "
            "distinct event segments whose starts all fit inside window_seconds. Use "
            "for repeated-event prompts like 'two right turns within 15 seconds'. "
            "Source must be a unary event filter (turning, accelerating, ...)."
        ),
        optional_params={"window_seconds": 15.0, "min_events": 2, "event_gap_seconds": 0.5},
    ),
    "on_lane_type": OperationSpec(
        name="on_lane_type",
        kind="unary",
        description="Filter tracks by lane type.",
        required_params=("lane_type",),
    ),
    "on_relative_side_of_road": OperationSpec(
        name="on_relative_side_of_road",
        kind="unary",
        description="Filter tracks by road side.",
        required_params=("side",),
    ),
    "has_objects_in_relative_direction": OperationSpec(
        name="has_objects_in_relative_direction",
        kind="relational",
        description="Filter tracks with related objects in a direction.",
        required_params=("direction",),
        optional_params={
            "min_number": 1,
            "max_number": float("inf"),
            "within_distance": 50.0,
            "lateral_thresh": float("inf"),
        },
        allows_related_source=True,
    ),
    "get_objects_in_relative_direction": OperationSpec(
        name="get_objects_in_relative_direction",
        kind="relational",
        description="Return related objects in a direction.",
        required_params=("direction",),
        optional_params={
            "min_number": 1,
            "max_number": float("inf"),
            "within_distance": 50.0,
            "lateral_thresh": float("inf"),
        },
        allows_related_source=True,
    ),
    "near_objects": OperationSpec(
        name="near_objects",
        kind="relational",
        description="Filter tracks near related objects.",
        optional_params={"distance_thresh": 10.0},
        allows_related_source=True,
    ),
    "following": OperationSpec(
        name="following",
        kind="relational",
        description="Filter tracks following related objects.",
        optional_params={"distance_thresh": 20.0},
        allows_related_source=True,
    ),
    "facing_toward": OperationSpec(
        name="facing_toward",
        kind="relational",
        description="Filter tracks facing toward related objects.",
        allows_related_source=True,
    ),
    "heading_toward": OperationSpec(
        name="heading_toward",
        kind="relational",
        description="Filter tracks heading toward related objects.",
        allows_related_source=True,
    ),
    "heading_in_relative_direction_to": OperationSpec(
        name="heading_in_relative_direction_to",
        kind="relational",
        description="Filter heading relation between objects.",
        required_params=("direction",),
        allows_related_source=True,
    ),
    "being_crossed_by": OperationSpec(
        name="being_crossed_by",
        kind="relational",
        description="Filter tracks being crossed by related objects.",
        allows_related_source=True,
    ),
    "in_same_lane": OperationSpec(
        name="in_same_lane",
        kind="relational",
        description="Filter tracks in the same lane as related objects.",
        allows_related_source=True,
    ),
    "union": OperationSpec(
        name="union",
        kind="relational",
        description="Merge two flat scenario dicts (OR logic). Use to express 'A or B' over the same track category.",
        allows_related_source=True,
    ),
}


def build_dsl_reference() -> str:
    lines = [
        "Program schema:",
        "{",
        '  "description": "<original prompt>",',
        '  "result": "<step name>",',
        '  "steps": [',
        "    {",
        '      "name": "vehicles",',
        '      "op": "get_objects_of_category",',
        '      "params": {"category": "REGULAR_VEHICLE"}',
        "    }",
        "  ]",
        "}",
        "",
        "Allowed operations:",
    ]
    for spec in DEFAULT_OPERATION_SPECS.values():
        line = f"- {spec.name} [{spec.kind}]"
        if spec.required_params:
            line += f" required={list(spec.required_params)}"
        if spec.optional_params:
            line += f" optional={list(spec.optional_params.keys())}"
        if spec.allows_related_source:
            line += " uses related_source"
        line += f" :: {spec.description}"
        lines.append(line)
    return "\n".join(lines)


def infer_category_mentions(prompt: str) -> list[str]:
    prompt_lower = prompt.lower()
    mentions: list[tuple[int, int, str]] = []
    for phrase, category in COMMON_CATEGORY_ALIASES.items():
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
        for match in re.finditer(pattern, prompt_lower):
            mentions.append((match.start(), match.end(), category))
    mentions.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    ordered = []
    seen = set()
    occupied_spans: list[tuple[int, int]] = []
    for start, end, category in mentions:
        overlaps_existing = any(not (end <= span_start or start >= span_end) for span_start, span_end in occupied_spans)
        if overlaps_existing:
            continue
        occupied_spans.append((start, end))
        if category not in seen:
            ordered.append(category)
            seen.add(category)
    return ordered
