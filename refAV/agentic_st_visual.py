from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from refAV.agentic_st_config import AgenticSearchConfig
from refAV.agentic_st_ops import infer_category_mentions
from refAV.utils import cache_manager, get_best_crop, get_img_crop


COLOR_WORDS = ("white", "silver", "black", "red", "yellow", "blue")
CAMERA_ALIASES = {
    "front camera": "ring_front_center",
    "front-center camera": "ring_front_center",
    "front left camera": "ring_front_left",
    "front-right camera": "ring_front_right",
    "rear camera": "ring_rear_center",
    "side left camera": "ring_side_left",
    "side right camera": "ring_side_right",
    "ring_front_center": "ring_front_center",
    "ring_front_left": "ring_front_left",
    "ring_front_right": "ring_front_right",
    "ring_rear_center": "ring_rear_center",
    "ring_side_left": "ring_side_left",
    "ring_side_right": "ring_side_right",
}
WEATHER_HINTS = {
    "snow": "snow",
    "snowy": "snow",
}
_SCENE_STATS_CACHE: dict[tuple[str, int, str], dict[str, float]] = {}


@dataclass
class VisualTrackEvidence:
    track_uuid: str
    category: str
    timestamp: int
    camera_name: str
    crop_score: float
    visual_score: float
    color: Optional[str] = None


@dataclass
class VisualSceneEvidence:
    timestamp: int
    camera_name: str
    score: float
    attributes: dict[str, float] = field(default_factory=dict)


@dataclass
class VisualCoarseContext:
    summary: str
    top_timestamps: list[int] = field(default_factory=list)
    top_tracks: list[str] = field(default_factory=list)
    track_evidence: list[VisualTrackEvidence] = field(default_factory=list)
    scene_evidence: list[VisualSceneEvidence] = field(default_factory=list)
    planner_images: list[tuple[str, int]] = field(default_factory=list)
    relevant_categories: list[str] = field(default_factory=list)
    mentioned_colors: list[str] = field(default_factory=list)
    mentioned_cameras: list[str] = field(default_factory=list)
    weather_hints: list[str] = field(default_factory=list)
    scene_attributes: dict[int, dict[str, float]] = field(default_factory=dict)


def extract_prompt_visual_hints(prompt: str) -> tuple[list[str], list[str], list[str]]:
    prompt_lower = prompt.lower()
    mentioned_colors = [color for color in COLOR_WORDS if color in prompt_lower]
    mentioned_cameras = []
    for phrase, camera_name in CAMERA_ALIASES.items():
        if phrase in prompt_lower and camera_name not in mentioned_cameras:
            mentioned_cameras.append(camera_name)
    weather_hints = []
    for phrase, weather in WEATHER_HINTS.items():
        if phrase in prompt_lower and weather not in weather_hints:
            weather_hints.append(weather)
    return mentioned_colors, mentioned_cameras, weather_hints


def _image_to_stats(image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB").resize((160, 90)), dtype=np.float32) / 255.0
    brightness = float(rgb.mean())
    channel_range = rgb.max(axis=2) - rgb.min(axis=2)
    saturation = float(channel_range.mean())
    whiteness = float(((rgb.mean(axis=2) > 0.72) & (channel_range < 0.12)).mean())
    snow_score = float(np.clip(0.50 * whiteness + 0.35 * brightness + 0.15 * (1.0 - saturation), 0.0, 1.0))
    return {
        "brightness": brightness,
        "saturation": saturation,
        "whiteness": whiteness,
        "snow_score": snow_score,
    }


def get_scene_visual_stats(log_dir: Path, timestamp: int, camera_name: str) -> dict[str, float]:
    key = (str(Path(log_dir)), int(timestamp), camera_name)
    if key in _SCENE_STATS_CACHE:
        return _SCENE_STATS_CACHE[key]

    image = get_img_crop(camera_name, int(timestamp), Path(log_dir))
    if image is None:
        stats = {}
    else:
        stats = _image_to_stats(image)
    _SCENE_STATS_CACHE[key] = stats
    return stats


class VisualCoarseFilter:
    def __init__(self, config: AgenticSearchConfig):
        self.config = config

    @staticmethod
    def _sample_timestamps(timestamps: list[int], budget: int) -> list[int]:
        if len(timestamps) <= budget:
            return timestamps
        indices = np.round(np.linspace(0, len(timestamps) - 1, num=budget)).astype(int)
        unique_indices = sorted(set(indices.tolist()))
        return [timestamps[index] for index in unique_indices]

    def build_context(self, prompt: str, log_dir: Path) -> VisualCoarseContext:
        log_dir = Path(log_dir)
        try:
            cache_manager.load_custom_caches(log_dir)
        except Exception:
            pass

        try:
            annotations_df = pd.read_feather(log_dir / "annotations.feather")
        except Exception as exc:
            return VisualCoarseContext(summary=f"Visual coarse filter unavailable: {exc}")

        relevant_categories = infer_category_mentions(prompt)
        mentioned_colors, mentioned_cameras, weather_hints = extract_prompt_visual_hints(prompt)
        candidate_df = annotations_df
        if relevant_categories:
            category_df = annotations_df[annotations_df["category"].isin(relevant_categories)]
            if not category_df.empty:
                candidate_df = category_df

        grouped = (
            candidate_df.groupby(["track_uuid", "category"])
            .size()
            .reset_index(name="num_timestamps")
            .sort_values("num_timestamps", ascending=False)
            .head(self.config.visual_coarse_track_budget)
        )

        track_evidence: list[VisualTrackEvidence] = []
        for row in grouped.itertuples(index=False):
            track_uuid = str(row.track_uuid)
            category = str(row.category)
            try:
                best_crop = get_best_crop(track_uuid, log_dir)
            except Exception:
                best_crop = None
            if not best_crop:
                continue

            track_color = None
            if cache_manager.color_cache:
                track_color = cache_manager.color_cache.get(track_uuid)

            visual_score = float(best_crop.get("score", 0.0))
            if track_color is not None and track_color in mentioned_colors:
                visual_score += 1.0
            if mentioned_cameras and best_crop["cam"] in mentioned_cameras:
                visual_score += 0.5
            if "snow" in weather_hints:
                stats = get_scene_visual_stats(log_dir, int(best_crop["timestamp"]), self.config.visual_scene_camera)
                visual_score += 0.75 * stats.get("snow_score", 0.0)

            track_evidence.append(
                VisualTrackEvidence(
                    track_uuid=track_uuid,
                    category=category,
                    timestamp=int(best_crop["timestamp"]),
                    camera_name=str(best_crop["cam"]),
                    crop_score=float(best_crop.get("score", 0.0)),
                    visual_score=visual_score,
                    color=track_color,
                )
            )

        track_evidence.sort(key=lambda item: item.visual_score, reverse=True)

        all_timestamps = sorted(annotations_df["timestamp_ns"].unique().tolist())
        sampled_timestamps = self._sample_timestamps(all_timestamps, self.config.visual_coarse_timestamp_budget)
        scene_attributes: dict[int, dict[str, float]] = {}
        scene_evidence: list[VisualSceneEvidence] = []
        timestamp_scores: dict[int, float] = {}

        for evidence in track_evidence[: self.config.visual_coarse_track_budget]:
            timestamp_scores[evidence.timestamp] = max(timestamp_scores.get(evidence.timestamp, 0.0), evidence.visual_score)

        if "snow" in weather_hints or not timestamp_scores:
            for timestamp in sampled_timestamps:
                stats = get_scene_visual_stats(log_dir, int(timestamp), self.config.visual_scene_camera)
                if not stats:
                    continue
                scene_attributes[int(timestamp)] = stats
                score = stats.get("snow_score", 0.0) if "snow" in weather_hints else stats.get("brightness", 0.0)
                scene_evidence.append(
                    VisualSceneEvidence(
                        timestamp=int(timestamp),
                        camera_name=self.config.visual_scene_camera,
                        score=float(score),
                        attributes=stats,
                    )
                )
                timestamp_scores[int(timestamp)] = max(timestamp_scores.get(int(timestamp), 0.0), float(score))

        scene_evidence.sort(key=lambda item: item.score, reverse=True)
        if timestamp_scores:
            top_timestamps = [
                timestamp
                for timestamp, _ in sorted(timestamp_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.visual_coarse_timestamp_budget]
            ]
        else:
            top_timestamps = sampled_timestamps[: self.config.visual_coarse_timestamp_budget]
        top_tracks = [item.track_uuid for item in track_evidence[: self.config.visual_coarse_track_budget]]

        planner_images: list[tuple[str, int]] = []
        for evidence in scene_evidence[: self.config.visual_planner_num_images]:
            planner_images.append((evidence.camera_name, evidence.timestamp))
        for evidence in track_evidence[: self.config.visual_planner_num_images]:
            candidate = (evidence.camera_name, evidence.timestamp)
            if candidate not in planner_images:
                planner_images.append(candidate)
        for timestamp in top_timestamps:
            candidate = (self.config.visual_scene_camera, timestamp)
            if candidate not in planner_images:
                planner_images.append(candidate)
        planner_images = planner_images[: self.config.visual_planner_num_images]

        summary = self._format_summary(
            prompt=prompt,
            relevant_categories=relevant_categories,
            mentioned_colors=mentioned_colors,
            mentioned_cameras=mentioned_cameras,
            weather_hints=weather_hints,
            track_evidence=track_evidence,
            scene_evidence=scene_evidence,
            top_timestamps=top_timestamps,
        )

        return VisualCoarseContext(
            summary=summary,
            top_timestamps=top_timestamps,
            top_tracks=top_tracks,
            track_evidence=track_evidence,
            scene_evidence=scene_evidence,
            planner_images=planner_images,
            relevant_categories=relevant_categories,
            mentioned_colors=mentioned_colors,
            mentioned_cameras=mentioned_cameras,
            weather_hints=weather_hints,
            scene_attributes=scene_attributes,
        )

    @staticmethod
    def _format_summary(
        prompt: str,
        relevant_categories: list[str],
        mentioned_colors: list[str],
        mentioned_cameras: list[str],
        weather_hints: list[str],
        track_evidence: list[VisualTrackEvidence],
        scene_evidence: list[VisualSceneEvidence],
        top_timestamps: list[int],
    ) -> str:
        lines = [f"Visual coarse-filter summary for prompt: {prompt}"]
        lines.append(f"Relevant categories: {', '.join(relevant_categories) if relevant_categories else 'none inferred'}")
        lines.append(f"Mentioned colors: {', '.join(mentioned_colors) if mentioned_colors else 'none'}")
        lines.append(f"Mentioned cameras: {', '.join(mentioned_cameras) if mentioned_cameras else 'none'}")
        lines.append(f"Weather hints: {', '.join(weather_hints) if weather_hints else 'none'}")
        if track_evidence:
            lines.append("Top visually-grounded tracks:")
            for evidence in track_evidence[:3]:
                color_text = evidence.color or "unknown"
                lines.append(
                    f"- uuid={evidence.track_uuid} category={evidence.category} ts={evidence.timestamp} "
                    f"cam={evidence.camera_name} visual_score={evidence.visual_score:.3f} color={color_text}"
                )
        else:
            lines.append("Top visually-grounded tracks: none")
        if scene_evidence:
            lines.append("Top scene timestamps:")
            for evidence in scene_evidence[:3]:
                snow_score = evidence.attributes.get("snow_score")
                if snow_score is None:
                    lines.append(f"- ts={evidence.timestamp} cam={evidence.camera_name} score={evidence.score:.3f}")
                else:
                    lines.append(
                        f"- ts={evidence.timestamp} cam={evidence.camera_name} score={evidence.score:.3f} snow_score={snow_score:.3f}"
                    )
        else:
            lines.append("Top scene timestamps: none")
        if top_timestamps:
            lines.append(f"Suggested timestamps to prioritize: {top_timestamps[:6]}")
        return "\n".join(lines)
