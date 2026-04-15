from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from refAV.agentic_st_config import AgenticSearchConfig
from refAV.agentic_st_visual import VisualCoarseContext, get_scene_visual_stats
from refAV.utils import cache_manager, get_scenario_timestamps, reconstruct_track_dict


@dataclass
class RerankBreakdown:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    feedback: str = ""


class TrajectoryReranker:
    def __init__(self, config: AgenticSearchConfig):
        self.config = config

    def score(
        self,
        prompt: str,
        scenario_dict: dict,
        log_dir: Path,
        visual_context: VisualCoarseContext | None = None,
    ) -> RerankBreakdown:
        if not scenario_dict:
            return RerankBreakdown(total=0.0, feedback="reranker skipped: empty scenario")

        try:
            cache_manager.load_custom_caches(Path(log_dir))
        except Exception:
            pass

        track_dict = reconstruct_track_dict(scenario_dict)
        predicted_tracks = list(track_dict.keys())
        predicted_timestamps = set(get_scenario_timestamps(scenario_dict))
        if not predicted_tracks or not predicted_timestamps:
            return RerankBreakdown(total=0.0, feedback="reranker skipped: empty track/timestamp set")

        components: dict[str, float] = {}
        if visual_context is not None and visual_context.top_timestamps:
            coarse_timestamps = set(visual_context.top_timestamps)
            components["timestamp_overlap"] = len(predicted_timestamps & coarse_timestamps) / len(predicted_timestamps)

        if visual_context is not None and visual_context.mentioned_colors and cache_manager.color_cache:
            matched_colors = 0
            for track_uuid in predicted_tracks:
                if cache_manager.color_cache.get(str(track_uuid)) in visual_context.mentioned_colors:
                    matched_colors += 1
            components["color_match"] = matched_colors / max(len(predicted_tracks), 1)

        if visual_context is not None and "snow" in visual_context.weather_hints:
            snow_scores = []
            for timestamp in list(sorted(predicted_timestamps))[: self.config.visual_coarse_timestamp_budget]:
                if timestamp in visual_context.scene_attributes:
                    snow_scores.append(visual_context.scene_attributes[timestamp].get("snow_score", 0.0))
                else:
                    stats = get_scene_visual_stats(Path(log_dir), int(timestamp), self.config.visual_scene_camera)
                    if stats:
                        snow_scores.append(stats.get("snow_score", 0.0))
            if snow_scores:
                components["snow_scene_match"] = sum(snow_scores) / len(snow_scores)

        if not components:
            return RerankBreakdown(total=0.0, feedback="reranker skipped: no visual signals available")

        component_avg = sum(components.values()) / len(components)
        total = self.config.reranker_weight * component_avg
        feedback = ", ".join(f"{name}={value:.3f}" for name, value in sorted(components.items()))
        return RerankBreakdown(total=total, components=components, feedback=feedback)
