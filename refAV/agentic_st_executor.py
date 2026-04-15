from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import traceback
from typing import Optional

from refAV.agentic_st_compiler import compile_program_to_code, validate_program
from refAV.agentic_st_dsl import ScenarioProgram
from refAV.atomic_functions import (
    accelerating,
    at_pedestrian_crossing,
    at_stop_sign,
    being_crossed_by,
    changing_lanes,
    facing_toward,
    following,
    get_objects_in_relative_direction,
    get_objects_of_category,
    has_objects_in_relative_direction,
    has_velocity,
    heading_in_relative_direction_to,
    heading_toward,
    in_drivable_area,
    in_same_lane,
    is_color,
    is_snowy_scene,
    near_intersection,
    near_objects,
    on_intersection,
    on_lane_type,
    on_relative_side_of_road,
    on_road,
    output_scenario,
    stationary,
    turning,
    union,
    within_camera_view,
    within_time_window,
)
from refAV.utils import cache_manager


OPERATION_FNS = {
    "get_objects_of_category": get_objects_of_category,
    "turning": turning,
    "changing_lanes": changing_lanes,
    "accelerating": accelerating,
    "has_velocity": has_velocity,
    "stationary": stationary,
    "near_intersection": near_intersection,
    "on_intersection": on_intersection,
    "at_stop_sign": at_stop_sign,
    "at_pedestrian_crossing": at_pedestrian_crossing,
    "on_road": on_road,
    "in_drivable_area": in_drivable_area,
    "within_camera_view": within_camera_view,
    "is_color": is_color,
    "is_snowy_scene": is_snowy_scene,
    "on_lane_type": on_lane_type,
    "on_relative_side_of_road": on_relative_side_of_road,
    "has_objects_in_relative_direction": has_objects_in_relative_direction,
    "get_objects_in_relative_direction": get_objects_in_relative_direction,
    "near_objects": near_objects,
    "following": following,
    "facing_toward": facing_toward,
    "heading_toward": heading_toward,
    "heading_in_relative_direction_to": heading_in_relative_direction_to,
    "being_crossed_by": being_crossed_by,
    "in_same_lane": in_same_lane,
    "union": union,
    "within_time_window": within_time_window,
}


@dataclass
class ExecutionResult:
    program: ScenarioProgram
    code: str
    runtime_sec: float
    scenario_dict: Optional[dict]
    prediction_path: Optional[Path]
    error: Optional[str]


class ScenarioExecutor:
    def __init__(self, scratch_output_root: Path):
        self.scratch_output_root = Path(scratch_output_root)
        self.scratch_output_root.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        program: ScenarioProgram,
        log_dir: Path,
        output_root: Optional[Path] = None,
        prediction_name: Optional[str] = None,
        persist_prediction: bool = True,
    ) -> ExecutionResult:
        log_dir = Path(log_dir)
        output_root = Path(output_root) if output_root is not None else self.scratch_output_root
        scenario_state: dict[str, dict] = {}
        prediction_path = None
        code = ""

        start_time = time.time()
        try:
            validate_program(program)
            code = compile_program_to_code(program)
            cache_manager.load_custom_caches(log_dir)
            for step in program.steps:
                fn = OPERATION_FNS[step.op]
                if step.op == "get_objects_of_category":
                    scenario_state[step.name] = fn(log_dir, **step.params)
                elif step.related_source:
                    scenario_state[step.name] = fn(
                        scenario_state[step.source],
                        scenario_state[step.related_source],
                        log_dir,
                        **step.params,
                    )
                else:
                    scenario_state[step.name] = fn(
                        scenario_state[step.source],
                        log_dir,
                        **step.params,
                    )

            final_dict = scenario_state[program.result]
            if persist_prediction:
                description = prediction_name or program.description
                output_scenario(final_dict, description, log_dir, output_root)
                prediction_path = output_root / log_dir.name / f"{description}_predictions.pkl"
            return ExecutionResult(
                program=program,
                code=code,
                runtime_sec=time.time() - start_time,
                scenario_dict=final_dict,
                prediction_path=prediction_path,
                error=None,
            )
        except Exception:
            return ExecutionResult(
                program=program,
                code=code,
                runtime_sec=time.time() - start_time,
                scenario_dict=None,
                prediction_path=None,
                error=traceback.format_exc(),
            )
