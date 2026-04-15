from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from refAV.agentic_st_config import AgenticSearchConfig
from refAV.agentic_st_dsl import ProgramCandidate, ScenarioProgram, StepSpec
from refAV.agentic_st_executor import ScenarioExecutor
from refAV.agentic_st_memory import MemoryStore
from refAV.agentic_st_planner import CombinedPlanner
from refAV.agentic_st_reranker import TrajectoryReranker
from refAV.agentic_st_reward import RewardBreakdown, score_scenario_dict
from refAV.agentic_st_visual import VisualCoarseFilter


@dataclass
class SearchResult:
    best_program: ScenarioProgram
    best_reward: RewardBreakdown
    best_score: float
    best_feedback: str
    search_trace: list[dict]
    visual_context_summary: str = ""


class ProgramMutator:
    def __init__(self, config: AgenticSearchConfig):
        self.config = config

    @staticmethod
    def _rename_symbol(program: ScenarioProgram, old_name: str, new_name: str) -> None:
        for step in program.steps:
            if step.name == old_name:
                step.name = new_name
            if step.source == old_name:
                step.source = new_name
            if step.related_source == old_name:
                step.related_source = new_name
        if program.result == old_name:
            program.result = new_name

    def mutate(self, candidate: ProgramCandidate) -> list[ProgramCandidate]:
        program = candidate.program
        mutations: list[ProgramCandidate] = []
        for step in program.steps:
            if step.op == "has_objects_in_relative_direction":
                for distance in (12.0, 20.0, 35.0, 50.0):
                    clone = program.clone()
                    for clone_step in clone.steps:
                        if clone_step.name == step.name:
                            old_name = clone_step.name
                            new_name = f"{clone_step.name}_wd{int(distance)}"
                            clone_step.params["within_distance"] = distance
                            self._rename_symbol(clone, old_name, new_name)
                            break
                    mutations.append(
                        ProgramCandidate(program=clone, provenance="mutation:distance", search_round=candidate.search_round + 1)
                    )
            elif step.op == "near_objects":
                for distance in (8.0, 12.0, 20.0):
                    clone = program.clone()
                    for clone_step in clone.steps:
                        if clone_step.name == step.name:
                            old_name = clone_step.name
                            new_name = f"{clone_step.name}_d{int(distance)}"
                            clone_step.params["distance_thresh"] = distance
                            self._rename_symbol(clone, old_name, new_name)
                            break
                    mutations.append(
                        ProgramCandidate(program=clone, provenance="mutation:near_distance", search_round=candidate.search_round + 1)
                    )
            elif step.op == "accelerating":
                for max_accel in (-1.0, -0.5, 1.5, 2.5):
                    clone = program.clone()
                    for clone_step in clone.steps:
                        if clone_step.name == step.name:
                            old_name = clone_step.name
                            new_name = f"{clone_step.name}_a{str(max_accel).replace('.', '_').replace('-', 'm')}"
                            clone_step.params["min_accel"] = float("-inf") if max_accel < 0 else 0.5
                            clone_step.params["max_accel"] = max_accel
                            self._rename_symbol(clone, old_name, new_name)
                            break
                    mutations.append(
                        ProgramCandidate(program=clone, provenance="mutation:accel", search_round=candidate.search_round + 1)
                    )
            elif step.op in {"turning", "changing_lanes"}:
                for direction in ("left", "right", None):
                    clone = program.clone()
                    for clone_step in clone.steps:
                        if clone_step.name == step.name:
                            old_name = clone_step.name
                            clone_step.params["direction"] = direction
                            suffix = "any" if direction is None else direction
                            new_name = f"{clone_step.name}_{suffix}"
                            self._rename_symbol(clone, old_name, new_name)
                            break
                    mutations.append(
                        ProgramCandidate(program=clone, provenance="mutation:direction", search_round=candidate.search_round + 1)
                    )
            elif step.op == "has_velocity":
                for min_vel in (0.3, 0.5, 1.0, 2.0):
                    clone = program.clone()
                    for clone_step in clone.steps:
                        if clone_step.name == step.name:
                            old_name = clone_step.name
                            new_name = f"{clone_step.name}_v{str(min_vel).replace('.', '_')}"
                            clone_step.params["min_velocity"] = min_vel
                            self._rename_symbol(clone, old_name, new_name)
                            break
                    mutations.append(
                        ProgramCandidate(program=clone, provenance="mutation:velocity", search_round=candidate.search_round + 1)
                    )
            elif step.op == "near_intersection":
                for threshold in (3.0, 5.0, 10.0, 20.0):
                    clone = program.clone()
                    for clone_step in clone.steps:
                        if clone_step.name == step.name:
                            old_name = clone_step.name
                            new_name = f"{clone_step.name}_t{int(threshold)}"
                            clone_step.params["threshold"] = threshold
                            self._rename_symbol(clone, old_name, new_name)
                            break
                    mutations.append(
                        ProgramCandidate(program=clone, provenance="mutation:intersection_dist", search_round=candidate.search_round + 1)
                    )
            elif step.op == "following":
                for distance in (10.0, 20.0, 35.0):
                    clone = program.clone()
                    for clone_step in clone.steps:
                        if clone_step.name == step.name:
                            old_name = clone_step.name
                            new_name = f"{clone_step.name}_fd{int(distance)}"
                            clone_step.params["distance_thresh"] = distance
                            self._rename_symbol(clone, old_name, new_name)
                            break
                    mutations.append(
                        ProgramCandidate(program=clone, provenance="mutation:follow_dist", search_round=candidate.search_round + 1)
                    )

        unique = {}
        for mutation in mutations:
            unique[mutation.program.fingerprint()] = mutation
        return list(unique.values())[: self.config.max_mutations_per_program]


class SpatioTemporalSearch:
    def __init__(
        self,
        config: AgenticSearchConfig,
        planner: CombinedPlanner,
        executor: ScenarioExecutor,
        memory: MemoryStore,
    ):
        self.config = config
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.mutator = ProgramMutator(config)
        self.visual_coarse_filter = (
            VisualCoarseFilter(config)
            if (config.enable_visual_coarse_filter or config.enable_trajectory_reranker)
            else None
        )
        self.reranker = TrajectoryReranker(config) if config.enable_trajectory_reranker else None

        # ---- SMc2f CLIP coarse filter (Stage 1) ----
        # Lazily constructed on first use so that the CLIP model is only
        # loaded when the switch is on. Cache segments per (log_id, prompt)
        # so we don't re-encode/score the same prompt twice.
        self._smc2f_clip_filter = None
        self._smc2f_segments_cache: dict[tuple[str, str], list[tuple[int, int]]] = {}
        # ---- DETTM fine filter (Stage 2) ----
        self._dettm_matcher = None        # None = not yet attempted
        self._dettm_load_attempted = False

    def _get_smc2f_clip_filter(self):
        if not self.config.enable_smc2f_clip_filter:
            return None
        if self._smc2f_clip_filter is None:
            try:
                from refAV.smc2f import CLIPCoarseFilter, SMc2fConfig
                if self.config.smc2f_clip_cache_dir is not None:
                    # Override the vendored config's cache path at runtime so
                    # users can point to any precomputed feature directory.
                    SMc2fConfig.CLIP_CACHE_DIR = Path(self.config.smc2f_clip_cache_dir)
                self._smc2f_clip_filter = CLIPCoarseFilter(device="cuda")
            except Exception as exc:
                print(f"[smc2f] CLIP filter init failed: {exc}; disabling for this run")
                self.config.enable_smc2f_clip_filter = False
                return None
        return self._smc2f_clip_filter

    def _get_smc2f_segments(self, prompt: str, log_dir: Path) -> list[tuple[int, int]]:
        """Return precomputed CLIP-based allowed temporal segments for this
        (log, prompt). Empty list means "no filtering" (either disabled or
        precomputed features are missing for this log)."""
        clip_filter = self._get_smc2f_clip_filter()
        if clip_filter is None:
            return []
        log_id = Path(log_dir).name
        key = (log_id, prompt)
        cached = self._smc2f_segments_cache.get(key)
        if cached is not None:
            return cached
        try:
            segments = clip_filter.get_filtered_segments(prompt, log_id, self.config.smc2f_split)
        except FileNotFoundError:
            # Precomputed features missing for this log — silently skip.
            segments = []
        except Exception as exc:
            print(f"[smc2f] segment extraction failed for {log_id}/{prompt!r}: {exc}")
            segments = []
        self._smc2f_segments_cache[key] = segments
        return segments

    def _apply_smc2f_filter(self, scenario_dict, segments):
        """Apply CLIP segments to a scenario_dict. Returns the filtered dict
        (or the original if segments is empty)."""
        if not segments or not scenario_dict:
            return scenario_dict
        from refAV.smc2f import filter_scenario_by_segments
        return filter_scenario_by_segments(scenario_dict, segments)

    def _get_dettm_matcher(self):
        """Lazily load the DETTM matcher; returns None if disabled or load fails."""
        if not self.config.enable_dettm_filter:
            return None
        if self._dettm_load_attempted:
            return self._dettm_matcher
        self._dettm_load_attempted = True
        ckpt = self.config.dettm_checkpoint_path
        if ckpt is None or not Path(ckpt).exists():
            print(f"[dettm] checkpoint not found ({ckpt}); disabling filter")
            return None
        try:
            from refAV.smc2f import DETTMMatcher
            self._dettm_matcher = DETTMMatcher(Path(ckpt), device="cuda")
        except Exception as exc:
            print(f"[dettm] load failed: {exc}; disabling filter")
        return self._dettm_matcher

    def _apply_dettm_filter(self, scenario_dict, prompt, log_dir, matcher):
        """Run DETTM fine filter; falls back to original dict on error or empty result."""
        if not scenario_dict:
            return scenario_dict
        try:
            filtered = matcher.filter_tracks(
                scenario_dict, prompt, Path(log_dir),
                threshold=self.config.dettm_threshold,
            )
            print(f"[dettm] {len(scenario_dict)} → {len(filtered)} tracks "
                  f"(threshold={self.config.dettm_threshold})")
            return filtered  # DETTMMatcher.filter_tracks already falls back if empty
        except Exception as exc:
            print(f"[dettm] filter failed: {exc}; keeping original")
            return scenario_dict

    def search(
        self,
        prompt: str,
        log_dir: Path,
        gt_prompt_df: Optional[pd.DataFrame] = None,
        prediction_name: Optional[str] = None,
        official_output_root: Optional[Path] = None,
        seed_programs: Optional[list] = None,
    ) -> SearchResult:
        retrieved_memory = self.memory.retrieve(prompt, top_k=self.config.max_memory_examples)
        visual_context = self.visual_coarse_filter.build_context(prompt, log_dir) if self.visual_coarse_filter is not None else None

        # SMc2f CLIP coarse filter — compute allowed temporal segments once
        # per (log, prompt). Empty list when disabled or missing features.
        smc2f_segments = self._get_smc2f_segments(prompt, log_dir)

        # --- Fast path: try cached prompt-level seed first WITHOUT calling the LLM. ---
        # If a previously discovered program for this prompt already scores well on
        # this log, we skip the (very expensive) HF planner generate() entirely.
        candidates: list[ProgramCandidate] = []
        skip_initial_llm = False
        if seed_programs:
            from refAV.agentic_st_dsl import ProgramCandidate as _PC
            seeded = [_PC(program=p, provenance="prompt_cache") for p in seed_programs]
            for seed in seeded:
                exec_result = self.executor.execute(
                    seed.program,
                    log_dir=log_dir,
                    output_root=self.config.scratch_output_root,
                    prediction_name=f"{prediction_name or prompt}__seedprobe",
                    persist_prediction=False,
                )
                if not exec_result.error:
                    filtered_seed_dict = self._apply_smc2f_filter(
                        exec_result.scenario_dict or {}, smc2f_segments
                    )
                    seed_reward = score_scenario_dict(
                        filtered_seed_dict,
                        gt_prompt_df=gt_prompt_df,
                        runtime_sec=exec_result.runtime_sec,
                        config=self.config,
                    )
                    if seed_reward.total >= self.config.cached_seed_skip_score:
                        skip_initial_llm = True
            candidates = seeded + candidates

        if not skip_initial_llm:
            llm_candidates = self.planner.propose(prompt, retrieved_memory=retrieved_memory, visual_context=visual_context, log_dir=log_dir)
            candidates = candidates + llm_candidates
        elif self.planner.heuristic is not None:
            # Still mix in cheap heuristic candidates so mutation has a base set.
            candidates = candidates + self.planner.heuristic.propose(
                prompt, max_candidates=self.config.max_candidates_per_round
            )

        if not candidates and self.planner.heuristic is not None:
            candidates = self.planner.heuristic.propose(prompt, max_candidates=self.config.max_candidates_per_round)

        seen = set()
        best_candidate: Optional[ProgramCandidate] = None
        best_reward: Optional[RewardBreakdown] = None
        best_score = float("-inf")
        best_feedback = "No candidate succeeded."
        search_trace: list[dict] = []
        feedback = None

        # --- Adaptive round budget ---
        # Start with the configured max_rounds; expand for hard prompts, compress for easy ones.
        cfg = self.config
        adaptive = cfg.adaptive_rounds
        round_cap = cfg.max_rounds           # may grow up to adaptive_max_rounds
        prev_round_best = float("-inf")
        plateau_count = 0
        max_possible_rounds = cfg.adaptive_max_rounds if adaptive else cfg.max_rounds

        for round_index in range(max_possible_rounds):
            if round_index >= round_cap:
                break

            evaluated: list[tuple[ProgramCandidate, RewardBreakdown, float]] = []
            for candidate in candidates:
                fingerprint = candidate.program.fingerprint()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                execution = self.executor.execute(
                    candidate.program,
                    log_dir=log_dir,
                    output_root=self.config.scratch_output_root,
                    prediction_name=f"{prediction_name or prompt}__{fingerprint[:8]}",
                    persist_prediction=self.config.keep_candidate_predictions,
                )

                if execution.error:
                    reward = RewardBreakdown(
                        total=0.0,
                        referred_track_f1=0.0,
                        related_track_f1=0.0,
                        referred_timestamp_f1=0.0,
                        related_timestamp_f1=0.0,
                        runtime_penalty=0.0,
                        feedback=f"Execution error: {execution.error.splitlines()[-1]}",
                    )
                else:
                    # Apply SMc2f CLIP segment mask before scoring so that
                    # spurious out-of-window timestamps don't contribute to
                    # the reward. Tracks that become empty are pruned.
                    execution.scenario_dict = self._apply_smc2f_filter(
                        execution.scenario_dict or {}, smc2f_segments
                    )
                    reward = score_scenario_dict(
                        execution.scenario_dict or {},
                        gt_prompt_df=gt_prompt_df,
                        runtime_sec=execution.runtime_sec,
                        config=self.config,
                    )

                rerank_bonus = 0.0
                rerank_feedback = ""
                if not execution.error and self.reranker is not None:
                    rerank = self.reranker.score(
                        prompt=prompt,
                        scenario_dict=execution.scenario_dict or {},
                        log_dir=log_dir,
                        visual_context=visual_context,
                    )
                    rerank_bonus = rerank.total
                    rerank_feedback = rerank.feedback

                combined_feedback = reward.feedback
                if rerank_feedback:
                    combined_feedback = f"{combined_feedback}; reranker={rerank_feedback}"

                candidate.candidate_score = reward.total + rerank_bonus
                candidate.candidate_feedback = combined_feedback
                evaluated.append((candidate, reward, rerank_bonus))
                search_trace.append(
                    {
                        "round": round_index,
                        "fingerprint": fingerprint,
                        "provenance": candidate.provenance,
                        "score": candidate.candidate_score,
                        "base_score": reward.total,
                        "rerank_score": rerank_bonus,
                        "feedback": combined_feedback,
                        "program": candidate.program.to_dict(),
                        "error": execution.error,
                    }
                )

            if not evaluated:
                break

            evaluated.sort(key=lambda item: item[0].candidate_score or float("-inf"), reverse=True)
            top_candidate, top_reward, _ = evaluated[0]
            if best_reward is None or (top_candidate.candidate_score or float("-inf")) > best_score:
                best_candidate = top_candidate
                best_reward = top_reward
                best_score = top_candidate.candidate_score or top_reward.total
                best_feedback = top_candidate.candidate_feedback or top_reward.feedback
                self.memory.append(
                    prompt=prompt,
                    program=top_candidate.program,
                    score=best_score,
                    feedback=best_feedback,
                    log_id=Path(log_dir).name,
                    planner_model_name=self.config.planner_model_name,
                )

            # --- Hard exit: score is already near-perfect ---
            if best_score >= cfg.early_exit_score:
                break

            # --- Adaptive round budget adjustment (after round 0) ---
            if adaptive and round_index >= 0:
                improvement = best_score - prev_round_best
                if improvement < cfg.adaptive_plateau_threshold:
                    plateau_count += 1
                else:
                    plateau_count = 0
                prev_round_best = best_score

                if round_index == 0:
                    # After first round: extend budget for hard prompts, compress for easy ones
                    if best_score < cfg.adaptive_hard_threshold:
                        round_cap = cfg.adaptive_max_rounds
                        print(f"[search] hard prompt (score={best_score:.3f}), extending to {round_cap} rounds")
                    elif best_score >= cfg.adaptive_easy_threshold:
                        round_cap = min(round_cap, 2)
                        print(f"[search] easy prompt (score={best_score:.3f}), capping at {round_cap} rounds")
                else:
                    # Subsequent rounds: allow early exit on plateau if score is acceptable
                    if (plateau_count >= cfg.adaptive_plateau_patience
                            and best_score >= cfg.adaptive_min_exit_score):
                        print(f"[search] plateaued at {best_score:.3f} after {plateau_count} flat rounds, stopping")
                        break

            beam = [candidate for candidate, _, _ in evaluated[: self.config.beam_width]]
            feedback = top_candidate.candidate_feedback or top_reward.feedback

            # --- LLM call budget for refinement rounds ---
            # By default we DO NOT call the HF planner again after round 0.
            # Mutations of the current beam (cheap, CPU-only) provide enough
            # exploration for parameter tuning. Only when iterative refinement
            # is explicitly enabled do we re-prompt the LLM, and even then we
            # cap how many refinement rounds may invoke it.
            llm_refinement_allowed = (
                cfg.iterative_llm_refinement
                and round_index < cfg.llm_refinement_round_budget
            )
            if llm_refinement_allowed:
                try:
                    next_candidates = self.planner.propose(
                        prompt,
                        retrieved_memory=retrieved_memory,
                        feedback=feedback,
                        visual_context=visual_context,
                        log_dir=log_dir,
                    )
                except Exception as exc:
                    print(f"[search] LLM refinement failed ({exc}); falling back to mutations only")
                    next_candidates = []
                finally:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            else:
                next_candidates = []
            for beam_candidate in beam:
                next_candidates.extend(self.mutator.mutate(beam_candidate))

            deduped = {}
            for next_candidate in next_candidates:
                deduped[next_candidate.program.fingerprint()] = next_candidate
            candidates = list(deduped.values())[: self.config.max_candidates_per_round]

        if best_candidate is None or best_reward is None:
            from refAV.agentic_st_planner import _category_for_prompt
            fallback_category = _category_for_prompt(prompt)
            fallback_program = candidates[0].program if candidates else ScenarioProgram(
                description=prompt,
                result="objects",
                steps=[StepSpec(name="objects", op="get_objects_of_category", params={"category": fallback_category})],
                notes="fallback",
            )
            best_reward = RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No candidate succeeded.")
            best_candidate = ProgramCandidate(program=fallback_program, provenance="fallback")
            best_score = 0.0
            best_feedback = "No candidate succeeded."

        if official_output_root is not None:
            _dettm = self._get_dettm_matcher()
            _need_postprocess = bool(smc2f_segments) or (_dettm is not None)
            if _need_postprocess:
                # Re-execute without persisting so we can apply post-filters
                # (CLIP segment mask and/or DETTM fine filter) before writing.
                final_exec = self.executor.execute(
                    best_candidate.program,
                    log_dir=log_dir,
                    output_root=official_output_root,
                    prediction_name=prediction_name or prompt,
                    persist_prediction=False,
                )
                final_dict = final_exec.scenario_dict or {}
                if smc2f_segments:
                    final_dict = self._apply_smc2f_filter(final_dict, smc2f_segments)
                if _dettm and final_dict:
                    final_dict = self._apply_dettm_filter(
                        final_dict, prompt, log_dir, _dettm
                    )
                try:
                    from refAV.atomic_functions import output_scenario
                    output_scenario(
                        final_dict,
                        prediction_name or prompt,
                        Path(log_dir),
                        Path(official_output_root),
                    )
                except Exception as exc:
                    print(f"[smc2f/dettm] filtered persist failed ({exc}); "
                          "falling back to unfiltered executor persist")
                    self.executor.execute(
                        best_candidate.program,
                        log_dir=log_dir,
                        output_root=official_output_root,
                        prediction_name=prediction_name or prompt,
                        persist_prediction=True,
                    )
            else:
                self.executor.execute(
                    best_candidate.program,
                    log_dir=log_dir,
                    output_root=official_output_root,
                    prediction_name=prediction_name or prompt,
                    persist_prediction=True,
                )

        return SearchResult(
            best_program=best_candidate.program,
            best_reward=best_reward,
            best_score=best_score,
            best_feedback=best_feedback,
            search_trace=search_trace,
            visual_context_summary=visual_context.summary if visual_context is not None else "",
        )
