from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import refAV.paths as paths


@dataclass
class AgenticSearchConfig:
    planner_model_name: Optional[str] = None
    local_model_path: Optional[str] = None
    knowledge_base_path: Optional[Path] = paths.PROMPT_DIR / "RefAV" / "agentic_st_knowledge_base.jsonl"
    beam_width: int = 4
    max_rounds: int = 2
    max_candidates_per_round: int = 6
    max_mutations_per_program: int = 6
    max_memory_examples: int = 4
    max_knowledge_examples: int = 4
    max_programs_saved_per_prompt: int = 3
    max_new_tokens: int = 512
    temperature: float = 0.4
    top_p: float = 0.92
    num_return_sequences: int = 2
    use_4bit: bool = True
    model_device_map: str = "auto"
    model_max_memory_gpu: str = "38GiB"
    model_max_memory_cpu: str = "48GiB"
    runtime_timeout_sec: int = 300
    official_output_root: Path = paths.SM_PRED_DIR / "agentic_spatiotemporal"
    scratch_output_root: Path = paths.SM_PRED_DIR / "agentic_spatiotemporal_scratch"
    memory_path: Path = paths.SM_PRED_DIR / "agentic_spatiotemporal" / "memory.jsonl"
    summary_path: Path = paths.SM_PRED_DIR / "agentic_spatiotemporal" / "search_summary.jsonl"
    context_dir: Path = paths.PROMPT_DIR / "RefAV"
    enable_hf_planner: bool = True
    enable_heuristic_planner: bool = True
    enable_visual_coarse_filter: bool = True
    enable_trajectory_reranker: bool = True
    # ---- SMc2f CLIP coarse filter (Stage 1 integration) ----
    # When True, at the start of each (log, prompt) iteration we run the
    # SMc2f CLIP coarse filter using the precomputed per-camera features at
    # `smc2f_clip_cache_dir`. It returns a list of (start_ts, end_ts) segments
    # that we use to gate which frames the GT and predictions are compared on.
    # Off by default so existing runs stay unchanged until features have been
    # precomputed via `python -m refAV.smc2f.precompute_clip`.
    enable_smc2f_clip_filter: bool = False
    smc2f_clip_cache_dir: Optional[Path] = None
    smc2f_split: str = "val"
    # ---- DETTM fine filter (Stage 2 integration) ----
    # When True, the best DSL prediction for each (log, prompt) is passed
    # through the trained DETTM dual-encoder to drop candidate tracks whose
    # cosine similarity to the description falls below `dettm_threshold`.
    # Applied after the CLIP segment filter (if enabled) and before persisting
    # the final prediction. Safe fallback: if DETTM would drop all tracks it
    # keeps the full set instead (never catastrophic recall loss).
    # Requires a checkpoint at `dettm_checkpoint_path` produced by
    # /home/rornelas5/scenario-mining/train_dettm.py.
    enable_dettm_filter: bool = False
    dettm_checkpoint_path: Optional[Path] = None
    dettm_threshold: float = 0.0  # start permissive; tune up after A/B test
    # If True the HF planner is invoked again after round 0 so it can REFLECT
    # on the per-timestamp / per-track diff vs GT and emit a corrected program.
    # This is the iterative reflect-and-correct loop that the SMc2f-style
    # agentic-mining design depends on. We cap the LLM at
    # `llm_refinement_round_budget` extra calls (default 1) so total LLM cost
    # per prompt stays at ~2 generate() calls.
    iterative_llm_refinement: bool = True
    # When the prompt-level cached seed (from prior logs) already scores at
    # least this much on round 0, skip the LLM entirely and short-circuit the
    # search. Tuned conservatively so we don't lose accuracy.
    cached_seed_skip_score: float = 0.55
    # Max number of refinement rounds in which the LLM may be re-invoked
    # (only used when iterative_llm_refinement=True). 0 = round-0 only.
    llm_refinement_round_budget: int = 1
    build_visual_caches: bool = False
    keep_candidate_predictions: bool = False
    visual_scene_camera: str = "ring_front_center"
    visual_coarse_track_budget: int = 12
    visual_coarse_timestamp_budget: int = 12
    visual_planner_num_images: int = 2
    reranker_weight: float = 0.15
    reward_runtime_penalty_weight: float = 0.05
    reward_empty_prediction_bias: float = 0.02
    early_exit_score: float = 0.70
    # Embedding-based retrieval
    embedding_model_name: str = "all-MiniLM-L6-v2"
    # Adaptive round budget
    adaptive_rounds: bool = True
    adaptive_hard_threshold: float = 0.25   # below this after round 0 → extend to adaptive_max_rounds
    adaptive_easy_threshold: float = 0.75   # above this after round 1 → allow 1 more round then stop
    adaptive_plateau_threshold: float = 0.01  # min per-round improvement to not count as plateau
    adaptive_plateau_patience: int = 2        # plateau rounds before early exit
    adaptive_max_rounds: int = 5              # extended round cap for hard prompts
    adaptive_min_exit_score: float = 0.30    # min score required to allow plateau early exit
