#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import refAV.paths as paths
from refAV.agentic_st_batch import AgenticBatchRunner
from refAV.agentic_st_config import AgenticSearchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agentic spatio-temporal RefAV search.")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--log-prompt-pairs", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True, help="Root directory containing per-log RefAV-ready annotations.")
    parser.add_argument("--planner-model-name", type=str, default=None)
    parser.add_argument("--local-model-path", type=str, default=None)
    parser.add_argument("--gt-annotations", type=Path, default=None)
    parser.add_argument("--gt-combined-pkl", type=Path, default=None)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-candidates-per-round", type=int, default=10)
    parser.add_argument("--num-return-sequences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--model-max-memory-gpu", type=str, default="44GiB")
    parser.add_argument("--model-max-memory-cpu", type=str, default="32GiB")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--disable-hf-planner", action="store_true")
    parser.add_argument("--disable-heuristic-planner", action="store_true")
    parser.add_argument("--disable-visual-coarse-filter", action="store_true")
    parser.add_argument("--disable-trajectory-reranker", action="store_true")
    parser.add_argument("--build-visual-caches", action="store_true")
    parser.add_argument("--memory-path", type=Path, default=None)
    parser.add_argument("--knowledge-base-path", type=Path, default=None)
    parser.add_argument("--official-output-root", type=Path, default=None)
    parser.add_argument("--scratch-output-root", type=Path, default=None)
    parser.add_argument("--max-knowledge-examples", type=int, default=4)
    parser.add_argument("--visual-scene-camera", type=str, default="ring_front_center")
    parser.add_argument("--visual-planner-num-images", type=int, default=2)
    parser.add_argument("--reranker-weight", type=float, default=0.15)
    parser.add_argument("--keep-candidate-predictions", action="store_true")
    # ---- SMc2f CLIP coarse filter (Stage 1) ----
    parser.add_argument("--enable-smc2f-clip-filter", action="store_true",
                        help="Apply SMc2f CLIP coarse segment filter (requires precomputed features).")
    parser.add_argument("--smc2f-clip-cache-dir", type=Path, default=None,
                        help="Directory containing precomputed per-camera CLIP features.")
    # ---- DETTM fine filter (Stage 2) ----
    parser.add_argument("--enable-dettm-filter", action="store_true",
                        help="Apply trained DETTM to prune false-positive tracks.")
    parser.add_argument("--dettm-checkpoint", type=Path, default=None,
                        help="Path to dettm_checkpoint.pt produced by train_dettm.py.")
    parser.add_argument("--dettm-threshold", type=float, default=0.0,
                        help="Min DETTM score to keep a track (default 0.0 = keep positives).")
    # ---- Round budget controls (main runtime knob) ----
    parser.add_argument("--adaptive-max-rounds", type=int, default=None,
                        help="Cap for adaptive round extension on hard prompts (default 5).")
    parser.add_argument("--disable-adaptive-rounds", action="store_true",
                        help="Hard-cap search at --max-rounds (no hard-prompt extension).")
    return parser.parse_args()


def _default_base_output_root() -> Path:
    env_root = os.environ.get("REFAV_OUTPUT_ROOT")
    if env_root:
        return Path(env_root)

    scratch_root = Path.home() / "scratch"
    if scratch_root.exists():
        return scratch_root / "refav_output"

    return REPO_ROOT / "output"


def main() -> None:
    args = parse_args()
    base_output_root = _default_base_output_root()
    official_output_root = args.official_output_root or base_output_root / "sm_predictions" / "agentic_spatiotemporal"
    scratch_output_root = args.scratch_output_root or base_output_root / "sm_predictions" / "agentic_spatiotemporal_scratch"
    memory_path = args.memory_path or official_output_root / args.experiment_name / "memory.jsonl"

    config = AgenticSearchConfig(
        planner_model_name=args.planner_model_name,
        local_model_path=args.local_model_path,
        beam_width=args.beam_width,
        max_rounds=args.max_rounds,
        max_candidates_per_round=args.max_candidates_per_round,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_knowledge_examples=args.max_knowledge_examples,
        use_4bit=not args.disable_4bit,
        model_max_memory_gpu=args.model_max_memory_gpu,
        model_max_memory_cpu=args.model_max_memory_cpu,
        enable_hf_planner=not args.disable_hf_planner,
        enable_heuristic_planner=not args.disable_heuristic_planner,
        enable_visual_coarse_filter=not args.disable_visual_coarse_filter,
        enable_trajectory_reranker=not args.disable_trajectory_reranker,
        build_visual_caches=args.build_visual_caches,
        official_output_root=official_output_root,
        scratch_output_root=scratch_output_root,
        memory_path=memory_path,
        knowledge_base_path=args.knowledge_base_path or AgenticSearchConfig.knowledge_base_path,
        visual_scene_camera=args.visual_scene_camera,
        visual_planner_num_images=args.visual_planner_num_images,
        reranker_weight=args.reranker_weight,
        keep_candidate_predictions=args.keep_candidate_predictions,
        enable_smc2f_clip_filter=args.enable_smc2f_clip_filter,
        smc2f_clip_cache_dir=args.smc2f_clip_cache_dir,
        enable_dettm_filter=args.enable_dettm_filter,
        dettm_checkpoint_path=args.dettm_checkpoint,
        dettm_threshold=args.dettm_threshold,
        adaptive_rounds=not args.disable_adaptive_rounds,
        **({"adaptive_max_rounds": args.adaptive_max_rounds}
           if args.adaptive_max_rounds is not None else {}),
    )

    runner = AgenticBatchRunner(config)
    metrics = runner.run(
        split=args.split,
        log_prompt_pairs_path=args.log_prompt_pairs,
        log_root=args.log_root,
        experiment_name=args.experiment_name,
        gt_annotations_path=args.gt_annotations,
        gt_combined_output_path=args.gt_combined_pkl,
        max_items=args.max_items,
    )
    if metrics:
        print(metrics)


if __name__ == "__main__":
    main()
