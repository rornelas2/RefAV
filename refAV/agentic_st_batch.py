from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

import refAV.paths as paths
from refAV.agentic_st_compiler import compile_program_to_code
from refAV.agentic_st_config import AgenticSearchConfig
from refAV.agentic_st_dsl import ScenarioProgram
from refAV.agentic_st_executor import ScenarioExecutor
from refAV.agentic_st_memory import MemoryStore
from refAV.agentic_st_planner import CombinedPlanner
from refAV.agentic_st_search import SpatioTemporalSearch
from refAV.dataset_conversion import create_gt_mining_pkls_parallel, create_gt_pkl_file, separate_scenario_mining_annotations
from refAV.eval import combine_pkls, evaluate_pkls
from refAV.utils import construct_caches


class AgenticBatchRunner:
    def __init__(self, config: AgenticSearchConfig):
        self.config = config
        self.planner = CombinedPlanner(config)
        self.executor = ScenarioExecutor(config.scratch_output_root)
        self.memory = MemoryStore(config.memory_path, embedding_model_name=config.embedding_model_name)
        self.search = SpatioTemporalSearch(config, self.planner, self.executor, self.memory)
        # prompt → (best_program, best_score) accumulated across all logs
        self._prompt_program_cache: dict[str, tuple[ScenarioProgram, float]] = {}

    @staticmethod
    def _resolve_log_dir(log_id: str, log_root: Path) -> Path:
        log_dir = Path(log_root) / log_id
        if not log_dir.exists():
            raise FileNotFoundError(f"Missing log directory for {log_id}: {log_dir}")
        return log_dir

    @staticmethod
    def _resolve_gt_source_log_root(split: str, log_root: Optional[Path]) -> Optional[Path]:
        if log_root is None:
            candidate = paths.AV2_DATA_DIR / split
            return candidate if candidate.exists() else None

        log_root = Path(log_root)

        # If the runner is using a converted RefAV tree like .../refav_sensor/val,
        # prefer the sibling raw AV2 tree for ego poses and map assets.
        if log_root.name == split and log_root.parent.name == "refav_sensor":
            raw_candidate = log_root.parent.parent / "sensor" / split
            if raw_candidate.exists():
                return raw_candidate

        if log_root.exists():
            return log_root

        candidate = paths.AV2_DATA_DIR / split
        return candidate if candidate.exists() else None

    @staticmethod
    def _filter_gt_annotations_for_pairs(
        gt_annotations_path: Path,
        log_prompt_pairs_path: Path,
        output_path: Path,
    ) -> Path:
        with open(log_prompt_pairs_path, "r") as f:
            selected_pairs: dict[str, list[str]] = json.load(f)

        selected_rows = [
            {"log_id": log_id, "prompt": prompt}
            for log_id, prompts in selected_pairs.items()
            for prompt in prompts
        ]

        if not selected_rows:
            raise ValueError("No selected log-prompt pairs were provided for GT filtering.")

        gt_df = pd.read_feather(gt_annotations_path)
        selected_df = pd.DataFrame(selected_rows).drop_duplicates()
        filtered_df = gt_df.merge(selected_df, on=["log_id", "prompt"], how="inner")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_feather(output_path)
        return output_path

    @staticmethod
    def _prepare_gt_if_needed(
        split: str,
        log_prompt_pairs_path: Path,
        gt_annotations_path: Optional[Path],
        gt_combined_output_path: Optional[Path],
        log_root: Optional[Path] = None,
    ) -> Optional[Path]:
        if gt_combined_output_path is None:
            return None
        if gt_combined_output_path.exists():
            return gt_combined_output_path
        if gt_annotations_path is None:
            return None

        filtered_gt_annotations_path = gt_combined_output_path.parent / f"{split}_selected_annotations.feather"
        filtered_gt_annotations_path = AgenticBatchRunner._filter_gt_annotations_for_pairs(
            gt_annotations_path,
            log_prompt_pairs_path,
            filtered_gt_annotations_path,
        )

        split_output_root = gt_combined_output_path.parent / f"{split}_scenario_pkls"
        split_output_root.mkdir(parents=True, exist_ok=True)
        gt_source_log_root = AgenticBatchRunner._resolve_gt_source_log_root(split, log_root)
        separate_scenario_mining_annotations(filtered_gt_annotations_path, split_output_root)
        create_gt_mining_pkls_parallel(
            filtered_gt_annotations_path,
            split_output_root,
            source_log_root=gt_source_log_root,
        )
        return create_gt_pkl_file(split_output_root, log_prompt_pairs_path, gt_combined_output_path)

    def run(
        self,
        split: str,
        log_prompt_pairs_path: Path,
        log_root: Path,
        experiment_name: str,
        gt_annotations_path: Optional[Path] = None,
        gt_combined_output_path: Optional[Path] = None,
        max_items: Optional[int] = None,
    ) -> dict:
        log_prompt_pairs_path = Path(log_prompt_pairs_path)
        with open(log_prompt_pairs_path, "r") as f:
            log_prompt_pairs: dict[str, list[str]] = json.load(f)

        official_output_root = self.config.official_output_root / experiment_name / "scenario_predictions"
        official_output_root.mkdir(parents=True, exist_ok=True)
        program_output_root = self.config.official_output_root / experiment_name / "programs"
        program_output_root.mkdir(parents=True, exist_ok=True)
        summary_output_path = self.config.official_output_root / experiment_name / "search_summary.jsonl"
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        # Reset the per-experiment summary so reruns don't append to stale lines
        # from earlier probes (the file is opened with "a" below).
        if summary_output_path.exists():
            summary_output_path.unlink()

        items = []
        for log_id, prompts in log_prompt_pairs.items():
            for prompt in prompts:
                items.append((log_id, prompt))
        if max_items is not None:
            items = items[:max_items]

        selected_log_prompt_pairs: dict[str, list[str]] = {}
        for log_id, prompt in items:
            selected_log_prompt_pairs.setdefault(log_id, []).append(prompt)

        effective_lpp_path = log_prompt_pairs_path
        if max_items is not None:
            effective_lpp_path = self.config.official_output_root / experiment_name / "selected_log_prompt_pairs.json"
            effective_lpp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(effective_lpp_path, "w") as f:
                json.dump(selected_log_prompt_pairs, f, indent=2, sort_keys=True)

        if self.config.build_visual_caches and selected_log_prompt_pairs:
            try:
                selected_log_dirs = [self._resolve_log_dir(log_id, log_root) for log_id in selected_log_prompt_pairs.keys()]
                construct_caches(selected_log_dirs)
            except Exception as exc:
                print(f"[warn] Visual cache construction failed: {exc}")

        gt_df = None
        if gt_annotations_path is not None:
            gt_df = pd.read_feather(gt_annotations_path)

        for log_id, prompt in tqdm(items, desc="Agentic spatio-temporal search"):
            log_dir = self._resolve_log_dir(log_id, log_root)
            gt_prompt_df = None
            if gt_df is not None:
                gt_prompt_df = gt_df[(gt_df["log_id"] == log_id) & (gt_df["prompt"] == prompt)].copy()

            # Seed with the best program found for this prompt across prior logs
            cached = self._prompt_program_cache.get(prompt)
            seed_programs = [cached[0]] if cached is not None else None

            result = self.search.search(
                prompt=prompt,
                log_dir=log_dir,
                gt_prompt_df=gt_prompt_df,
                prediction_name=prompt,
                official_output_root=official_output_root,
                seed_programs=seed_programs,
            )

            # Update the prompt-level cache if this result is the best seen so far
            prev_score = self._prompt_program_cache.get(prompt, (None, float("-inf")))[1]
            if result.best_score > prev_score:
                self._prompt_program_cache[prompt] = (result.best_program, result.best_score)

            code = compile_program_to_code(result.best_program)
            log_program_dir = program_output_root / log_id
            try:
                log_program_dir.mkdir(parents=True, exist_ok=True)
                program_json_path = log_program_dir / f"{prompt}.json"
                program_code_path = log_program_dir / f"{prompt}.py"
                with open(program_json_path, "w") as f:
                    json.dump(result.best_program.to_dict(), f, indent=2, sort_keys=True)
                with open(program_code_path, "w") as f:
                    f.write(code + "\n")
                with open(summary_output_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "log_id": log_id,
                                "prompt": prompt,
                                "score": result.best_score,
                                "feedback": result.best_feedback,
                                "program_path": str(program_json_path),
                                "trace_len": len(result.search_trace),
                                "visual_context_summary": result.visual_context_summary,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError as exc:
                print(f"[warn] Failed to persist program artifacts for {log_id} / {prompt}: {exc}")

            # Periodic GPU memory cleanup to prevent OOM on long runs
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            # Drop cached EasyDataLoader entries for *other* logs so memory does
            # not grow without bound across the val set. We keep the entry for
            # the current log because the next iteration is usually the same log.
            try:
                from refAV.utils import _LOADER_CACHE
                if len(_LOADER_CACHE) > 4:
                    keep_key = str(log_dir.parent)
                    for key in list(_LOADER_CACHE.keys()):
                        if key != keep_key:
                            _LOADER_CACHE.pop(key, None)
            except Exception:
                pass

        metrics = {}
        combined_gt_path = self._prepare_gt_if_needed(
            split=split,
            log_prompt_pairs_path=effective_lpp_path,
            gt_annotations_path=gt_annotations_path,
            gt_combined_output_path=gt_combined_output_path,
            log_root=Path(log_root),
        )
        if combined_gt_path is not None and combined_gt_path.exists():
            combined_pred_path = combine_pkls(official_output_root, effective_lpp_path, suffix="_predictions")
            metrics = evaluate_pkls(combined_pred_path, combined_gt_path, self.config.official_output_root / experiment_name)
        return metrics
