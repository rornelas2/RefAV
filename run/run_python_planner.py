"""Agentic Python-code planner for AV2 Scenario Mining.

The LLM emits Python code that uses refAV.atomic_functions. After each
attempt we:

  1. Execute the generated code (exec in a prepared namespace).
  2. If execution raises, feed the traceback + prior code back as feedback.
  3. If execution succeeds and a prediction pkl is written, QUANTITATIVELY
     SCORE it against ground truth for this (log, prompt) pair, and feed a
     metric diagnosis + the prior code back as feedback.
  4. Iterate up to --max-attempts times. Keep the best attempt even if no
     attempt reaches the target score, so we never regress.
  5. Successful attempts (score >= memory threshold) are written to a
     persistent memory jsonl. Future prompts retrieve top-k similar memory
     entries (Jaccard over prompt tokens) as few-shot examples for their
     first attempt — continuous learning across runs.

For splits without GT (test), scoring is skipped and only traceback
feedback is used. The agentic loop still runs; it just can't measure.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import refAV.paths as paths
from refAV.code_generation import build_context
from refAV.dataset_conversion import (
    create_gt_mining_pkls_parallel,
    create_gt_pkl_file,
    separate_scenario_mining_annotations,
)
from refAV.eval import combine_pkls, evaluate_pkls
from refAV.utils import construct_caches
import refAV.atomic_functions as atomic_functions
from refAV.atomic_functions import *  # noqa: F401,F403 - for exec namespace

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "into", "that", "this", "is", "are", "has",
    "have", "be", "been", "being", "as", "it", "its", "near", "while",
    "any", "some", "within",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic Python-code planner.")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--log-prompt-pairs", type=Path, required=True)
    p.add_argument("--log-root", type=Path, required=True)
    p.add_argument("--gt-annotations", type=Path, required=True)
    p.add_argument("--gt-combined-pkl", type=Path, required=True)
    p.add_argument("--output-root", type=Path,
                   default=Path("/home/rornelas5/scratch/refav_output"))
    p.add_argument("--planner-model-name",
                   default="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4")
    p.add_argument("--model-max-memory-gpu", default="34GiB")
    p.add_argument("--model-max-memory-cpu", default="24GiB")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=4,
                   help="Total generation attempts per prompt (1 initial + N corrections).")
    p.add_argument("--target-score", type=float, default=0.5,
                   help="If an attempt reaches this score, stop refining.")
    p.add_argument("--memory-threshold", type=float, default=0.3,
                   help="Write successful attempts to long-term memory above this score.")
    p.add_argument("--memory-path", type=Path,
                   default=Path("/home/rornelas5/scratch/refav_output/python_planner_memory.jsonl"))
    p.add_argument("--memory-topk", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_planner(model_name: str, max_gpu: str, max_cpu: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[planner] loading {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory={0: max_gpu, "cpu": max_cpu},
        trust_remote_code=True,
    )
    model.eval()
    print("[planner] load complete", flush=True)
    return model, tokenizer


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int,
             temperature: float, top_p: float) -> str:
    import torch
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = None
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(gen, skip_special_tokens=True)
    finally:
        del inputs
        if out is not None:
            del out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Code extraction + execution
# --------------------------------------------------------------------------- #

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(decoded: str) -> str | None:
    m = CODE_FENCE_RE.search(decoded)
    if m:
        return m.group(1).strip()
    if "output_scenario" in decoded:
        return decoded.strip()
    return None


def execute_code(code: str, description: str, log_dir: Path, output_dir: Path) -> None:
    ns: dict[str, Any] = {}
    for name in dir(atomic_functions):
        if not name.startswith("_"):
            ns[name] = getattr(atomic_functions, name)
    ns.update({
        "log_dir": log_dir,
        "output_dir": output_dir,
        "description": description,
        "__name__": "__refav_exec__",
    })
    exec(code, ns)


# --------------------------------------------------------------------------- #
# Quantitative scoring: per-prompt comparison of predicted vs GT REFERRED sets
# --------------------------------------------------------------------------- #

def _collect_referred(frames: list[dict]) -> tuple[set[str], int, int]:
    """Return (track_uuid set, #timestamps with any REFERRED, total #timestamps)."""
    uuids: set[str] = set()
    ts_with_ref = 0
    for fr in frames:
        names = fr.get("name", [])
        tids = fr.get("track_id", [])
        hit = False
        for n, t in zip(names, tids):
            if n == "REFERRED_OBJECT":
                uuids.add(t)
                hit = True
        if hit:
            ts_with_ref += 1
    return uuids, ts_with_ref, len(frames)


def score_prediction(pred_pkl: Path, gt_combined: dict, log_id: str, prompt: str
                     ) -> tuple[float, str]:
    """Return (score in [0,1], human-readable diagnosis)."""
    if not pred_pkl.exists():
        return 0.0, "No prediction pkl was written."
    try:
        with open(pred_pkl, "rb") as f:
            pred = pickle.load(f)
    except Exception as exc:
        return 0.0, f"Prediction pkl failed to load: {exc}"
    pred_frames = pred.get((log_id, prompt), [])
    gt_frames = gt_combined.get((log_id, prompt), [])
    if not gt_frames:
        return 0.0, "Ground truth missing for this (log, prompt) — cannot score."

    pred_ref, pred_ts_ref, pred_ts_total = _collect_referred(pred_frames)
    gt_ref, gt_ts_ref, gt_ts_total = _collect_referred(gt_frames)

    if not gt_ref:
        # Degenerate: GT has no REFERRED. Score = 1 iff we also return nothing.
        if not pred_ref:
            return 1.0, "GT empty and prediction empty — trivial match."
        return 0.0, (f"GT has no REFERRED tracks for this prompt, but your code "
                     f"returned {len(pred_ref)} tracks. This category/filter combo "
                     f"does not match the scene.")

    inter = pred_ref & gt_ref
    union = pred_ref | gt_ref
    track_iou = len(inter) / max(len(union), 1)

    # Timestamp-level: overlap of ts_ref windows if we have frame counts
    if gt_ts_ref > 0:
        ts_recall = min(pred_ts_ref, gt_ts_ref) / gt_ts_ref
    else:
        ts_recall = 1.0 if pred_ts_ref == 0 else 0.0

    score = 0.6 * track_iou + 0.4 * ts_recall

    diag_parts = [
        f"Track IoU={track_iou:.3f} (pred={len(pred_ref)}, gt={len(gt_ref)}, overlap={len(inter)}).",
        f"Timestamp coverage ratio={ts_recall:.3f} (pred frames w/ REFERRED={pred_ts_ref}, gt={gt_ts_ref}).",
    ]
    if not pred_ref:
        diag_parts.append("Your prediction is EMPTY; the filter chain is probably too strict or the category is wrong.")
    elif len(pred_ref) > 5 * max(len(gt_ref), 1):
        diag_parts.append(f"Your prediction has {len(pred_ref)} tracks vs only {len(gt_ref)} in GT — filter is too loose.")
    elif not inter:
        diag_parts.append("None of your track uuids overlap GT — likely wrong category or wrong spatial/temporal predicate.")
    elif len(inter) < len(gt_ref):
        miss = len(gt_ref) - len(inter)
        diag_parts.append(f"You are missing {miss} GT track(s). Consider loosening the filter or checking the scenario description more carefully.")
    return score, " ".join(diag_parts)


# --------------------------------------------------------------------------- #
# Long-term memory of successful code
# --------------------------------------------------------------------------- #

def _tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z_]+", s.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 2}


class Memory:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.entries.append(json.loads(line))
                    except Exception:
                        pass
            print(f"[memory] loaded {len(self.entries)} prior entries from {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    def retrieve(self, prompt: str, k: int = 3) -> list[dict]:
        if not self.entries:
            return []
        toks = _tokens(prompt)
        scored = []
        for e in self.entries:
            et = _tokens(e.get("prompt", ""))
            if not toks or not et:
                continue
            j = len(toks & et) / max(len(toks | et), 1)
            if j > 0:
                scored.append((j, e))
        scored.sort(key=lambda x: (-x[0], -x[1].get("score", 0.0)))
        return [e for _, e in scored[:k]]

    def add(self, prompt: str, code: str, score: float, diagnosis: str) -> None:
        entry = {
            "prompt": prompt,
            "code": code,
            "score": round(score, 4),
            "diagnosis": diagnosis,
        }
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #

def build_user_prompt(base_context: str, description: str,
                      memory_examples: list[dict],
                      attempt_history: list[dict]) -> str:
    prompt = base_context.replace("{natural_language_description}", description)

    if memory_examples:
        mem_blob = "\n\n".join(
            f"# Prior successful solution (score={e['score']}) for prompt: "
            f"{e['prompt']!r}\n```python\n{e['code']}\n```"
            for e in memory_examples
        )
        prompt += (
            "\n\nFew-shot memory of prior successful solutions to similar prompts:\n"
            + mem_blob
        )

    if attempt_history:
        last = attempt_history[-1]
        prompt += "\n\nYou have made " + str(len(attempt_history)) + " prior attempt(s) on this prompt."
        prompt += "\nYour most recent attempt was:\n```python\n" + last["code"].strip() + "\n```\n"
        if last["error"]:
            prompt += "It raised the following error:\n```\n" + last["error"].strip() + "\n```\n"
        else:
            prompt += (
                f"It executed successfully but scored {last['score']:.3f}.\n"
                f"Evaluation diagnosis: {last['diagnosis']}\n"
            )
        prompt += (
            "Reflect on what went wrong and produce a CORRECTED Python block. "
            "Keep the log_dir/description/output_dir interface and still call "
            "output_scenario(<result>, description, log_dir, output_dir) at the end."
        )
    return prompt


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    print(f"[run_python_planner] args={vars(args)}", flush=True)

    output_root = args.output_root
    exp_dir = output_root / "sm_predictions" / "python_planner" / args.experiment_name
    pred_dir = exp_dir / "scenario_predictions"
    code_dir = exp_dir / "code"
    trace_dir = exp_dir / "traces"
    for d in (pred_dir, code_dir, trace_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(args.log_prompt_pairs) as f:
        lpp: dict[str, list[str]] = json.load(f)
    pairs = [(lid, p) for lid, prompts in lpp.items() for p in prompts]
    if args.max_items and len(pairs) > args.max_items:
        pairs = pairs[: args.max_items]
        capped: dict[str, list[str]] = {}
        for lid, p in pairs:
            capped.setdefault(lid, []).append(p)
        lpp = capped
    print(f"[run_python_planner] processing {len(pairs)} pairs", flush=True)

    sm_data_split_path = output_root / "sm_dataset" / args.split
    sm_data_split_path.mkdir(parents=True, exist_ok=True)
    combined_gt_path = args.gt_combined_pkl
    combined_gt_path.parent.mkdir(parents=True, exist_ok=True)
    capped_lpp_path = exp_dir / "log_prompt_pairs_capped.json"
    with open(capped_lpp_path, "w") as f:
        json.dump(lpp, f, indent=2)

    if not combined_gt_path.exists():
        separate_scenario_mining_annotations(args.gt_annotations, sm_data_split_path)
        create_gt_mining_pkls_parallel(
            args.gt_annotations,
            sm_data_split_path,
            num_processes=max(1, int(0.9 * os.cpu_count())),
        )
        create_gt_pkl_file(sm_data_split_path, capped_lpp_path, output_path=combined_gt_path)
        print(f"[run_python_planner] GT combined -> {combined_gt_path}", flush=True)

    with open(combined_gt_path, "rb") as f:
        gt_combined = pickle.load(f)

    construct_caches([args.log_root / lid for lid in lpp.keys()])

    base_context = build_context(context_path=REPO_ROOT / "run" / "llm_prompting" / "RefAV")
    memory = Memory(args.memory_path)

    model, tokenizer = load_planner(
        args.planner_model_name, args.model_max_memory_gpu, args.model_max_memory_cpu
    )

    stats = {"total": 0, "hit_target": 0, "kept_partial": 0, "fallback_empty": 0,
             "retries_used": 0, "memory_writes": 0}
    t_start = time.time()

    for log_id, description in pairs:
        stats["total"] += 1
        log_dir = args.log_root / log_id
        out_pkl = pred_dir / log_id / f"{description}_predictions.pkl"
        print(f"\n[pair {stats['total']}/{len(pairs)}] log={log_id[:8]} :: {description!r}", flush=True)

        mem_examples = memory.retrieve(description, k=args.memory_topk)
        if mem_examples:
            print(f"  memory: {len(mem_examples)} few-shot examples retrieved")

        attempt_history: list[dict] = []
        best = None  # {"score": float, "code": str, "diagnosis": str, "attempt": int}

        for attempt in range(args.max_attempts):
            user_prompt = build_user_prompt(base_context, description, mem_examples, attempt_history)
            decoded = generate(
                model, tokenizer, user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            code = extract_code(decoded)
            if code is None:
                attempt_history.append({
                    "code": "<no code extracted>", "error": "no ```python``` code block found",
                    "score": 0.0, "diagnosis": "",
                })
                print(f"  attempt {attempt}: no code block extracted")
                continue

            (code_dir / log_id).mkdir(parents=True, exist_ok=True)
            with open(code_dir / log_id / f"{description}__try{attempt}.py", "w") as f:
                f.write(code)

            # Wipe stale pkl so we detect whether this attempt wrote a new one.
            if out_pkl.exists():
                try: out_pkl.unlink()
                except Exception: pass

            err_text: str | None = None
            try:
                execute_code(code, description, log_dir, pred_dir)
            except Exception as exc:
                err_text = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=6)
                print(f"  attempt {attempt} raised: {type(exc).__name__}: {exc}")

            if err_text is not None:
                attempt_history.append({
                    "code": code, "error": err_text, "score": 0.0, "diagnosis": "",
                })
                continue

            if not out_pkl.exists():
                attempt_history.append({
                    "code": code, "error": None, "score": 0.0,
                    "diagnosis": ("Code ran but no prediction pkl was emitted — "
                                  "make sure to call output_scenario(...)."),
                })
                print(f"  attempt {attempt}: ran but no pkl emitted")
                continue

            score, diagnosis = score_prediction(out_pkl, gt_combined, log_id, description)
            print(f"  attempt {attempt} score={score:.3f}  {diagnosis}")
            attempt_history.append({
                "code": code, "error": None, "score": score, "diagnosis": diagnosis,
            })
            if best is None or score > best["score"]:
                best = {"score": score, "code": code, "diagnosis": diagnosis, "attempt": attempt}
            if score >= args.target_score:
                break

        # Save the full attempt trace for later debugging.
        (trace_dir / log_id).mkdir(parents=True, exist_ok=True)
        with open(trace_dir / log_id / f"{description}__trace.json", "w") as f:
            json.dump({
                "log_id": log_id,
                "prompt": description,
                "memory_used": [e["prompt"] for e in mem_examples],
                "attempts": attempt_history,
                "best": best,
            }, f, indent=2, default=str)

        # Ensure the winning attempt is what's on disk.
        if best is not None and best["score"] > 0:
            if best["attempt"] != len(attempt_history) - 1 or not out_pkl.exists():
                try:
                    if out_pkl.exists():
                        out_pkl.unlink()
                    execute_code(best["code"], description, log_dir, pred_dir)
                except Exception as exc:
                    print(f"  re-exec of best attempt failed: {exc}")
            if best["score"] >= args.target_score:
                stats["hit_target"] += 1
            else:
                stats["kept_partial"] += 1
            if best["score"] >= args.memory_threshold:
                memory.add(description, best["code"], best["score"], best["diagnosis"])
                stats["memory_writes"] += 1
        else:
            print(f"  all {args.max_attempts} attempts failed; writing empty fallback")
            try:
                from refAV.atomic_functions import output_scenario as _os
                _os({}, description, log_dir, pred_dir)
            except Exception as e:
                print(f"  fallback write failed: {e}")
            stats["fallback_empty"] += 1

        stats["retries_used"] += max(0, len(attempt_history) - 1)
        print(f"  running stats: {stats}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n[run_python_planner] finished in {elapsed:.1f}s; stats={stats}", flush=True)

    combined_preds_path = combine_pkls(pred_dir, capped_lpp_path, suffix="_predictions")
    print(f"[run_python_planner] combined preds -> {combined_preds_path}")

    if args.split in ("train", "val"):
        metrics = evaluate_pkls(combined_preds_path, combined_gt_path, exp_dir)
        print(f"[run_python_planner] metrics: {metrics}")


if __name__ == "__main__":
    main()
