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
import shutil
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
from refAV.utils import construct_caches, read_feather
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
    p.add_argument("--gt-annotations", type=Path, default=None,
                   help="Path to GT annotations feather (required for val/test evaluation, optional for test submission).")
    p.add_argument("--gt-combined-pkl", type=Path, default=None,
                   help="Path to combined GT pkl (required for val evaluation, optional for test submission).")
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
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                   help="Model backend. 'vllm' requires a separate venv with vLLM installed "
                        "and handles per-expert MoE GPTQ checkpoints that transformers cannot.")
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.92,
                   help="vLLM gpu_memory_utilization (0..1). Only used with --backend vllm.")
    p.add_argument("--vllm-max-model-len", type=int, default=8192,
                   help="vLLM max_model_len cap. Only used with --backend vllm.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip pairs whose prediction pkl already exists on disk "
                        "(for resuming an interrupted run).")
    p.add_argument("--quantize", choices=["none", "bnb-4bit", "bnb-8bit", "no-marlin"],
                   default="none",
                   help="Quantization mode. 'bnb-4bit'/'bnb-8bit' use bitsandbytes NF4/INT8. "
                        "'no-marlin' auto-detects the model's native quant format "
                        "(GPTQ/AutoRound/AWQ) and overrides its backend to avoid "
                        "the missing gptqmodel_marlin_kernels C++ extension.")
    p.add_argument("--enable-smc2f-clip-filter", action="store_true",
                   help="Pre-filter annotations.feather to CLIP-selected temporal windows "
                        "before the LLM generates code. Requires --smc2f-clip-cache-dir.")
    p.add_argument("--smc2f-clip-cache-dir", type=Path, default=None,
                   help="Path to precomputed CLIP feature cache (contains val/<log_id>/*.npz).")
    p.add_argument("--smc2f-split", default="val",
                   help="Dataset split name used to look up CLIP features (default: val).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_planner(model_name: str, max_gpu: str, max_cpu: str,
                  quantize: str = "none", backend: str = "hf",
                  vllm_gpu_mem_util: float = 0.92, vllm_max_model_len: int = 8192):
    if backend == "vllm":
        return _load_with_vllm(model_name, vllm_gpu_mem_util, vllm_max_model_len)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[planner] loading {model_name}  (quantize={quantize})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs: dict = dict(
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory={0: max_gpu, "cpu": max_cpu},
        trust_remote_code=True,
    )

    if quantize.startswith("bnb-"):
        from transformers import BitsAndBytesConfig
        if quantize == "bnb-4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:  # bnb-8bit
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        # bitsandbytes handles placement; remove max_memory to avoid conflicts
        load_kwargs.pop("max_memory", None)
        load_kwargs.pop("torch_dtype", None)
    elif quantize == "no-marlin":
        return _load_with_backend_retry(model_name, tokenizer, load_kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    print("[planner] load complete", flush=True)
    return model, tokenizer


def _load_with_vllm(model_name: str, gpu_mem_util: float, max_model_len: int):
    """Load model via vLLM. Used for Qwen3.5 MoE GPTQ checkpoints where the
    per-expert GPTQ layout is incompatible with transformers 5.5's packed MoE
    classes (no conversion path in transformers/conversion_mapping.py for
    qwen3_5_moe, and the GPTQ quantizer contributes no expert-level converters).
    vLLM has a native fused-expert GPTQ MoE loader that handles this layout.
    """
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"[planner] loading {model_name} via vLLM "
          f"(gpu_mem_util={gpu_mem_util}, max_model_len={max_model_len})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        enforce_eager=True,  # skip CUDA graph capture — saves ~2GB for huge models
    )
    print("[planner] vLLM load complete", flush=True)
    return llm, tokenizer


def _is_vllm_model(model) -> bool:
    return type(model).__module__.startswith("vllm")


def _load_with_backend_retry(model_name, tokenizer, base_load_kwargs):
    """Try a list of candidate backends in order; return the first that loads.

    Why this exists: the installed gptqmodel wheel ships WITHOUT compiled CUDA
    extensions (no marlin/exllamav2/exllama/cuda kernels). Many "default"
    quantization backends therefore fail with ModuleNotFoundError at the layer
    construction step. Each model also has constraints: native packing format
    (auto_round / auto_round:auto_gptq / auto_round:auto_awq), sym vs asym
    quant, and architecture-specific layers (e.g. MoE experts) that some
    backends can't handle.

    Strategy: detect the model's quant_method, build an ordered list of
    candidate backends from fastest-likely-to-work to slowest-most-compatible
    (pure pytorch). Try each; on ValueError/ModuleNotFoundError/ImportError,
    log and try the next. If all fail, raise an informative error.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    # Pre-import gptqmodel: transformers lazily imports it inside from_pretrained
    # under accelerate's meta-device context, where gptqmodel's exllamav3_torch
    # module-level Tensor.item() call fails. Importing here caches the constants.
    import gptqmodel  # noqa: F401

    # Neutralize transformers' caching_allocator_warmup. For large quantized
    # models, gptqmodel's loader places weights on GPU *before* transformers
    # reaches this warmup; the warmup then tries to pre-allocate another full
    # model-sized contiguous buffer and OOMs the GPU (e.g. 122B GPTQ needs
    # ~62GB already-allocated plus another 62GB warmup on an 80GB A100).
    # The warmup is purely a loading-speed optimization, so disabling it only
    # slows first-load slightly and has no effect on inference correctness.
    import transformers.modeling_utils as _tmu
    _tmu.caching_allocator_warmup = lambda *a, **kw: None

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    qcfg = getattr(cfg, "quantization_config", None) or {}
    if not isinstance(qcfg, dict):
        qcfg = qcfg.to_dict() if hasattr(qcfg, "to_dict") else {}
    quant_method = qcfg.get("quant_method", "")
    bits = qcfg.get("bits", 4)
    group_size = qcfg.get("group_size", 128)
    print(f"[planner] detected quant_method={quant_method!r}, "
          f"bits={bits}, group_size={group_size}", flush=True)

    # Build candidate list: (label, quantization_config_instance) per quant_method.
    # Order: triton (fast, no C++ kernels) → pure torch (slow, always works).
    # Each quant_method demands a distinct config class — AutoRoundConfig on a
    # pure-GPTQ model raises "pre_quantized=False" because transformers checks
    # the class type against the stored quantization_method.
    from transformers import AwqConfig, AutoRoundConfig, GPTQConfig
    if quant_method == "awq":
        zp = qcfg.get("zero_point", True)
        candidates = [
            ("awq:gemm",        AwqConfig(bits=bits, group_size=group_size, zero_point=zp, backend="gemm")),
            ("awq:gemm_triton", AwqConfig(bits=bits, group_size=group_size, zero_point=zp, backend="gemm_triton")),
            ("awq:torch",       AwqConfig(bits=bits, group_size=group_size, zero_point=zp, backend="torch_awq")),
        ]
    elif quant_method == "gptq":
        # Pure GPTQ models need GPTQConfig (NOT AutoRoundConfig). The installed
        # gptqmodel wheel has no marlin/exllamav2/cuda kernels; disable exllama
        # to force the pure-torch path, then also try exllamav2 as a fallback.
        candidates = [
            ("gptq:no-exllama",   GPTQConfig(bits=bits, group_size=group_size, use_exllama=False)),
            ("gptq:exllamav2",    GPTQConfig(bits=bits, group_size=group_size, use_exllama=True,
                                             exllama_config={"version": 2})),
        ]
    elif quant_method == "auto-round":
        candidates = [
            ("autoround:tritonv2_zp", AutoRoundConfig(bits=bits, group_size=group_size,
                                                       backend="auto_round:tritonv2_zp")),
            ("autoround:torch_zp",    AutoRoundConfig(bits=bits, group_size=group_size,
                                                       backend="auto_round:torch_zp")),
            ("autoround:tritonv2",    AutoRoundConfig(bits=bits, group_size=group_size,
                                                       backend="auto_round:tritonv2")),
            ("autoround:torch",       AutoRoundConfig(bits=bits, group_size=group_size,
                                                       backend="auto_round:torch")),
        ]
    else:
        print(f"[planner] WARNING: unknown quant_method={quant_method!r}, "
              f"loading without backend override", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **base_load_kwargs)
        model.eval()
        print("[planner] load complete", flush=True)
        return model, tokenizer

    last_err = None
    for label, qconfig in candidates:
        load_kwargs = dict(base_load_kwargs)
        load_kwargs["quantization_config"] = qconfig
        print(f"[planner] trying backend={label!r} ...", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model.eval()
            print(f"[planner] load complete with backend={label!r}", flush=True)
            return model, tokenizer
        except (ValueError, ModuleNotFoundError, ImportError) as e:
            msg = str(e).split("\n")[0][:200]
            print(f"[planner] backend={label!r} failed: {type(e).__name__}: {msg}", flush=True)
            last_err = e
            # Free any partial allocations before the next attempt
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    raise RuntimeError(
        f"All candidate backends failed for {model_name} "
        f"(quant_method={quant_method!r}). Last error: {last_err}"
    )


def generate(model, tokenizer, prompt_text: str, max_new_tokens: int,
             temperature: float, top_p: float) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if _is_vllm_model(model):
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = model.generate([text], sp, use_tqdm=False)
        return outputs[0].outputs[0].text

    import torch
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


import ast as _ast


def _build_exec_namespace(log_dir: Path, output_dir: Path, description: str) -> dict:
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
    return ns


def execute_code(code: str, description: str, log_dir: Path, output_dir: Path) -> None:
    exec(code, _build_exec_namespace(log_dir, output_dir, description))


def execute_code_traced(
    code: str, description: str, log_dir: Path, output_dir: Path
) -> tuple[list[str], list[tuple[str, dict]]]:
    """Execute code statement-by-statement, recording the size of every intermediate dict.

    Returns:
        trace_lines: human-readable lines such as "  vehicles = 47 objects"
                     with "← GOES EMPTY HERE" marking the first zero-result filter.
        non_empty_dicts: ordered list of (var_name, dict_val) for every assignment
                         that produced a non-empty scenario dict.  The last entry is
                         the best rescue candidate if the final output is empty.

    Raises SyntaxError / RuntimeError exactly like execute_code so callers can
    treat errors the same way.
    """
    ns = _build_exec_namespace(log_dir, output_dir, description)
    tree = _ast.parse(code)  # raises SyntaxError on bad code

    trace_lines: list[str] = []
    non_empty_dicts: list[tuple[str, dict]] = []
    went_empty = False

    for stmt in tree.body:
        exec(compile(_ast.Module(body=[stmt], type_ignores=[]), "<refav_traced>", "exec"), ns)

        if not isinstance(stmt, _ast.Assign):
            continue
        for target in stmt.targets:
            if not isinstance(target, _ast.Name):
                continue
            val = ns.get(target.id)
            if isinstance(val, dict):
                n = len(val)
                if n > 0:
                    trace_lines.append(f"  {target.id} = {n} objects")
                    non_empty_dicts.append((target.id, val))
                else:
                    marker = "  ← GOES EMPTY HERE" if not went_empty else ""
                    trace_lines.append(f"  {target.id} = 0 objects{marker}")
                    went_empty = True
            elif isinstance(val, (list, tuple)):
                trace_lines.append(f"  {target.id} = sequence({len(val)})")

    return trace_lines, non_empty_dicts


def rescue_with_most_precise(
    non_empty_dicts: list[tuple[str, dict]],
    description: str,
    log_dir: Path,
    pred_dir: Path,
) -> bool:
    """Write the most precise (smallest-count) non-empty intermediate as a rescue prediction.

    Choosing the smallest count rather than the last gives the most filtered
    candidate — e.g. 'vehicles_near_sign = 3' beats 'vehicles = 47' as a rescue.
    Called when all LLM attempts produced an empty output.  Returns True if a
    non-empty pkl was successfully written, False if there was nothing to rescue.
    """
    if not non_empty_dicts:
        return False
    rescue_var, rescue_dict = min(non_empty_dicts, key=lambda x: len(x[1]))
    try:
        atomic_functions.output_scenario(rescue_dict, description, log_dir, pred_dir)
        print(f"  rescue: wrote most precise intermediate '{rescue_var}' "
              f"({len(rescue_dict)} tracks) as fallback prediction")
        return True
    except Exception as exc:
        print(f"  rescue write failed: {exc}")
        return False


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
                     ) -> tuple[float, str, float, float]:
    """Return (score, diagnosis, track_iou, ts_recall).

    Returning track_iou and ts_recall separately lets callers detect the
    'wrong-track' failure mode (track_iou=0, ts_recall≥0.5) and give
    targeted retry feedback instead of the generic partial-score message.
    """
    if not pred_pkl.exists():
        return 0.0, "No prediction pkl was written.", 0.0, 0.0
    try:
        with open(pred_pkl, "rb") as f:
            pred = pickle.load(f)
    except Exception as exc:
        return 0.0, f"Prediction pkl failed to load: {exc}", 0.0, 0.0
    pred_frames = pred.get((log_id, prompt), [])
    gt_frames = gt_combined.get((log_id, prompt), [])
    if not gt_frames:
        return 0.0, "Ground truth missing for this (log, prompt) — cannot score.", 0.0, 0.0

    pred_ref, pred_ts_ref, pred_ts_total = _collect_referred(pred_frames)
    gt_ref, gt_ts_ref, gt_ts_total = _collect_referred(gt_frames)

    if not gt_ref:
        if not pred_ref:
            return 1.0, "GT empty and prediction empty — trivial match.", 1.0, 1.0
        return 0.0, (f"GT has no REFERRED tracks for this prompt, but your code "
                     f"returned {len(pred_ref)} tracks. This category/filter combo "
                     f"does not match the scene."), 0.0, 0.0

    inter = pred_ref & gt_ref
    union = pred_ref | gt_ref
    track_iou = len(inter) / max(len(union), 1)

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
    return score, " ".join(diag_parts), track_iou, ts_recall


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

    def add_negative(self, prompt: str, code: str,
                     trace: list[str], diagnosis: str) -> None:
        """Record a wholly-failed attempt so future prompts avoid the same strategy."""
        entry = {
            "prompt": prompt,
            "code": code,
            "score": 0.0,
            "diagnosis": diagnosis,
            "trace": trace,
            "negative": True,
        }
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def retrieve_negative(self, prompt: str, k: int = 2) -> list[dict]:
        """Return up to k negative-memory entries most similar to prompt."""
        if not self.entries:
            return []
        toks = _tokens(prompt)
        scored = []
        for e in self.entries:
            if not e.get("negative"):
                continue
            et = _tokens(e.get("prompt", ""))
            if not toks or not et:
                continue
            j = len(toks & et) / max(len(toks | et), 1)
            if j > 0:
                scored.append((j, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #

def build_user_prompt(base_context: str, description: str,
                      memory_examples: list[dict],
                      attempt_history: list[dict],
                      negative_examples: list[dict] | None = None) -> str:
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

    if negative_examples:
        def _neg_entry(e: dict) -> str:
            trace_note = ""
            if e.get("trace"):
                trace_note = "\n# Filter trace: " + " → ".join(e["trace"][:4])
            return (
                f"# FAILED for prompt: {e['prompt']!r}\n"
                f"# Why it failed: {e.get('diagnosis', 'all attempts scored 0')}"
                f"{trace_note}\n"
                f"```python\n{e['code']}\n```"
            )
        neg_blob = "\n\n".join(_neg_entry(e) for e in negative_examples)
        prompt += (
            "\n\nApproaches that FAILED for similar prompts — do NOT repeat these:\n"
            + neg_blob
        )

    if not attempt_history:
        # Change 3: Score formula — model knows what it is optimising for.
        prompt += (
            "\n\nSCORING: score = 0.6 × Track_IoU + 0.4 × Timestamp_recall. "
            "Track_IoU requires the exact annotated UUID(s). "
            "An object found at the right timestamps but with the wrong UUID scores only 0.4. "
            "Precision on WHICH object matters as much as WHEN."
        )
        # Change 2: Chain-of-thought planning on attempt 0.
        prompt += (
            "\n\nBefore writing code, add a brief comment block explaining:\n"
            "1. What is the single referred object the description asks about?\n"
            "2. What uniquely identifies it — spatial relationship, direction from ego, or count?\n"
            "3. If only one specific object is needed, use max_number=1 in "
            "get_objects_in_relative_direction to select the nearest match.\n"
            "IMPORTANT: always call atomic functions with POSITIONAL arguments "
            "(e.g. near_objects(vehicles, peds, log_dir)), never keyword arguments — "
            "the runtime wrappers require positional binding.\n"
            "Then write the Python code block."
        )
    else:
        last = attempt_history[-1]
        prompt += "\n\nYou have made " + str(len(attempt_history)) + " prior attempt(s) on this prompt."
        prompt += "\nYour most recent attempt was:\n```python\n" + last["code"].strip() + "\n```\n"
        if last["error"]:
            prompt += "It raised the following error:\n```\n" + last["error"].strip() + "\n```\n"
            prompt += (
                "Fix the Python error and produce a CORRECTED block. "
                "Keep the log_dir/description/output_dir interface and call "
                "output_scenario(<result>, description, log_dir, output_dir) at the end."
            )
        elif "EMPTY" in last.get("diagnosis", "") or last["score"] == 0.0:
            trace_lines = last.get("trace", [])
            trace_section = ""
            if trace_lines:
                trace_section = (
                    "Filter trace (how many objects survived each step):\n"
                    + "\n".join(trace_lines) + "\n\n"
                )
            prompt += (
                f"It executed but produced an EMPTY prediction (score={last['score']:.3f}).\n"
                f"Diagnosis: {last['diagnosis']}\n\n"
                f"{trace_section}"
                "IMPORTANT — your filter chain is too restrictive. Follow this debugging strategy:\n"
                "1. The trace above shows exactly which predicate killed the pipeline. Remove or loosen that filter first.\n"
                "2. Start minimal: query just the base category with no extra filters.\n"
                "   Only add a filter back if it keeps at least some objects.\n"
                "3. Prefer scenario_or over scenario_and wherever the description allows alternatives.\n"
                "4. Use loose distance thresholds (within_distance=50 or more); tighten only if too broad.\n"
                "5. Check you are using the right category: PEDESTRIAN (on foot), BICYCLIST (on bike), "
                "MOTORCYCLIST (on motorcycle), WHEELED_RIDER (scooter/skater). VEHICLE covers cars/trucks/buses.\n"
                "6. For braking: use accelerating(..., min_accel=-np.inf, max_accel=-1.0) — NOT has_lateral_acceleration.\n"
                "7. For temporal repetition ('two X within N seconds'): use within_time_window().\n"
                "8. For snow/weather: use is_snowy_scene().\n"
                "Produce a CORRECTED Python block that is simpler and looser than your previous attempt."
            )
        elif last.get("track_iou", 1.0) == 0.0 and last.get("ts_recall", 0.0) >= 0.5:
            # Change 1: Wrong-track retry branch — timestamps correct, UUIDs wrong.
            pred_n = last.get("pred_n", "?")
            gt_n = last.get("gt_n", "?")
            prompt += (
                f"It executed but scored {last['score']:.3f} with Track IoU=0.000 and "
                f"Timestamp recall={last['ts_recall']:.3f}.\n"
                f"Diagnosis: {last['diagnosis']}\n\n"
                "WRONG-TRACK FAILURE: your timestamps are correct but you identified the wrong "
                f"specific object(s) (pred={pred_n}, gt={gt_n}, overlap=0). "
                "This is a reference disambiguation problem — the description refers to ONE specific "
                "object among several candidates and your code picked a different one.\n\n"
                "Fix strategies (pick the most appropriate):\n"
                "1. Use max_number=1 in get_objects_in_relative_direction to select only the single "
                "   closest/most-relevant object rather than all matches.\n"
                "2. Flip the lookup direction: approach from the referred object's perspective "
                "   using reverse_relationship, or start from a related object and find what's near it.\n"
                "3. Add a tighter directional constraint (e.g. direction='forward', within_distance=15) "
                "   to narrow down to exactly one candidate.\n"
                "4. Chain two get_objects_in_relative_direction calls: first find the anchor object, "
                "   then find the referred object relative to that anchor.\n"
                "Produce a CORRECTED Python block that uniquely identifies the referred object."
            )
        else:
            prompt += (
                f"It executed successfully but scored {last['score']:.3f}.\n"
                f"Evaluation diagnosis: {last['diagnosis']}\n"
                "Reflect on what went wrong and produce a CORRECTED Python block. "
                "Keep the log_dir/description/output_dir interface and call "
                "output_scenario(<result>, description, log_dir, output_dir) at the end."
            )
    return prompt


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def prepare_clip_log_dir(
    log_dir: Path,
    clip_filter,
    description: str,
    split: str,
    tmp_root: Path,
) -> tuple[Path, bool]:
    """Create a temp log dir whose annotations.feather is CLIP-filtered.

    Returns (effective_log_dir, is_temp). If CLIP filtering is unavailable or
    would empty the annotations, returns (log_dir, False) so the caller can use
    the original directory unchanged. When is_temp=True the caller is responsible
    for deleting effective_log_dir after the pair is processed.
    """
    if clip_filter is None:
        return log_dir, False

    log_id = log_dir.name
    try:
        segments = clip_filter.get_filtered_segments(description, log_id, split)
    except Exception as exc:
        print(f"  [CLIP] segment lookup failed: {exc} — using full annotations")
        return log_dir, False

    if not segments:
        return log_dir, False

    try:
        full_ann = read_feather(log_dir / "annotations.feather")
        filtered_ann = clip_filter.filter_annotations(full_ann, segments)
    except Exception as exc:
        print(f"  [CLIP] annotation filter failed: {exc} — using full annotations")
        return log_dir, False

    n_before, n_after = len(full_ann), len(filtered_ann)
    if n_after == 0:
        print(f"  [CLIP] filter would zero out annotations — using full")
        return log_dir, False

    print(f"  [CLIP] {n_before} -> {n_after} annotation rows  ({len(segments)} segments retained)")

    tmp_dir = tmp_root / log_id
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    filtered_ann.reset_index(drop=True).to_feather(tmp_dir / "annotations.feather")

    # Symlink every other entry from the real log dir so atomic functions that
    # need the map, ego poses, calibration, and caches still work.
    for item in log_dir.iterdir():
        if item.name == "annotations.feather":
            continue
        (tmp_dir / item.name).symlink_to(item.resolve())

    return tmp_dir, True


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
    capped_lpp_path = exp_dir / "log_prompt_pairs_capped.json"
    with open(capped_lpp_path, "w") as f:
        json.dump(lpp, f, indent=2)

    # GT handling: optional for test submission (no GT labels available), required for val evaluation.
    gt_combined = None
    if args.gt_combined_pkl is not None and args.gt_annotations is not None:
        combined_gt_path = args.gt_combined_pkl
        combined_gt_path.parent.mkdir(parents=True, exist_ok=True)
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
    else:
        print(f"[run_python_planner] Running without GT (test submission mode)")


    construct_caches([args.log_root / lid for lid in lpp.keys()])

    base_context = build_context(context_path=REPO_ROOT / "run" / "llm_prompting" / "RefAV")
    memory = Memory(args.memory_path)

    # ── CLIP coarse filter (optional) ────────────────────────────────────────
    clip_filter = None
    clip_tmp_root = output_root / "clip_tmp"
    if args.enable_smc2f_clip_filter:
        if args.smc2f_clip_cache_dir is None:
            print("[CLIP] --smc2f-clip-cache-dir is required with --enable-smc2f-clip-filter; "
                  "disabling CLIP filter.")
        else:
            try:
                SMC2F_ROOT = Path("/home/rornelas5/scenario-mining/smc2f")
                if str(SMC2F_ROOT) not in sys.path:
                    sys.path.insert(0, str(SMC2F_ROOT))
                from SMc2f.config import SMc2fConfig
                from SMc2f.clip_coarse_filter import CLIPCoarseFilter
                SMc2fConfig.CLIP_CACHE_DIR = args.smc2f_clip_cache_dir
                clip_filter = CLIPCoarseFilter(device="cuda")
                clip_tmp_root.mkdir(parents=True, exist_ok=True)
                print(f"[CLIP] filter enabled  cache={args.smc2f_clip_cache_dir}")
            except Exception as exc:
                print(f"[CLIP] init failed: {exc}; disabling CLIP filter.")
    # ─────────────────────────────────────────────────────────────────────────

    model, tokenizer = load_planner(
        args.planner_model_name, args.model_max_memory_gpu, args.model_max_memory_cpu,
        quantize=args.quantize, backend=args.backend,
        vllm_gpu_mem_util=args.vllm_gpu_mem_util, vllm_max_model_len=args.vllm_max_model_len,
    )

    stats = {"total": 0, "hit_target": 0, "kept_partial": 0, "fallback_empty": 0,
             "retries_used": 0, "memory_writes": 0}
    t_start = time.time()

    for log_id, description in pairs:
        stats["total"] += 1
        log_dir = args.log_root / log_id
        out_pkl = pred_dir / log_id / f"{description}_predictions.pkl"

        if args.skip_existing and out_pkl.exists():
            print(f"\n[pair {stats['total']}/{len(pairs)}] SKIP (exists): {description!r}", flush=True)
            stats["kept_partial"] += 1  # conservative: credit as partial rather than querying GT again
            continue

        print(f"\n[pair {stats['total']}/{len(pairs)}] log={log_id[:8]} :: {description!r}", flush=True)

        # CLIP pre-filter: narrow annotations.feather to top temporal windows
        # before the LLM generates code, matching the smc2f Stage (a) approach.
        effective_log_dir, is_clip_temp = prepare_clip_log_dir(
            log_dir, clip_filter, description, args.smc2f_split, clip_tmp_root
        )

        mem_examples = memory.retrieve(description, k=args.memory_topk)
        neg_examples = memory.retrieve_negative(description, k=2)
        if mem_examples:
            print(f"  memory: {len(mem_examples)} few-shot examples retrieved")
        if neg_examples:
            print(f"  memory: {len(neg_examples)} negative examples retrieved (will warn model)")

        attempt_history: list[dict] = []
        best = None  # {"score": float, "code": str, "diagnosis": str, "attempt": int}
        # Accumulate all non-empty intermediates across every attempt so the
        # rescue can pick the globally most precise candidate.
        best_rescue_dicts: list[tuple[str, dict]] = []

        for attempt in range(args.max_attempts):
            # Temperature scheduling: start low for determinism, increase for diversity on retries.
            attempt_temp = min(args.temperature + attempt * 0.2, 1.0)
            user_prompt = build_user_prompt(
                base_context, description, mem_examples, attempt_history,
                negative_examples=neg_examples if attempt == 0 else None,
            )
            decoded = generate(
                model, tokenizer, user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=attempt_temp,
                top_p=args.top_p,
            )
            code = extract_code(decoded)
            if code is None:
                attempt_history.append({
                    "code": "<no code extracted>", "error": "no ```python``` code block found",
                    "score": 0.0, "diagnosis": "", "trace": [],
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
            trace_lines: list[str] = []
            try:
                trace_lines, non_empty_dicts = execute_code_traced(
                    code, description, effective_log_dir, pred_dir)
                # Accumulate across all attempts so rescue has the global best candidate.
                best_rescue_dicts.extend(non_empty_dicts)
            except Exception as exc:
                err_text = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=6)
                print(f"  attempt {attempt} raised: {type(exc).__name__}: {exc}")

            if err_text is not None:
                attempt_history.append({
                    "code": code, "error": err_text, "score": 0.0, "diagnosis": "",
                    "trace": trace_lines,
                })
                continue

            if not out_pkl.exists():
                attempt_history.append({
                    "code": code, "error": None, "score": 0.0,
                    "diagnosis": ("Code ran but no prediction pkl was emitted — "
                                  "make sure to call output_scenario(...)."),
                    "trace": trace_lines,
                })
                print(f"  attempt {attempt}: ran but no pkl emitted")
                continue

            if gt_combined is not None:
                score, diagnosis, track_iou, ts_recall = score_prediction(
                    out_pkl, gt_combined, log_id, description)
                print(f"  attempt {attempt} score={score:.3f}  {diagnosis}")
            else:
                score, diagnosis, track_iou, ts_recall = 1.0, "No GT available (test submission mode).", 1.0, 1.0
                print(f"  attempt {attempt} pkl_exists={out_pkl.exists()}")
            # Parse pred/gt counts from diagnosis for wrong-track retry context.
            _m = re.search(r"pred=(\d+), gt=(\d+)", diagnosis)
            pred_n = int(_m.group(1)) if _m else None
            gt_n   = int(_m.group(2)) if _m else None
            attempt_history.append({
                "code": code, "error": None, "score": score, "diagnosis": diagnosis,
                "trace": trace_lines, "track_iou": track_iou, "ts_recall": ts_recall,
                "pred_n": pred_n, "gt_n": gt_n,
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
                    execute_code(best["code"], description, effective_log_dir, pred_dir)
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
            print(f"  all {args.max_attempts} attempts scored 0; attempting rescue")
            # Write negative memory so future similar prompts avoid the same dead-end.
            last_real = next(
                (ah for ah in reversed(attempt_history)
                 if ah.get("code") and ah["code"] != "<no code extracted>"),
                None,
            )
            if last_real:
                memory.add_negative(
                    description,
                    last_real["code"],
                    last_real.get("trace", []),
                    last_real.get("diagnosis", "all attempts scored 0"),
                )
                stats["memory_writes"] += 1
            rescued = rescue_with_most_precise(best_rescue_dicts, description, effective_log_dir, pred_dir)
            if rescued:
                # Score the rescue prediction so it shows up in stats correctly.
                if gt_combined is not None:
                    rescue_score, rescue_diag, _, _ = score_prediction(out_pkl, gt_combined, log_id, description)
                    print(f"  rescue score={rescue_score:.3f}  {rescue_diag}")
                else:
                    rescue_score = 0.5  # test mode: can't score
                if rescue_score >= args.target_score:
                    stats["hit_target"] += 1
                else:
                    stats["kept_partial"] += 1
            else:
                print(f"  rescue failed (no non-empty intermediates found); writing empty fallback")
                try:
                    atomic_functions.output_scenario({}, description, effective_log_dir, pred_dir)
                except Exception as e:
                    print(f"  empty fallback write failed: {e}")
                stats["fallback_empty"] += 1

        # Clean up CLIP temp dir for this pair now that the pkl is on disk.
        if is_clip_temp and effective_log_dir.exists():
            shutil.rmtree(effective_log_dir, ignore_errors=True)

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
