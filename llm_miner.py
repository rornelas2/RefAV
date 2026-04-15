"""
=============================================================================
CELL 1 — Imports & Paths
=============================================================================
"""
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from refAV.atomic_functions import (
    get_objects_of_category as refav_get_objects_of_category,
    changing_lanes as refav_changing_lanes,
    turning as refav_turning,
    has_objects_in_relative_direction as refav_has_objects_in_relative_direction,
    get_objects_in_relative_direction as refav_get_objects_in_relative_direction,
    accelerating,
    has_velocity,
    stationary,
    near_intersection,
    on_intersection,
    at_stop_sign,
    at_pedestrian_crossing,
    on_road,
    in_drivable_area,
    within_camera_view,
    is_color,
    is_snowy_scene,
    on_lane_type,
    on_relative_side_of_road,
    near_objects as refav_near_objects,
    following as refav_following,
    facing_toward as refav_facing_toward,
    heading_toward as refav_heading_toward,
    heading_in_relative_direction_to as refav_heading_in_relative_direction_to,
    being_crossed_by as refav_being_crossed_by,
    in_same_lane as refav_in_same_lane,
    scenario_and,
    scenario_or,
    output_scenario,
)
from refAV.utils import cache_manager

RAW_DATA_ROOT = Path("/home/rornelas5/scratch/argoverse_data/sensor/val")
REFAV_DATA_ROOT = Path("/home/rornelas5/scratch/argoverse_data/refav_sensor/val")
log_prompt_input_path = Path("/home/rornelas5/scratch/argoverse_data/scenario_mining/log_prompt_pairs_val.json")


def resolve_log_dir(log_id: str) -> Path:
    converted_log_dir = REFAV_DATA_ROOT / log_id
    if converted_log_dir.exists():
        return converted_log_dir
    return RAW_DATA_ROOT / log_id

if log_prompt_input_path.exists():
    with open(log_prompt_input_path, 'rb') as f:
        log_prompts = json.load(f)
    print("✅ Metadata loaded successfully.")
else:
    print(f"❌ Still cannot find file at: {log_prompt_input_path}")

data_root = REFAV_DATA_ROOT if REFAV_DATA_ROOT.exists() else RAW_DATA_ROOT
logs = sorted([f for f in os.listdir(data_root) if os.path.isdir(data_root / f)])
print(f"✅ Found {len(logs)} logs in {data_root}.")


"""
=============================================================================
CELL 2 — Config
=============================================================================
"""
import gc
import re
import ast
import traceback
import time
import signal

from transformers import AutoProcessor, AutoModelForImageTextToText

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH        = "google/gemma-4-E4B-it"
GT_DATAFRAME_PATH = "/home/rornelas5/scratch/argoverse_data/scenario_mining/scenario_mining_val_annotations.feather"
ACTUAL_LOG_DIR    = str(resolve_log_dir("20dd185d-b4eb-3024-a17a-b4e5d8b15b65"))
OUTPUT_DIR        = "./tmp_output"
API_PATH          = "/home/rornelas5/data/RefAV/refAV/atomic_functions.py"
TARGET_LOG_ID     = "20dd185d-b4eb-3024-a17a-b4e5d8b15b65"
SCENARIO_DESC     = "vehicle turning right with pedestrian in front"

print("Config set.")


"""
=============================================================================
CELL 3 — Load GT data + Model
=============================================================================
"""
import pyarrow.dataset as ds
from transformers import BitsAndBytesConfig

print("Scanning GT feather (column-pruned, no full RAM load)...")
dataset = ds.dataset(GT_DATAFRAME_PATH, format="ipc")
needed_cols = ["log_id", "prompt", "track_uuid", "timestamp_ns", "mining_category"]
filtered_table = dataset.to_table(
    filter=ds.field("log_id") == TARGET_LOG_ID,
    columns=needed_cols,
)
gt_dataframe = filtered_table.to_pandas()
del filtered_table, dataset
gc.collect()
print(f"✅ Loaded {len(gt_dataframe)} GT rows ({gt_dataframe.memory_usage(deep=True).sum() / 1e6:.1f} MB).")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {MODEL_PATH} in 4-bit NF4...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={0: "44GiB", "cpu": "32GiB"},
)
model.eval()
print("✅ Model loaded.")


"""
=============================================================================
CELL 4 — Pre-compute and PERSIST RefAV intermediate results
=============================================================================
KEY FIX: We store the results in module-level variables and NEVER delete them.
         The exec_namespace lambdas return these cached objects instantly,
         so find_scenario never touches the NFS filesystem again.
=============================================================================
"""
print("Pre-computing RefAV intermediates (this takes ~60s, only runs once)...")
t0 = time.time()

# These are stored as globals — do NOT del them
CACHED_VEHICLES     = refav_get_objects_of_category(Path(ACTUAL_LOG_DIR), 'REGULAR_VEHICLE')
CACHED_PEDESTRIANS  = refav_get_objects_of_category(Path(ACTUAL_LOG_DIR), 'PEDESTRIAN')
CACHED_LEFT_CHANGES = refav_changing_lanes(CACHED_VEHICLES, Path(ACTUAL_LOG_DIR), direction='left')

print(f"✅ Pre-computed in {time.time()-t0:.1f}s")
print(f"   Vehicles found:      {len(CACHED_VEHICLES)}")
print(f"   Pedestrians found:   {len(CACHED_PEDESTRIANS)}")
print(f"   Left lane changes:   {len(CACHED_LEFT_CHANGES)}")


"""
=============================================================================
CELL 5 — ReflexionAgent
=============================================================================
"""

def extract_api_reference(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())
    api_docs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                api_docs.append(
                    f"def {node.name}(...):\n    \"\"\"\n    {docstring}\n    \"\"\""
                )
    return "\n\n".join(api_docs)


class ReflexionAgent:
    MAX_HISTORY_TURNS = 2

    def __init__(self, model, processor, api_file_path):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        self.full_api_code = extract_api_reference(api_file_path)
        self._system_prompt = ""
        self._rolling_history = []

    def start_new_task(self, description):
        self._system_prompt = f"""You are a senior autonomous vehicle data mining engineer.
Your task is to write a single Python function named `find_scenario(log_dir, output_dir)`.
Here is the API documentation for the RefAV library you must use:
<api_reference>
{self.full_api_code}
</api_reference>

CRITICAL RULES:
1. Base Objects: Start with `get_objects_of_category(log_dir, 'REGULAR_VEHICLE')`.
2. Chaining: Pass the OUTPUT of one function directly as the INPUT to the next. DO NOT use `scenario_and`. DO NOT manually intersect dicts.
3. Braking means passing min_accel=-np.inf and max_accel=-1.0 to accelerating().
4. If your final dict is not empty, call `output_scenario(final_dict, '{description}', log_dir, output_dir)`.
5. Return the final dict. Return {{}} if empty (before calling output_scenario).
6. Do NOT use try/except blocks.
7. Output ONLY a ```python ... ``` code block. No prose, no explanation.

CORRECT chaining example (do exactly this pattern):
```python
def find_scenario(log_dir, output_dir):
    import numpy as np
    vehicles = get_objects_of_category(log_dir, 'REGULAR_VEHICLE')
    filtered = some_filter(vehicles, log_dir, param=value)
    final_dict = another_filter(filtered, log_dir, param=value)
    if final_dict:
        output_scenario(final_dict, '{description}', log_dir, output_dir)
        return final_dict
    return {{}}
```

Scenario: '{description}'. Write the find_scenario function now."""
        self._rolling_history = []

    def generate_code(self, feedback=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if feedback:
            self._rolling_history.append({
                "role": "user",
                "content": [{"type": "text", "text":
                    f"CRITICAL FEEDBACK: {feedback}\n"
                    f"Rewrite find_scenario. Fix the error. Output ONLY the python code block."}],
            })

        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(self._rolling_history) > max_msgs:
            self._rolling_history = self._rolling_history[-max_msgs:]

        first_user_turn = {
            "role": "user",
            "content": [{"type": "text", "text": self._system_prompt}],
        }
        messages = [first_user_turn] + self._rolling_history

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = inputs["input_ids"].shape[-1]
        print(f"  [mem] Input tokens: {input_length} | "
              f"GPU alloc: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

            new_tokens_cpu = generated_ids[0][input_length:].cpu()
            del generated_ids, inputs
            torch.cuda.empty_cache()

            response = self.processor.decode(new_tokens_cpu, skip_special_tokens=True)
            del new_tokens_cpu

            response_for_history = response[:800] + '...[truncated]' if len(response) > 800 else response
            self._rolling_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_for_history}],
            })

            def _extract_code(text):
                match = re.search(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
                if match:
                    code = match.group(1).strip()
                    code = re.sub(r'^```(?:python)?\n?', '', code).strip()
                    code = re.sub(r'\n?```$', '', code).strip()
                    return code
                lines = [l for l in text.splitlines() if not l.strip().startswith('```')]
                return '\n'.join(lines).strip()

            extracted = _extract_code(response)
            if extracted:
                return extracted
            else:
                print("  [warn] Could not extract code from response.")
                return ""

        except torch.cuda.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            gc.collect()
            print("  [OOM] Caught CUDA OOM during generation.")
            return ""

        except Exception as e:
            print(f"  [err] Generation error: {e}")
            return ""


"""
=============================================================================
CELL 6 — Evaluator
=============================================================================
"""

def evaluate_prediction(predicted_scenario_dict, log_id, df, target_prompt=None):
    if target_prompt is None:
        target_prompt = SCENARIO_DESC
    target_prompt = target_prompt.lower()
    log_events = df[
        (df['log_id'] == log_id) &
        (df['prompt'].str.lower() == target_prompt)
    ]

    if log_events.empty:
        if not predicted_scenario_dict:
            return 1.0, "True Negative: No events in GT, and code found nothing. Perfect."
        else:
            return 0.0, "False Positive: GT has NO events for this log, but code triggered."

    # --- Track-level F1 ---
    gt_referred = log_events[log_events['mining_category'] == 'REFERRED_OBJECT'] if 'mining_category' in log_events.columns else log_events
    gt_ids = set(gt_referred['track_uuid'].astype(str).tolist())
    predicted_ids = set(str(k) for k in predicted_scenario_dict.keys()) if predicted_scenario_dict else set()

    track_tp = len(predicted_ids & gt_ids)
    track_fp = len(predicted_ids - gt_ids)
    track_fn = len(gt_ids - predicted_ids)

    if track_tp == 0 and not predicted_ids:
        return 0.0, f"False Negative: GT has {len(gt_ids)} referred tracks but code found nothing. Loosen filters."
    if track_tp == 0:
        return 0.0, f"Found IDs {list(predicted_ids)[:3]}, but correct IDs are {list(gt_ids)[:3]}."

    track_precision = track_tp / (track_tp + track_fp) if (track_tp + track_fp) > 0 else 0
    track_recall = track_tp / (track_tp + track_fn) if (track_tp + track_fn) > 0 else 0
    track_f1 = 2 * track_precision * track_recall / (track_precision + track_recall) if (track_precision + track_recall) > 0 else 0

    # --- Timestamp-level F1 (critical for HOTA metric) ---
    timestamp_f1 = 0.0
    if 'timestamp_ns' in log_events.columns and predicted_scenario_dict:
        gt_ts_map = {}
        for _, row in gt_referred.iterrows():
            ts = int(row['timestamp_ns'])
            gt_ts_map.setdefault(ts, set()).add(str(row['track_uuid']))

        pred_ts_map = {}
        for track_uuid, track_data in predicted_scenario_dict.items():
            if isinstance(track_data, dict):
                for ts in track_data.keys():
                    pred_ts_map.setdefault(int(ts), set()).add(str(track_uuid))

        all_ts = sorted(set(gt_ts_map.keys()) | set(pred_ts_map.keys()))
        if all_ts:
            ts_f1s = []
            for ts in all_ts:
                p_ids = pred_ts_map.get(ts, set())
                g_ids = gt_ts_map.get(ts, set())
                tp = len(p_ids & g_ids)
                fp = len(p_ids - g_ids)
                fn = len(g_ids - p_ids)
                pr = tp / (tp + fp) if (tp + fp) > 0 else 0
                rc = tp / (tp + fn) if (tp + fn) > 0 else 0
                ts_f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0)
            timestamp_f1 = sum(ts_f1s) / len(ts_f1s)
    else:
        timestamp_f1 = track_f1  # fallback if no timestamp data

    # Combined score: weight tracks and timestamps (matches competition emphasis)
    combined = 0.50 * track_f1 + 0.50 * timestamp_f1

    if combined >= 0.999:
        return 1.0, "Perfect match."

    feedback_parts = [f"track_F1={track_f1:.2f}(P={track_precision:.2f},R={track_recall:.2f})"]
    feedback_parts.append(f"timestamp_F1={timestamp_f1:.2f}")
    if track_fn > 0:
        feedback_parts.append(f"missed {track_fn} tracks")
    if track_fp > 0:
        feedback_parts.append(f"hallucinated {track_fp} tracks")
    return combined, ". ".join(feedback_parts) + ". Adjust logic."
    
"""
=============================================================================
CELL 7 — Reflexion Loop
=============================================================================
KEY FIX: exec_namespace uses lambda wrappers that return the pre-computed
         CACHED_VEHICLES and CACHED_LEFT_CHANGES instantly, bypassing NFS.
         Only accelerating() actually runs — it's a fast in-memory filter.

KEY FIX: Cleanup block does NOT call fn.cache_clear() or clear cache_manager,
         since we want the underlying RefAV cache to stay warm.
=============================================================================
"""

class ScenarioTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise ScenarioTimeoutError("find_scenario timed out after 300s")

agent = ReflexionAgent(model, processor, api_file_path=API_PATH)
agent.start_new_task(SCENARIO_DESC)

max_attempts = 5
best_score   = -1.0
current_feedback = None

def cached_get_objects_of_category(log_dir, category):
    log_dir = Path(log_dir)
    if log_dir == Path(ACTUAL_LOG_DIR):
        if category == 'REGULAR_VEHICLE':
            return CACHED_VEHICLES
        if category == 'PEDESTRIAN':
            return CACHED_PEDESTRIANS
    return refav_get_objects_of_category(log_dir, category)


def cached_changing_lanes(candidates, log_dir, **kwargs):
    log_dir = Path(log_dir)
    direction = kwargs.get('direction')
    if log_dir == Path(ACTUAL_LOG_DIR) and candidates is CACHED_VEHICLES and direction == 'left':
        return CACHED_LEFT_CHANGES
    return refav_changing_lanes(candidates, log_dir, **kwargs)


exec_namespace = {
    '__builtins__': __builtins__,
    # Cached wrappers for expensive ops
    'get_objects_of_category': cached_get_objects_of_category,
    'changing_lanes':          cached_changing_lanes,
    # Motion filters (fast in-memory)
    'turning':                 refav_turning,
    'accelerating':            accelerating,
    'has_velocity':            has_velocity,
    'stationary':              stationary,
    # Relational/spatial ops
    'has_objects_in_relative_direction': refav_has_objects_in_relative_direction,
    'get_objects_in_relative_direction': refav_get_objects_in_relative_direction,
    'near_objects':             refav_near_objects,
    'following':               refav_following,
    'facing_toward':           refav_facing_toward,
    'heading_toward':          refav_heading_toward,
    'heading_in_relative_direction_to': refav_heading_in_relative_direction_to,
    'being_crossed_by':        refav_being_crossed_by,
    'in_same_lane':            refav_in_same_lane,
    # Scene/map context
    'near_intersection':       near_intersection,
    'on_intersection':         on_intersection,
    'at_stop_sign':            at_stop_sign,
    'at_pedestrian_crossing':  at_pedestrian_crossing,
    'on_road':                 on_road,
    'in_drivable_area':        in_drivable_area,
    'on_lane_type':            on_lane_type,
    'on_relative_side_of_road': on_relative_side_of_road,
    # Visual
    'within_camera_view':      within_camera_view,
    'is_color':                is_color,
    'is_snowy_scene':          is_snowy_scene,
    # Combinators + output
    'scenario_and':            scenario_and,
    'scenario_or':             scenario_or,
    'output_scenario':         output_scenario,
    'np':                      np,
    'Path':                    Path,
}

print(f"\n🚀 Starting Autonomous Mining for: '{SCENARIO_DESC}'")
print(f"   GPU memory before loop: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated, "
      f"{torch.cuda.memory_reserved() / 1e9:.1f} GB reserved\n")

for attempt in range(max_attempts):
    print(f"--- Attempt {attempt + 1}/{max_attempts} ---")

    # 1. Generate Code
    t0 = time.time()
    code = agent.generate_code(feedback=current_feedback)
    print(f"  Generation took {time.time()-t0:.1f}s")

    if not code or 'def find_scenario' not in code:
        print("  Code missing or truncated. Skipping.")
        current_feedback = (
            "Your previous response was truncated or malformed. "
            "Output ONLY a complete ```python ... ``` block containing "
            "the full find_scenario function. Do not add any prose."
        )
        continue

    print("  Generated Code:\n" + code + "\n")

    # 2. Execute in isolated namespace
    execution_error   = None
    predicted_results = None
    run_ns = dict(exec_namespace)

    try:
        exec(compile(code, "<llm_code>", "exec"), run_ns)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)
        try:
            t0 = time.time()
            predicted_results = run_ns['find_scenario'](
                log_dir=Path(ACTUAL_LOG_DIR),
                output_dir=Path(OUTPUT_DIR),
            )
            print(f"  find_scenario took {time.time()-t0:.1f}s")
        finally:
            signal.alarm(0)

    except ScenarioTimeoutError:
        execution_error = "TimeoutError: find_scenario exceeded 300 seconds."
        print(f"  ⏰ {execution_error}")
    except Exception:
        execution_error = traceback.format_exc()

    # 3. Evaluate / set feedback
    if execution_error:
        print("  Code crashed:\n" + execution_error)
        current_feedback = (
            f"Your code threw a Python exception:\n{execution_error[-750:]}\n"
            "Fix the error and output ONLY the corrected python code block."
        )
        score = 0.0
    else:
        score, eval_feedback = evaluate_prediction(predicted_results, TARGET_LOG_ID, gt_dataframe)
        print(f"  Score: {score:.2f} | {eval_feedback}")

        if score == 1.0:
            print("\n✅ Perfect score achieved! Mining complete.")
            break

        if score > best_score:
            best_score = score

        if score == 0.0 and "False Positive" in eval_feedback:
            current_feedback = (
                "False Positive: GT has NO matching events for this log, but your code returned results. "
                "Your filters are too permissive. Add stricter filters to narrow down results. "
                "Chain function calls: output of one goes as input to the next."
            )
        elif score == 0.0 and "False Negative" in eval_feedback:
            current_feedback = f"{eval_feedback} Your filters are too strict. Loosen your thresholds or remove a filter step."
        elif "timestamp_F1" in eval_feedback and "track_F1" in eval_feedback:
            # Parse scores for targeted feedback
            current_feedback = eval_feedback
            if "timestamp_F1=0" in eval_feedback and "track_F1=1" not in eval_feedback:
                current_feedback += " Focus on getting the right timestamps — use temporal filters."
        else:
            current_feedback = eval_feedback

    # 4. Memory cleanup — NOTE: do NOT clear RefAV caches or call fn.cache_clear()
    del run_ns
    if predicted_results is not None:
        del predicted_results
    gc.collect()

    if hasattr(model, '_cache'):
        del model._cache
    if hasattr(model, 'past_key_values'):
        del model.past_key_values

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB alloc | "
          f"{torch.cuda.memory_reserved()/1e9:.1f} GB reserved")

else:
    print(f"\nFinished. Max attempts reached. Best score: {best_score:.2f}.")
