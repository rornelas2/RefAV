from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import re
from pathlib import Path
from typing import Optional

from refAV.agentic_st_compiler import validate_program
from refAV.agentic_st_config import AgenticSearchConfig
from refAV.agentic_st_dsl import ProgramCandidate, ScenarioProgram, StepSpec
from refAV.agentic_st_knowledge import FewShotKnowledgeBase, KnowledgeBaseEntry
from refAV.agentic_st_memory import MemoryEntry, MemoryStore
from refAV.agentic_st_ops import build_dsl_reference, infer_category_mentions
from refAV.agentic_st_visual import VisualCoarseContext, extract_prompt_visual_hints
from refAV.utils import get_img_crop


def _extract_json_objects(text: str) -> list[dict]:
    candidates: list[dict] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    raw_segments = fenced if fenced else [text]
    for segment in raw_segments:
        segment = segment.strip()
        try:
            data = json.loads(segment)
            if isinstance(data, dict):
                candidates.append(data)
            elif isinstance(data, list):
                candidates.extend(item for item in data if isinstance(item, dict))
            continue
        except json.JSONDecodeError:
            pass

        starts = [idx for idx, char in enumerate(segment) if char in "[{"]
        for start in starts:
            for end in range(len(segment), start + 1, -1):
                snippet = segment[start:end]
                try:
                    data = json.loads(snippet)
                    if isinstance(data, dict):
                        candidates.append(data)
                    elif isinstance(data, list):
                        candidates.extend(item for item in data if isinstance(item, dict))
                    break
                except json.JSONDecodeError:
                    continue
    return candidates


def _category_for_prompt(prompt: str) -> str:
    mentions = infer_category_mentions(prompt)
    if prompt.lower().startswith("ego vehicle"):
        return "EGO_VEHICLE"
    if mentions:
        return mentions[0]
    return "REGULAR_VEHICLE"


class HeuristicSeedPlanner:
    def propose(
        self,
        prompt: str,
        max_candidates: int = 8,
        retrieved_memory: Optional[list[MemoryEntry]] = None,
        feedback: Optional[str] = None,
    ) -> list[ProgramCandidate]:
        prompt_lower = prompt.lower()
        referred_category = _category_for_prompt(prompt)
        category_mentions = infer_category_mentions(prompt)
        related_categories = [category for category in category_mentions if category != referred_category]
        mentioned_colors, mentioned_cameras, weather_hints = extract_prompt_visual_hints(prompt)

        base_steps = [
            StepSpec(
                name="objects",
                op="get_objects_of_category",
                params={"category": referred_category},
            )
        ]
        current_symbol = "objects"
        variants: list[ScenarioProgram] = []

        if mentioned_colors:
            color_name = mentioned_colors[0]
            color_step_name = f"{color_name}_tracks"
            base_steps.append(StepSpec(name=color_step_name, op="is_color", source=current_symbol, params={"color": color_name}))
            current_symbol = color_step_name

        if mentioned_cameras:
            camera_step_name = "camera_visible_tracks"
            base_steps.append(
                StepSpec(
                    name=camera_step_name,
                    op="within_camera_view",
                    source=current_symbol,
                    params={"camera_name": mentioned_cameras[0]},
                )
            )
            current_symbol = camera_step_name

        if "turning right" in prompt_lower:
            base_steps.append(StepSpec(name="turning_right", op="turning", source=current_symbol, params={"direction": "right"}))
            current_symbol = "turning_right"
        elif "turning left" in prompt_lower:
            base_steps.append(StepSpec(name="turning_left", op="turning", source=current_symbol, params={"direction": "left"}))
            current_symbol = "turning_left"
        elif "turning" in prompt_lower:
            base_steps.append(StepSpec(name="turning_any", op="turning", source=current_symbol, params={"direction": None}))
            current_symbol = "turning_any"

        if "changing lanes" in prompt_lower or "lane change" in prompt_lower:
            direction = None
            if "left" in prompt_lower:
                direction = "left"
            elif "right" in prompt_lower:
                direction = "right"
            base_steps.append(StepSpec(name="lane_change", op="changing_lanes", source=current_symbol, params={"direction": direction}))
            current_symbol = "lane_change"

        if "accelerating" in prompt_lower:
            base_steps.append(
                StepSpec(
                    name="accelerating_tracks",
                    op="accelerating",
                    source=current_symbol,
                    params={"min_accel": 0.5, "max_accel": float("inf")},
                )
            )
            current_symbol = "accelerating_tracks"
        elif "braking" in prompt_lower or "decelerating" in prompt_lower:
            base_steps.append(
                StepSpec(
                    name="braking_tracks",
                    op="accelerating",
                    source=current_symbol,
                    params={"min_accel": float("-inf"), "max_accel": -1.0},
                )
            )
            current_symbol = "braking_tracks"

        if "stationary" in prompt_lower or "waiting" in prompt_lower or "stopped" in prompt_lower or "parked" in prompt_lower:
            base_steps.append(StepSpec(name="stationary_tracks", op="stationary", source=current_symbol))
            current_symbol = "stationary_tracks"

        if "moving" in prompt_lower or "in motion" in prompt_lower:
            base_steps.append(
                StepSpec(
                    name="moving_tracks",
                    op="has_velocity",
                    source=current_symbol,
                    params={"min_velocity": 0.5},
                )
            )
            current_symbol = "moving_tracks"

        if "jaywalking" in prompt_lower or "jay walking" in prompt_lower:
            base_steps.append(StepSpec(name="on_road_tracks", op="on_road", source=current_symbol))
            current_symbol = "on_road_tracks"

        if "drivable area" in prompt_lower:
            base_steps.append(StepSpec(name="drivable_tracks", op="in_drivable_area", source=current_symbol))
            current_symbol = "drivable_tracks"

        if "busy roundabout" in prompt_lower or "intersection" in prompt_lower:
            op = "near_intersection" if "near" in prompt_lower or "busy" in prompt_lower else "on_intersection"
            base_steps.append(StepSpec(name="intersection_tracks", op=op, source=current_symbol))
            current_symbol = "intersection_tracks"

        if "stop sign" in prompt_lower:
            base_steps.append(StepSpec(name="stop_sign_tracks", op="at_stop_sign", source=current_symbol))
            current_symbol = "stop_sign_tracks"

        if "on road" in prompt_lower:
            base_steps.append(StepSpec(name="road_tracks", op="on_road", source=current_symbol))
            current_symbol = "road_tracks"

        if "crosswalk" in prompt_lower or "pedestrian crossing" in prompt_lower:
            base_steps.append(StepSpec(name="crosswalk_tracks", op="at_pedestrian_crossing", source=current_symbol))
            current_symbol = "crosswalk_tracks"

        # Bike-lane phrasing has many forms in val prompts: "bike lane",
        # "bicycle lane", "bicycle sharing lane", "shared bike lane", etc.
        bike_lane_phrases = ("bike lane", "bicycle lane", "bicycle sharing lane",
                             "shared bike lane", "shared bicycle lane")
        if any(phrase in prompt_lower for phrase in bike_lane_phrases):
            base_steps.append(StepSpec(name="bike_lane_tracks", op="on_lane_type", source=current_symbol, params={"lane_type": "BIKE"}))
            current_symbol = "bike_lane_tracks"
        elif "bus lane" in prompt_lower:
            base_steps.append(StepSpec(name="bus_lane_tracks", op="on_lane_type", source=current_symbol, params={"lane_type": "BUS"}))
            current_symbol = "bus_lane_tracks"

        if "snow" in weather_hints:
            base_steps.append(
                StepSpec(
                    name="snowy_scene_tracks",
                    op="is_snowy_scene",
                    source=current_symbol,
                    params={"camera_name": "ring_front_center"},
                )
            )
            current_symbol = "snowy_scene_tracks"

        direction_phrase_map = {
            "in front": "forward",
            "ahead": "forward",
            "behind": "backward",
            "on the left": "left",
            "on the right": "right",
        }
        relational_variants: list[StepSpec] = []
        if related_categories:
            related_category = related_categories[0]
            related_source = "related_objects"
            relational_prefix = [
                StepSpec(name=related_source, op="get_objects_of_category", params={"category": related_category})
            ]
            relation_direction = None
            for phrase, direction in direction_phrase_map.items():
                if phrase in prompt_lower:
                    relation_direction = direction
                    break
            if relation_direction:
                for distance in (15.0, 25.0, 50.0):
                    relational_variants.append(
                        StepSpec(
                            name=f"result_{int(distance)}",
                            op="has_objects_in_relative_direction",
                            source=current_symbol,
                            related_source=related_source,
                            params={
                                "direction": relation_direction,
                                "min_number": 1,
                                "within_distance": distance,
                                "lateral_thresh": float("inf"),
                            },
                        )
                    )
            elif "nearby" in prompt_lower or "near " in prompt_lower or "with nearby" in prompt_lower or "with near" in prompt_lower:
                for distance in (8.0, 12.0, 20.0):
                    relational_variants.append(
                        StepSpec(
                            name=f"result_{int(distance)}",
                            op="near_objects",
                            source=current_symbol,
                            related_source=related_source,
                            params={"distance_thresh": distance},
                        )
                    )
            elif "following" in prompt_lower:
                relational_variants.append(
                    StepSpec(
                        name="following_result",
                        op="following",
                        source=current_symbol,
                        related_source=related_source,
                        params={"distance_thresh": 20.0},
                    )
                )
            elif "toward" in prompt_lower:
                relational_variants.append(
                    StepSpec(
                        name="toward_result",
                        op="heading_toward",
                        source=current_symbol,
                        related_source=related_source,
                        params={},
                    )
                )
            elif "same lane" in prompt_lower:
                relational_variants.append(
                    StepSpec(
                        name="same_lane_result",
                        op="in_same_lane",
                        source=current_symbol,
                        related_source=related_source,
                        params={},
                    )
                )
            elif "crossing" in prompt_lower or "crossed by" in prompt_lower:
                relational_variants.append(
                    StepSpec(
                        name="crossing_result",
                        op="being_crossed_by",
                        source=current_symbol,
                        related_source=related_source,
                        params={},
                    )
                )

            if not relational_variants:
                relational_variants.append(
                    StepSpec(
                        name="result_default",
                        op="near_objects",
                        source=current_symbol,
                        related_source=related_source,
                        params={"distance_thresh": 15.0},
                    )
                )

            for relational_step in relational_variants:
                steps = [StepSpec.from_dict(step.to_dict()) for step in base_steps]
                steps.extend(relational_prefix)
                steps.append(relational_step)
                variants.append(
                    ScenarioProgram(
                        description=prompt,
                        result=relational_step.name,
                        steps=steps,
                        notes="heuristic seed",
                        metadata={"planner": "heuristic"},
                    )
                )

        if not variants:
            variants.append(
                ScenarioProgram(
                    description=prompt,
                    result=current_symbol,
                    steps=[StepSpec.from_dict(step.to_dict()) for step in base_steps],
                    notes="heuristic seed",
                    metadata={"planner": "heuristic"},
                )
            )

        deduped = {}
        for program in variants:
            deduped[program.fingerprint()] = program
        ordered = list(deduped.values())[:max_candidates]
        return [ProgramCandidate(program=program, provenance="heuristic") for program in ordered]


@dataclass
class HuggingFaceJSONPlanner:
    config: AgenticSearchConfig
    _model: object = None
    _tokenizer: object = None
    _processor: object = None
    _device: object = None

    def _model_name(self) -> str:
        model_name = self.config.local_model_path or self.config.planner_model_name
        if model_name is None:
            raise ValueError("planner_model_name or local_model_path must be set for HF planner.")
        return model_name

    def _uses_gemma_image_text_backend(self) -> bool:
        return "gemma-4" in self._model_name().lower()

    def _uses_gptq_model(self) -> bool:
        """Detect GPTQ models by name convention (e.g. '-GPTQ-' or '-gptq-')."""
        return "gptq" in self._model_name().lower()

    def _lazy_load(self) -> None:
        if self._model is not None and (self._tokenizer is not None or self._processor is not None):
            return
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            GPTQConfig,
        )
        import torch

        model_name = self._model_name()

        quantization_config = None
        if self._uses_gptq_model():
            quantization_config = GPTQConfig(bits=4, use_exllama=True)
        elif self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if self._uses_gemma_image_text_backend():
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.config.model_device_map,
                max_memory={0: self.config.model_max_memory_gpu, "cpu": self.config.model_max_memory_cpu},
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.config.model_device_map,
                max_memory={0: self.config.model_max_memory_gpu, "cpu": self.config.model_max_memory_cpu},
            )
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def _build_prompt(
        self,
        prompt: str,
        retrieved_memory: list[MemoryEntry],
        knowledge_examples: list[KnowledgeBaseEntry],
        feedback: Optional[str],
        seed_candidates: list[ProgramCandidate],
        visual_context: Optional[VisualCoarseContext],
    ) -> str:
        lines = [
            "You are planning a RefAV spatio-temporal mining program for autonomous vehicle scenario mining.",
            "Your program must identify the CORRECT tracks AND the CORRECT timestamps when the scenario occurs.",
            "Timestamp precision is critical: the competition metric (HOTA) heavily penalizes temporal misalignment.",
            "Use specific filters (turning, accelerating, near_intersection, etc.) to narrow timestamps, not just track IDs.",
            "When a prompt describes two alternative behaviors (e.g. 'turning OR accelerating'), use the 'union' op to merge both filter chains.",
            "CRITICAL: Never append a 'union' step after a binary predicate (heading_toward, has_objects_in_relative_direction, near_objects, following, being_crossed_by, etc.). Binary predicates already attach the related objects as nested metadata; the REFERRED object set is exactly their output. If you union the output with the related category (e.g. 'pedestrians'), every pedestrian in the log becomes REFERRED and HOTA collapses to 0. The final result of a scenario that involves a relationship should be the direct output of the binary predicate.",
            "CRITICAL: Respect direction words literally. If the prompt says 'right', use direction='right'; if it says 'left', use direction='left'. Never silently swap them or default to the opposite.",
            "CRITICAL: Do not chain mutually contradictory unary filters on the same source. 'stationary' after 'turning', 'accelerating' after 'stationary', 'has_velocity(min_velocity>0)' after 'stationary', etc. all yield the empty set. If the prompt says 'waiting', use 'stationary'; if it says 'turning', use 'turning' — never both in the same chain.",
            "CRITICAL: Do not compose 'at_stop_sign' with a 'has_objects_in_relative_direction(..., stop_signs, forward, ...)' step. A vehicle that is 'at' a stop sign has the sign inside its bounding box, not forward of it — the composition returns empty. For 'vehicle in front of stop sign', use has_objects_in_relative_direction(vehicles, stop_signs, direction='forward') on the plain vehicle category directly.",
            "CRITICAL: To express a TEMPORAL window like 'within 15 seconds' or 'twice within N seconds', use the 'within_time_window' op with window_seconds=N and min_events=2 on the event filter. Do NOT use 'near_objects(distance_thresh=N)' for a time window — distance_thresh is meters, not seconds.",
            "Return ONLY JSON. Prefer concise, valid programs over clever ones.",
            build_dsl_reference(),
            "",
            "Retrieved memory:",
            MemoryStore.format_for_prompt(retrieved_memory),
            "",
            "Few-shot knowledge base:",
            FewShotKnowledgeBase.format_for_prompt(knowledge_examples),
            "",
            "Visual coarse-filter context:",
            visual_context.summary if visual_context is not None else "No visual coarse-filter context available.",
            "",
            "Seed candidates:",
        ]
        for idx, candidate in enumerate(seed_candidates, start=1):
            lines.append(f"Seed {idx}: {json.dumps(candidate.program.to_dict(), indent=2, sort_keys=True)}")
        if feedback:
            lines.extend(["", "Feedback from prior attempts:", feedback])
        lines.extend(
            [
                "",
                f"Prompt: {prompt}",
                "",
                "Before writing JSON, reason through these three dimensions:",
                "  1. Referred object category — which AV category is the main subject? (e.g. PEDESTRIAN, REGULAR_VEHICLE, BICYCLIST)",
                "  2. Temporal behavior — what is the object doing? (e.g. turning, accelerating, stationary, changing_lanes)",
                "  3. Spatial relationship — is there a secondary object or location constraint? (e.g. near intersection, ahead of vehicle, at crosswalk)",
                "     If the prompt describes two alternative behaviors, plan a 'union' branch for the OR case.",
                "",
                "Now output a JSON array of 1-6 candidate programs.",
            ]
        )
        return "\n".join(lines)

    def propose(
        self,
        prompt: str,
        max_candidates: int = 6,
        retrieved_memory: Optional[list[MemoryEntry]] = None,
        knowledge_examples: Optional[list[KnowledgeBaseEntry]] = None,
        feedback: Optional[str] = None,
        seed_candidates: Optional[list[ProgramCandidate]] = None,
        visual_context: Optional[VisualCoarseContext] = None,
        log_dir: Optional[Path] = None,
    ) -> list[ProgramCandidate]:
        self._lazy_load()
        retrieved_memory = retrieved_memory or []
        knowledge_examples = knowledge_examples or []
        seed_candidates = seed_candidates or []

        prompt_text = self._build_prompt(prompt, retrieved_memory, knowledge_examples, feedback, seed_candidates, visual_context)
        if self._uses_gemma_image_text_backend():
            content = [{"type": "text", "text": prompt_text}]
            if visual_context is not None and log_dir is not None:
                for camera_name, timestamp in visual_context.planner_images[: self.config.visual_planner_num_images]:
                    try:
                        image = get_img_crop(camera_name, int(timestamp), log_dir)
                        if image is not None:
                            content.append({"type": "image", "image": image.convert("RGB")})
                    except Exception:
                        continue
            messages = [{"role": "user", "content": content}]
            try:
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                fallback_messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
                inputs = self._processor.apply_chat_template(
                    fallback_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            prompt_length = inputs["input_ids"].shape[-1]
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.temperature > 0,
                "num_return_sequences": self.config.num_return_sequences,
            }
            if self.config.temperature > 0:
                generation_kwargs["temperature"] = self.config.temperature
                generation_kwargs["top_p"] = self.config.top_p
            try:
                outputs = self._model.generate(**inputs, **generation_kwargs)
                decoded_outputs = [
                    self._processor.decode(output[prompt_length:].cpu(), skip_special_tokens=True)
                    for output in outputs
                ]
            finally:
                # Aggressive cleanup: image-text backend (gemma-4) leaks
                # intermediate tensors across calls and OOMs after a few rounds.
                try:
                    del outputs
                except Exception:
                    pass
                try:
                    del inputs
                except Exception:
                    pass
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.temperature > 0,
                "temperature": self.config.temperature,
                "pad_token_id": self._tokenizer.eos_token_id,
                "num_return_sequences": self.config.num_return_sequences,
            }
            if self.config.temperature > 0:
                generation_kwargs["top_p"] = self.config.top_p

            try:
                outputs = self._model.generate(**inputs, **generation_kwargs)
                prompt_length = inputs["input_ids"].shape[-1]
                decoded_outputs = [
                    self._tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
                    for output in outputs
                ]
            finally:
                try:
                    del outputs
                except Exception:
                    pass
                try:
                    del inputs
                except Exception:
                    pass
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        candidates: list[ProgramCandidate] = []
        parse_errors: list[str] = []
        extracted_any = False
        for decoded in decoded_outputs:
            for candidate_dict in _extract_json_objects(decoded):
                extracted_any = True
                try:
                    if "steps" in candidate_dict:
                        candidate_dict.setdefault("description", prompt)
                        program = ScenarioProgram.from_dict(candidate_dict)
                        validate_program(program)
                        candidates.append(
                            ProgramCandidate(
                                program=program,
                                provenance="hf_planner",
                                planner_feedback=feedback or "",
                            )
                        )
                    elif "programs" in candidate_dict:
                        for item in candidate_dict["programs"]:
                            if isinstance(item, dict) and "steps" in item:
                                item.setdefault("description", prompt)
                                program = ScenarioProgram.from_dict(item)
                                validate_program(program)
                                candidates.append(
                                    ProgramCandidate(
                                        program=program,
                                        provenance="hf_planner",
                                        planner_feedback=feedback or "",
                                    )
                                )
                except Exception as exc:
                    parse_errors.append(f"{type(exc).__name__}: {exc}")
                    continue
        if not candidates:
            head = (decoded_outputs[0] if decoded_outputs else "")[:500].replace("\n", " | ")
            print(
                f"[planner-debug] prompt={prompt!r} decoded_n={len(decoded_outputs)} "
                f"extracted_json={extracted_any} parse_errors={parse_errors[:3]} "
                f"head=\"{head}\"",
                flush=True,
            )
        deduped = {}
        for candidate in candidates:
            deduped[candidate.program.fingerprint()] = candidate
        return list(deduped.values())[:max_candidates]


class CombinedPlanner:
    def __init__(self, config: AgenticSearchConfig):
        self.config = config
        self.heuristic = HeuristicSeedPlanner() if config.enable_heuristic_planner else None
        self.knowledge_base = FewShotKnowledgeBase(config.knowledge_base_path, embedding_model_name=config.embedding_model_name)
        self.hf_planner = HuggingFaceJSONPlanner(config) if config.enable_hf_planner and (
            config.planner_model_name or config.local_model_path
        ) else None

    def propose(
        self,
        prompt: str,
        retrieved_memory: Optional[list[MemoryEntry]] = None,
        feedback: Optional[str] = None,
        visual_context: Optional[VisualCoarseContext] = None,
        log_dir: Optional[Path] = None,
    ) -> list[ProgramCandidate]:
        retrieved_memory = retrieved_memory or []
        knowledge_examples = self.knowledge_base.retrieve(prompt, top_k=self.config.max_knowledge_examples)
        heuristic_candidates: list[ProgramCandidate] = []
        if self.heuristic is not None:
            heuristic_candidates = self.heuristic.propose(
                prompt=prompt,
                max_candidates=self.config.max_candidates_per_round,
                retrieved_memory=retrieved_memory,
                feedback=feedback,
            )

        model_candidates: list[ProgramCandidate] = []
        if self.hf_planner is not None:
            try:
                model_candidates = self.hf_planner.propose(
                    prompt=prompt,
                    max_candidates=self.config.max_candidates_per_round,
                    retrieved_memory=retrieved_memory,
                    knowledge_examples=knowledge_examples,
                    feedback=feedback,
                    seed_candidates=heuristic_candidates,
                    visual_context=visual_context,
                    log_dir=log_dir,
                )
            except Exception as exc:
                print(f"[warn] HF planner failed: {exc}")
                model_candidates = []
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                except Exception:
                    pass

        deduped = {}
        for candidate in heuristic_candidates + model_candidates:
            try:
                validate_program(candidate.program)
            except Exception:
                continue
            deduped[candidate.program.fingerprint()] = candidate
        return list(deduped.values())[: self.config.max_candidates_per_round]
