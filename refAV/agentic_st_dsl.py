from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
import json
from typing import Any, Optional


JsonValue = Any


class ProgramValidationError(ValueError):
    """Raised when a DSL program is malformed."""


@dataclass
class StepSpec:
    name: str
    op: str
    source: Optional[str] = None
    related_source: Optional[str] = None
    params: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "StepSpec":
        # Robustness: Qwen and other LLMs frequently nest `source` / `related_source`
        # inside `params` instead of placing them at the step top level. Lift them
        # back out so we don't reject structurally valid programs over shape drift.
        params = dict(data.get("params", {}))
        source = data.get("source")
        related_source = data.get("related_source")
        if source is None and "source" in params:
            source = params.pop("source")
        if related_source is None and "related_source" in params:
            related_source = params.pop("related_source")
        return cls(
            name=str(data["name"]),
            op=str(data["op"]),
            source=source,
            related_source=related_source,
            params=params,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "op": self.op,
            "source": self.source,
            "related_source": self.related_source,
            "params": self.params,
        }


@dataclass
class ScenarioProgram:
    description: str
    result: str
    steps: list[StepSpec]
    notes: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ScenarioProgram":
        steps = [StepSpec.from_dict(step) for step in data.get("steps", [])]
        # Robustness: Qwen often omits `source` on unary steps, relying on
        # implicit chaining from the previous step. Fill in the chain so we
        # don't reject otherwise-valid programs.
        for i in range(1, len(steps)):
            if steps[i].source is None and steps[i].op != "get_objects_of_category":
                steps[i].source = steps[i - 1].name
        return cls(
            description=str(data["description"]),
            result=str(data["result"]),
            steps=steps,
            notes=str(data.get("notes", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "description": self.description,
            "result": self.result,
            "steps": [step.to_dict() for step in self.steps],
            "notes": self.notes,
            "metadata": self.metadata,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return md5(payload.encode("utf-8")).hexdigest()

    def clone(self) -> "ScenarioProgram":
        return ScenarioProgram.from_dict(self.to_dict())


@dataclass
class ProgramCandidate:
    program: ScenarioProgram
    provenance: str
    planner_feedback: str = ""
    search_round: int = 0
    candidate_score: Optional[float] = None
    candidate_feedback: Optional[str] = None
