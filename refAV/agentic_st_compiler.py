from __future__ import annotations

import math
from typing import Any

from refAV.agentic_st_dsl import ProgramValidationError, ScenarioProgram
from refAV.agentic_st_ops import DEFAULT_OPERATION_SPECS


def validate_program(program: ScenarioProgram) -> None:
    if not program.steps:
        raise ProgramValidationError("Program must contain at least one step.")

    available_symbols: set[str] = set()
    for step in program.steps:
        if step.name in available_symbols:
            raise ProgramValidationError(f"Duplicate step name: {step.name}")
        if step.op not in DEFAULT_OPERATION_SPECS:
            raise ProgramValidationError(f"Unsupported operation: {step.op}")

        spec = DEFAULT_OPERATION_SPECS[step.op]
        if spec.kind != "source" and not step.source:
            raise ProgramValidationError(f"{step.op} requires a source.")
        if step.source and step.source not in available_symbols:
            raise ProgramValidationError(f"Unknown source '{step.source}' in step '{step.name}'.")
        if spec.allows_related_source:
            if not step.related_source:
                raise ProgramValidationError(f"{step.op} requires related_source.")
            if step.related_source not in available_symbols:
                raise ProgramValidationError(
                    f"Unknown related_source '{step.related_source}' in step '{step.name}'."
                )
        elif step.related_source:
            raise ProgramValidationError(f"{step.op} does not accept related_source.")

        missing = [param for param in spec.required_params if param not in step.params]
        if missing:
            raise ProgramValidationError(f"{step.op} missing required params {missing}.")

        available_symbols.add(step.name)

    if program.result not in available_symbols:
        raise ProgramValidationError(f"Result step '{program.result}' is not defined.")


def _python_literal(value: Any) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "np.inf" if value > 0 else "-np.inf"
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(_python_literal(item) for item in value) + ")"
    if isinstance(value, dict):
        body = ", ".join(f"{_python_literal(key)}: {_python_literal(val)}" for key, val in value.items())
        return "{" + body + "}"
    return repr(value)


def compile_program_to_code(program: ScenarioProgram) -> str:
    validate_program(program)
    lines = [
        "import numpy as np",
        "",
        "def find_scenario(log_dir, output_dir):",
    ]
    for step in program.steps:
        spec = DEFAULT_OPERATION_SPECS[step.op]
        params = ", ".join(f"{key}={_python_literal(val)}" for key, val in step.params.items())
        if spec.kind == "source":
            call = f"{step.op}(log_dir, {params})" if params else f"{step.op}(log_dir)"
        elif spec.allows_related_source:
            call_args = [step.source, step.related_source, "log_dir"]
            if params:
                call_args.append(params)
            call = f"{step.op}(" + ", ".join(call_args) + ")"
        else:
            call_args = [step.source, "log_dir"]
            if params:
                call_args.append(params)
            call = f"{step.op}(" + ", ".join(call_args) + ")"
        lines.append(f"    {step.name} = {call}")
    lines.append(f"    final_dict = {program.result}")
    lines.append("    if final_dict:")
    safe_description = program.description.replace("\\", "\\\\").replace("'", "\\'")
    lines.append(f"        output_scenario(final_dict, '{safe_description}', log_dir, output_dir)")
    lines.append("        return final_dict")
    lines.append("    return {}")
    return "\n".join(lines)
