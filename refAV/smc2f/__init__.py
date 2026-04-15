"""Vendored subset of the SMc2f (scenario mining coarse-to-fine) pipeline.

Upstream: https://anonymous.4open.science/r/test-EE20  (local copy at
/home/rornelas5/scenario-mining/smc2f/SMc2f).

Stage 1: CLIP coarse filter (CLIPCoarseFilter / CLIPFeatureExtractor).
Stage 2: DETTM fine filter (DETTMMatcher) — checkpoint trained via
         /home/rornelas5/scenario-mining/train_dettm.py.

Import layout:
  - `SMc2fConfig` and `filter_scenario_by_segments` are pure-python and
    always safe to import.
  - Heavy classes (CLIPCoarseFilter, CLIPFeatureExtractor, DETTMMatcher)
    are imported lazily via `__getattr__` because they pull in the heavy
    `clip` + `torch` + `av2` deps which are only present in the runtime venv.
"""

from .config import SMc2fConfig
from .segment_filter import filter_scenario_by_segments

__all__ = [
    "SMc2fConfig",
    "CLIPCoarseFilter",
    "CLIPFeatureExtractor",
    "DETTMMatcher",
    "filter_scenario_by_segments",
]


def __getattr__(name: str):
    if name in ("CLIPCoarseFilter", "CLIPFeatureExtractor"):
        from . import clip_coarse_filter as _ccf
        return getattr(_ccf, name)
    if name == "DETTMMatcher":
        from .dettm_matcher import DETTMMatcher
        return DETTMMatcher
    raise AttributeError(f"module 'refAV.smc2f' has no attribute {name!r}")
