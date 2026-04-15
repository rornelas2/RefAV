from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import threading
from typing import Optional

import numpy as np

from refAV.agentic_st_dsl import ScenarioProgram


# ---------------------------------------------------------------------------
# Tokenize / Jaccard (fallback when sentence-transformers unavailable)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ---------------------------------------------------------------------------
# Sentence-transformer singleton (CPU, thread-safe lazy load)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_LOCK = threading.Lock()
_EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"


def _load_embed_model(model_name: str):
    global _EMBED_MODEL, _EMBED_MODEL_NAME
    if _EMBED_MODEL is not None and _EMBED_MODEL_NAME == model_name:
        return _EMBED_MODEL
    with _EMBED_LOCK:
        if _EMBED_MODEL is None or _EMBED_MODEL_NAME != model_name:
            try:
                from sentence_transformers import SentenceTransformer
                _EMBED_MODEL = SentenceTransformer(model_name, device="cpu")
                _EMBED_MODEL_NAME = model_name
                print(f"[memory] sentence-transformers loaded: {model_name}")
            except Exception as exc:
                print(f"[warn] sentence-transformers unavailable, falling back to Jaccard: {exc}")
                _EMBED_MODEL = None
    return _EMBED_MODEL


def _embed_texts(texts: list[str], model_name: str) -> Optional[np.ndarray]:
    """Return L2-normalised embeddings of shape (N, D), or None on failure."""
    model = _load_embed_model(model_name)
    if model is None:
        return None
    try:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    except Exception as exc:
        print(f"[warn] embedding failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    prompt: str
    score: float
    feedback: str
    program: dict
    log_id: Optional[str] = None
    planner_model_name: Optional[str] = None
    split: Optional[str] = None


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, path: Path, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.path = Path(path)
        self.embedding_model_name = embedding_model_name
        self.entries: list[MemoryEntry] = []
        # Parallel list of embeddings (None until first retrieve() call)
        self._embeddings: list[Optional[np.ndarray]] = []
        self._embeddings_built = False
        self.enabled = True
        self.last_error: Optional[str] = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with open(self.path, "r") as f:
                    for line in f:
                        if line.strip():
                            self.entries.append(MemoryEntry(**json.loads(line)))
                            self._embeddings.append(None)
        except OSError as exc:
            self.enabled = False
            self.last_error = str(exc)
            print(f"[warn] MemoryStore disabled for {self.path}: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_embeddings(self) -> bool:
        """Build embeddings for all entries that don't have one yet. Returns True if usable."""
        if not self.entries:
            return False
        missing_idx = [i for i, e in enumerate(self._embeddings) if e is None]
        if not missing_idx:
            return True
        texts = [self.entries[i].prompt for i in missing_idx]
        vecs = _embed_texts(texts, self.embedding_model_name)
        if vecs is None:
            return False
        for local_i, global_i in enumerate(missing_idx):
            self._embeddings[global_i] = vecs[local_i]
        return True

    def _prompt_similarity(self, query_vec: Optional[np.ndarray], query_tokens: set[str],
                           entry_idx: int) -> float:
        entry = self.entries[entry_idx]
        if query_vec is not None and self._embeddings[entry_idx] is not None:
            return float(np.dot(query_vec, self._embeddings[entry_idx]))
        return _jaccard(query_tokens, _tokenize(entry.prompt))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        prompt: str,
        program: ScenarioProgram,
        score: float,
        feedback: str,
        log_id: Optional[str] = None,
        planner_model_name: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        entry = MemoryEntry(
            prompt=prompt,
            score=float(score),
            feedback=feedback,
            program=program.to_dict(),
            log_id=log_id,
            planner_model_name=planner_model_name,
            split=split,
        )
        self.entries.append(entry)
        # Pre-compute embedding for the new entry if model is already loaded
        vec = _embed_texts([prompt], self.embedding_model_name)
        self._embeddings.append(vec[0] if vec is not None else None)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        except OSError as exc:
            self.enabled = False
            self.last_error = str(exc)
            print(f"[warn] MemoryStore append failed for {self.path}: {exc}")

    def retrieve(self, prompt: str, top_k: int = 4) -> list[MemoryEntry]:
        if not self.entries:
            return []

        # Try embedding similarity; fall back to Jaccard per-entry
        use_embed = self._ensure_embeddings()
        query_vec: Optional[np.ndarray] = None
        if use_embed:
            q = _embed_texts([prompt], self.embedding_model_name)
            query_vec = q[0] if q is not None else None

        query_tokens = _tokenize(prompt) if query_vec is None else set()

        scored = []
        for i, entry in enumerate(self.entries):
            prompt_score = self._prompt_similarity(query_vec, query_tokens, i)
            if prompt_score <= 0:
                continue
            total_score = 0.65 * prompt_score + 0.35 * max(0.0, min(1.0, entry.score))
            scored.append((total_score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for score, entry in scored[:top_k]]

    @staticmethod
    def format_for_prompt(entries: list[MemoryEntry]) -> str:
        if not entries:
            return "No retrieved memory examples."
        blocks = []
        for index, entry in enumerate(entries, start=1):
            blocks.append(
                f"Example {index}\n"
                f"Prompt: {entry.prompt}\n"
                f"Score: {entry.score:.3f}\n"
                f"Feedback: {entry.feedback}\n"
                f"Program JSON:\n{json.dumps(entry.program, indent=2, sort_keys=True)}"
            )
        return "\n\n".join(blocks)
