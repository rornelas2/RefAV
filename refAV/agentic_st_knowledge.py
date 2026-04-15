from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Optional

import numpy as np

from refAV.agentic_st_ops import infer_category_mentions


# ---------------------------------------------------------------------------
# Tokenize / Jaccard (fallback)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ---------------------------------------------------------------------------
# Reuse the singleton from agentic_st_memory
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str], model_name: str) -> Optional[np.ndarray]:
    try:
        from refAV.agentic_st_memory import _embed_texts as _mem_embed
        return _mem_embed(texts, model_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeBaseEntry:
    prompt: str
    program: dict
    notes: str = ""
    tags: list[str] | None = None
    score_hint: float = 1.0


# ---------------------------------------------------------------------------
# FewShotKnowledgeBase
# ---------------------------------------------------------------------------

class FewShotKnowledgeBase:
    def __init__(self, path: Optional[Path], embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.path = Path(path) if path is not None else None
        self.embedding_model_name = embedding_model_name
        self.entries: list[KnowledgeBaseEntry] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, D) once built

        if self.path is None or not self.path.exists():
            return

        with open(self.path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.entries.append(KnowledgeBaseEntry(**data))

        # Pre-compute embeddings for all entries at load time
        if self.entries:
            texts = [e.prompt for e in self.entries]
            self._embeddings = _embed_texts(texts, self.embedding_model_name)
            if self._embeddings is not None:
                print(f"[knowledge] built embeddings for {len(self.entries)} KB entries")

    # ------------------------------------------------------------------

    def retrieve(self, prompt: str, top_k: int = 4) -> list[KnowledgeBaseEntry]:
        if not self.entries:
            return []

        prompt_categories = set(infer_category_mentions(prompt))

        # Prompt similarity: embedding cosine if available, else Jaccard
        use_embed = self._embeddings is not None
        query_vec: Optional[np.ndarray] = None
        if use_embed:
            q = _embed_texts([prompt], self.embedding_model_name)
            query_vec = q[0] if q is not None else None

        prompt_tokens = _tokenize(prompt) if query_vec is None else set()

        scored: list[tuple[float, KnowledgeBaseEntry]] = []
        for i, entry in enumerate(self.entries):
            # Prompt similarity
            if query_vec is not None:
                prompt_score = float(np.dot(query_vec, self._embeddings[i]))
            else:
                prompt_score = _jaccard(prompt_tokens, _tokenize(entry.prompt))

            # Category overlap (kept as-is; embeddings don't capture AV categories well)
            entry_categories = set(infer_category_mentions(entry.prompt)) | set(entry.tags or [])
            category_score = _jaccard(prompt_categories, entry_categories)

            # Tag token overlap
            tag_tokens: set[str] = set()
            if entry.tags:
                for tag in entry.tags:
                    tag_tokens.update(_tokenize(tag))
            tag_score = _jaccard(_tokenize(prompt), tag_tokens) if tag_tokens else 0.0

            total_score = 0.55 * prompt_score + 0.25 * category_score + 0.20 * tag_score
            scored.append((total_score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for score, entry in scored[:top_k] if score > 0]

    @staticmethod
    def format_for_prompt(entries: list[KnowledgeBaseEntry]) -> str:
        if not entries:
            return "No few-shot knowledge base examples retrieved."
        blocks = []
        for index, entry in enumerate(entries, start=1):
            tags = ", ".join(entry.tags or []) or "none"
            blocks.append(
                f"Example {index}\n"
                f"Prompt: {entry.prompt}\n"
                f"Tags: {tags}\n"
                f"Notes: {entry.notes or 'none'}\n"
                f"Program JSON:\n{json.dumps(entry.program, indent=2, sort_keys=True)}"
            )
        return "\n\n".join(blocks)
