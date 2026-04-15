"""DETTM inference wrapper for the RefAV agentic pipeline.

Loads the checkpoint produced by train_dettm.py and exposes a
`DETTMMatcher` that scores/filters candidate track UUIDs against a
natural-language description.

Model architecture must match exactly what train_dettm.py trained:
  DETTM.text_enc  (CLIPTextEncoder with float16 safety fix)
  DETTM.track_enc (PatchTSTEncoder — hardcoded hyperparams)
  DETTM.track_proj, DETTM.Wq, DETTM.Wk
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters (must match train_dettm.py) ───────────────────────────────
_TRACK_DIM    = 10
_PATCH_LENGTH = 16
_PATCH_STRIDE = 8
_D_MODEL      = 256
_N_HEADS      = 8
_N_LAYERS     = 3
_EMBED_DIM    = 512
_CLIP_TOK_DIM = 768
_MAX_TRACK    = 300
_FEAT_COLS    = ['tx_m','ty_m','tz_m','qw','qx','qy','qz','length_m','width_m','height_m']


# ── Model definition (mirrors train_dettm.py exactly) ─────────────────────────
class _PatchTSTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_len = _PATCH_LENGTH
        self.stride    = _PATCH_STRIDE
        self.patch_embed = nn.Conv1d(_TRACK_DIM, _D_MODEL,
                                     kernel_size=_PATCH_LENGTH, stride=_PATCH_STRIDE)
        max_patches = (_MAX_TRACK - _PATCH_LENGTH) // _PATCH_STRIDE + 1
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, _D_MODEL) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=_D_MODEL, nhead=_N_HEADS,
            dim_feedforward=_D_MODEL*4, dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=_N_LAYERS)
        self.norm = nn.LayerNorm(_D_MODEL)

    def forward(self, x, padding_mask=None):
        x = self.patch_embed(x.transpose(1,2)).transpose(1,2)
        n = x.size(1)
        x = x + self.pos_embed[:, :n, :]
        pm = self._patch_mask(padding_mask, n) if padding_mask is not None else None
        return self.norm(self.transformer(x, src_key_padding_mask=pm)), pm

    def _patch_mask(self, tm, n):
        B, T = tm.shape
        masks = []
        for i in range(n):
            s, e = i*self.stride, i*self.stride+self.patch_len
            masks.append(tm[:,s:e].float().mean(1)>0.5 if e<=T
                         else torch.ones(B, dtype=torch.bool, device=tm.device))
        return torch.stack(masks, dim=1)


class _CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.mlp  = nn.Sequential(nn.Linear(_CLIP_TOK_DIM, _EMBED_DIM), nn.GELU(),
                                  nn.Linear(_EMBED_DIM, _EMBED_DIM))
        self.skip = nn.Linear(_CLIP_TOK_DIM, _EMBED_DIM)
        self.conv = nn.Conv1d(_EMBED_DIM, _EMBED_DIM, 3, padding=1)

    def forward(self, tokens):
        with torch.no_grad():
            dtype = next(self.clip_model.transformer.parameters()).dtype
            x = self.clip_model.token_embedding(tokens).to(dtype)
            x = x + self.clip_model.positional_embedding.to(dtype)
            x = self.clip_model.transformer(x.permute(1,0,2)).permute(1,0,2)
            x = self.clip_model.ln_final(x).float()
            gi = tokens.argmax(dim=-1)
            gf = x[torch.arange(x.size(0)), gi] @ self.clip_model.text_projection.float()
        tok = self.conv((self.mlp(x)+self.skip(x)).transpose(1,2)).transpose(1,2)
        return tok, self.mlp(gf)+self.skip(gf)


class _DETTM(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        import clip
        cm, _ = clip.load("ViT-L/14", device=device)
        self.text_enc   = _CLIPTextEncoder(cm)
        self.track_enc  = _PatchTSTEncoder()
        self.track_proj = nn.Linear(_D_MODEL, _EMBED_DIM)
        self.Wq = nn.Linear(_EMBED_DIM, _EMBED_DIM)
        self.Wk = nn.Linear(_EMBED_DIM, _EMBED_DIM)

    def forward(self, tracks, tokens, mask=None):
        patches, pm = self.track_enc(tracks, mask)
        patches = self.track_proj(patches)
        if pm is not None:
            v  = (~pm).unsqueeze(-1).float()
            tg = (patches*v).sum(1) / v.sum(1).clamp(min=1)
        else:
            tg = patches.mean(1)
        tok, txg = self.text_enc(tokens)
        tg  = F.normalize(tg,  dim=-1)
        txg = F.normalize(txg, dim=-1)
        Q, K = self.Wq(patches), self.Wk(tok)
        aln  = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(_EMBED_DIM)
        return tg, txg, aln, pm


# ── Public inference class ─────────────────────────────────────────────────────
class DETTMMatcher:
    """Loads a trained DETTM checkpoint and filters/scores candidate tracks.

    Usage::

        matcher = DETTMMatcher(Path("/path/to/dettm_checkpoint.pt"))
        scores  = matcher.score_tracks(scenario_dict, prompt, log_dir)
        filtered = matcher.filter_tracks(scenario_dict, prompt, log_dir)
    """

    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        import clip  # noqa: F401 — needed to resolve CLIP in _DETTM
        self.device = device
        # weights_only=False required because the checkpoint carries numpy
        # mean/std arrays alongside the state_dict. Safe here because we
        # produced the checkpoint ourselves via train_dettm.py.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        self.model = _DETTM(device).to(device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.mean: np.ndarray = ckpt['mean']
        self.std:  np.ndarray = ckpt['std']
        print(f"[dettm] matcher loaded from {checkpoint_path}")

    def _load_annotations(self, log_dir: Path) -> Optional[pd.DataFrame]:
        """Try annotations.feather (RefAV standard) then sm_annotations.feather."""
        for fname in ('annotations.feather', 'sm_annotations.feather'):
            p = log_dir / fname
            if p.exists():
                df = pd.read_feather(p)
                if all(c in df.columns for c in _FEAT_COLS):
                    return df
        return None

    def _build_batch(self, tracks_dict: dict, df: pd.DataFrame):
        """Build padded tensors for all tracks; return (uuids, tracks, masks)."""
        uuids, track_list, mask_list = [], [], []
        for uuid, timestamps in tracks_dict.items():
            tdf = df[(df['track_uuid'] == uuid) &
                     (df['timestamp_ns'].isin(timestamps))]
            if len(tdf) < _PATCH_LENGTH:
                continue
            feats = tdf[_FEAT_COLS].values.astype(np.float32)
            feats = (feats - self.mean) / self.std
            vl = min(len(feats), _MAX_TRACK)
            if len(feats) > _MAX_TRACK:
                feats = feats[:_MAX_TRACK]
            mask = np.zeros(_MAX_TRACK, dtype=bool)
            if len(feats) < _MAX_TRACK:
                feats = np.pad(feats, ((0, _MAX_TRACK - len(feats)), (0, 0)))
                mask[vl:] = True
            uuids.append(uuid)
            track_list.append(torch.tensor(feats, dtype=torch.float32))
            mask_list.append(torch.tensor(mask,  dtype=torch.bool))
        return uuids, track_list, mask_list

    @torch.no_grad()
    def score_tracks(self, tracks_dict: dict, description: str,
                     log_dir: Path, chunk_size: int = 32) -> dict[str, float]:
        """Score each track UUID against the description.

        Returns a dict mapping track_uuid → float score.
        Tracks shorter than PATCH_LENGTH or missing from the feather get 0.0.
        """
        import clip

        if not tracks_dict:
            return {}

        df = self._load_annotations(log_dir)
        if df is None:
            print(f"[dettm] no usable annotations in {log_dir}; skipping")
            return {u: 0.0 for u in tracks_dict}

        uuids, track_list, mask_list = self._build_batch(tracks_dict, df)

        # Tracks too short to patch get score 0
        scores: dict[str, float] = {u: 0.0 for u in tracks_dict}
        if not uuids:
            return scores

        # Process in chunks to avoid OOM with many candidates
        for start in range(0, len(uuids), chunk_size):
            batch_uuids  = uuids[start:start+chunk_size]
            tracks_batch = torch.stack(track_list[start:start+chunk_size]).to(self.device)
            masks_batch  = torch.stack(mask_list[start:start+chunk_size]).to(self.device)
            B = len(batch_uuids)

            tokens = clip.tokenize([description] * B, truncate=True).to(self.device)

            tg, txg, aln, pm = self.model(tracks_batch, tokens, masks_batch)

            # Global cosine similarity (InfoNCE-trained, in [-1, 1])
            global_sim = F.cosine_similarity(tg, txg)  # (B,)

            # MIL evidence: max alignment score across patches × tokens
            if pm is not None:
                aln = aln.masked_fill(pm.unsqueeze(-1).expand_as(aln), float('-inf'))
            mil_ev = aln.max(dim=-1)[0].max(dim=-1)[0]  # (B,)

            combined = (global_sim + mil_ev) / 2  # (B,)

            for i, uuid in enumerate(batch_uuids):
                scores[uuid] = combined[i].item()

        return scores

    def filter_tracks(self, tracks_dict: dict, description: str,
                      log_dir: Path, threshold: float = 0.0) -> dict:
        """Return the subset of tracks_dict whose score ≥ threshold.

        Falls back to the full tracks_dict if no track passes (avoids
        catastrophic precision drop from an under-calibrated threshold).
        """
        scores = self.score_tracks(tracks_dict, description, log_dir)
        filtered = {u: ts for u, ts in tracks_dict.items()
                    if scores.get(u, 0.0) >= threshold}
        if not filtered:
            # safe fallback — never drop everything
            return tracks_dict
        return filtered
