"""Configuration for the vendored SMc2f modules.

Stage 1 only exercises the CLIP coarse filter fields. DETTM-related
fields are kept as-is (unused) so that if we later vendor DETTM the code
does not diverge from upstream.
"""

from pathlib import Path


# Resolve a writable scratch root. This file is vendored inside the RefAV
# package so we do not want to hardcode absolute paths that are specific to
# one machine — but the upstream config did, so we anchor to the user's
# scratch directory when it exists and fall back to the repo-local cache dir.
def _default_smc2f_cache_root() -> Path:
    scratch = Path.home() / "scratch" / "refav_output" / "smc2f_cache"
    if scratch.parent.exists():
        return scratch
    return Path.home() / ".cache" / "refav" / "smc2f"


_SMC2F_ROOT = _default_smc2f_cache_root()


class SMc2fConfig:
    # ---- CLIP coarse filter ----
    CLIP_MODEL = "ViT-L/14"
    CAMERA_FPS = 20  # AV2 ring cameras are 20Hz
    WINDOW_LENGTH_S = 3
    WINDOW_STRIDE_S = 1
    FRAMES_PER_WINDOW = 5
    TOP_K_WINDOWS = 5

    # Precomputed CLIP image features live here as <split>/<log_id>/<cam>.npz
    CLIP_CACHE_DIR = _SMC2F_ROOT / "clip_features"

    # ---- Knowledge base (DETTM-only, unused in Stage 1) ----
    KB_PATH = _SMC2F_ROOT / "kb_database.json"
    SBERT_MODEL = "all-MiniLM-L6-v2"
    KB_RETRIEVAL_K = 10

    # ---- DETTM placeholders (unused in Stage 1) ----
    TRACK_DIM = 10
    PATCH_LENGTH = 16
    PATCH_STRIDE = 8
    PATCHTST_D_MODEL = 256
    PATCHTST_N_HEADS = 8
    PATCHTST_N_LAYERS = 3
    EMBED_DIM = 512
    CLIP_TOKEN_DIM = 768
    MAX_TEXT_TOKENS = 77
    MAX_TRACK_LEN = 300

    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 5
    LAMBDA_MIL = 1.0
    LAMBDA_GLOBAL = 1.0
    GAMMA = 0.1
    TAU = 0.07

    DETTM_CHECKPOINT_DIR = _SMC2F_ROOT / "dettm_checkpoints"
    PATCHTST_PRETRAINED_PATH = _SMC2F_ROOT / "pt" / "patchtst_pretrained.pt"
    TRACK_NORM_STATS_PATH = _SMC2F_ROOT / "track_norm_stats.npz"
