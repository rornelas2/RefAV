"""Vendored from SMc2f/clip_coarse_filter.py.

Behavior matches upstream exactly. Only differences:
  * `SMc2fConfig` imported from the sibling config module
  * spaCy import is wrapped so it is a soft optional dependency
  * `extract_and_save_log_features` accepts an explicit `av2_root` and
    resolves camera directories from `<av2_root>/<split>/<log_id>/sensors/cameras/<cam>`
    which matches the upstream layout.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import clip
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras

from .config import SMc2fConfig

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover - optional dep
    spacy = None  # type: ignore


class CLIPFeatureExtractor:
    """Offline feature extraction - run once per dataset."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(SMc2fConfig.CLIP_MODEL, device=device)
        self.model.eval()

    def extract_and_save_log_features(self, log_id: str, split: str, av2_root: Path) -> None:
        """Extract and save features for all frames in a log.

        Idempotent: if `meta.json` already exists in the cache dir for this
        log, returns immediately.
        """
        av2_root = Path(av2_root)
        cache_dir = SMc2fConfig.CLIP_CACHE_DIR / split / log_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        meta_file = cache_dir / "meta.json"
        if meta_file.exists():
            return  # already extracted

        data_loader = AV2SensorDataLoader(
            data_dir=av2_root / split,
            labels_dir=av2_root / split,
        )

        cam_timestamps: dict[str, list[int]] = {}
        for cam in RingCameras:
            cam_name = cam.value
            cam_dir = av2_root / split / log_id / "sensors" / "cameras" / cam_name
            if cam_dir.exists():
                ts_list = sorted(int(p.stem) for p in cam_dir.glob("*.jpg"))
                cam_timestamps[cam_name] = ts_list

        for cam_name, timestamps in cam_timestamps.items():
            print(f"  Extracting {cam_name}: {len(timestamps)} frames")

            features: list[np.ndarray] = []
            valid_ts: list[int] = []

            for ts in tqdm(timestamps, desc=cam_name):
                try:
                    img_path = data_loader.get_closest_img_fpath(log_id, cam_name, ts)
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        feat = self.model.encode_image(img_tensor)
                        feat = feat / feat.norm(dim=-1, keepdim=True)

                    features.append(feat.cpu().numpy())
                    valid_ts.append(ts)
                except Exception:
                    continue

            if features:
                features_np = np.vstack(features)
                np.savez_compressed(
                    cache_dir / f"{cam_name}.npz",
                    features=features_np,
                    timestamps=np.array(valid_ts),
                )

        with open(meta_file, "w") as f:
            json.dump(
                {
                    "log_id": log_id,
                    "cameras": list(cam_timestamps.keys()),
                    "fps": SMc2fConfig.CAMERA_FPS,
                },
                f,
            )

    def extract_dataset(self, split: str, av2_root: Path, log_ids: list[str]) -> None:
        """Extract features for an entire list of logs."""
        for i, log_id in enumerate(log_ids):
            print(f"[{i + 1}/{len(log_ids)}] Processing {log_id}")
            self.extract_and_save_log_features(log_id, split, av2_root)


class CLIPCoarseFilter:
    """Query-time coarse filtering using precomputed features."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, _ = clip.load(SMc2fConfig.CLIP_MODEL, device=device)
        self.model.eval()

        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

    def _load_precomputed_features(self, log_id: str, split: str) -> dict:
        cache_dir = SMc2fConfig.CLIP_CACHE_DIR / split / log_id

        if not cache_dir.exists():
            raise FileNotFoundError(f"Precomputed features not found for {log_id}: {cache_dir}")

        features: dict[str, dict] = {}
        for cam in RingCameras:
            cam_name = cam.value
            npz_path = cache_dir / f"{cam_name}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                features[cam_name] = {
                    "features": torch.from_numpy(data["features"]),
                    "timestamps": data["timestamps"].tolist(),
                }
        return features

    def _extract_keywords(self, query: str) -> str:
        if self.nlp is None:
            return query

        doc = self.nlp(query)
        keywords: list[str] = []

        for token in doc:
            if token.pos_ in ("NOUN", "ADJ", "PROPN"):
                keywords.append(token.text)
            if token.dep_ in ("prep", "advmod", "amod"):
                keywords.append(token.text)

        for ent in doc.ents:
            keywords.append(ent.text)

        return ", ".join(list(set(keywords))) if keywords else query

    def _encode_text(self, query: str) -> torch.Tensor:
        keywords = self._extract_keywords(query)
        tokens = clip.tokenize([keywords], truncate=True).to(self.device)

        with torch.no_grad():
            text_feat = self.model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        return text_feat.cpu()

    def filter_temporal_windows(self, log_id: str, split: str):
        cam_features = self._load_precomputed_features(log_id, split)

        if not cam_features:
            return []

        all_timestamps: set[int] = set()
        for cam_data in cam_features.values():
            all_timestamps.update(cam_data["timestamps"])
        all_timestamps_sorted = sorted(all_timestamps)

        if len(all_timestamps_sorted) < 2:
            return (
                [(all_timestamps_sorted[0], all_timestamps_sorted[-1])]
                if all_timestamps_sorted
                else []
            )

        fps = SMc2fConfig.CAMERA_FPS
        window_frames = int(SMc2fConfig.WINDOW_LENGTH_S * fps)
        stride_frames = int(SMc2fConfig.WINDOW_STRIDE_S * fps)

        return all_timestamps_sorted, window_frames, stride_frames, cam_features

    def get_filtered_segments(self, query: str, log_id: str, split: str) -> list[tuple[int, int]]:
        """Return list of (start_ts, end_ts) tuples representing continuous segments."""
        result = self.filter_temporal_windows(log_id, split)
        if not result:
            return []

        all_timestamps, window_frames, stride_frames, cam_features = result

        text_feat = self._encode_text(query)

        windows: list[dict] = []
        n_frames = len(all_timestamps)

        for start_idx in range(0, max(1, n_frames - window_frames + 1), stride_frames):
            end_idx = min(start_idx + window_frames, n_frames)
            window_ts = all_timestamps[start_idx:end_idx]

            if len(window_ts) >= SMc2fConfig.FRAMES_PER_WINDOW:
                sample_indices = np.linspace(
                    0, len(window_ts) - 1, SMc2fConfig.FRAMES_PER_WINDOW, dtype=int
                )
                sampled_ts = [window_ts[i] for i in sample_indices]
            else:
                sampled_ts = window_ts

            cam_similarities: list[float] = []
            for cam_data in cam_features.values():
                cam_ts = cam_data["timestamps"]
                cam_feats = cam_data["features"]

                frame_sims: list[float] = []
                for ts in sampled_ts:
                    if ts in cam_ts:
                        idx = cam_ts.index(ts)
                        sim = torch.cosine_similarity(
                            text_feat, cam_feats[idx : idx + 1], dim=-1
                        ).item()
                        frame_sims.append(sim)

                if frame_sims:
                    cam_similarities.append(float(np.mean(frame_sims)))

            if cam_similarities:
                window_score = float(np.sum(cam_similarities))
                windows.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "start_ts": window_ts[0],
                        "end_ts": window_ts[-1],
                        "score": window_score,
                    }
                )

        if not windows:
            return [(all_timestamps[0], all_timestamps[-1])]

        windows.sort(key=lambda x: x["score"], reverse=True)
        top_windows = windows[: SMc2fConfig.TOP_K_WINDOWS]

        return self._merge_adjacent_windows(top_windows, all_timestamps)

    @staticmethod
    def _merge_adjacent_windows(windows: list[dict], all_timestamps: list[int]) -> list[tuple[int, int]]:
        if not windows:
            return []

        windows = sorted(windows, key=lambda x: x["start_idx"])

        segments: list[tuple[int, int]] = []
        current_start = windows[0]["start_idx"]
        current_end = windows[0]["end_idx"]

        for w in windows[1:]:
            if w["start_idx"] <= current_end + 1:
                current_end = max(current_end, w["end_idx"])
            else:
                segments.append(
                    (
                        all_timestamps[current_start],
                        all_timestamps[min(current_end, len(all_timestamps) - 1)],
                    )
                )
                current_start = w["start_idx"]
                current_end = w["end_idx"]

        segments.append(
            (
                all_timestamps[current_start],
                all_timestamps[min(current_end, len(all_timestamps) - 1)],
            )
        )

        return segments

    @staticmethod
    def filter_annotations(annotations_df, segments):
        """Filter a pandas annotations dataframe to rows inside any retained segment."""
        if not segments:
            return annotations_df

        mask = None
        for start_ts, end_ts in segments:
            segment_mask = (annotations_df["timestamp_ns"] >= start_ts) & (
                annotations_df["timestamp_ns"] <= end_ts
            )
            mask = segment_mask if mask is None else (mask | segment_mask)

        return annotations_df[mask]
