"""Team assignment: SigLIP crop embeddings -> UMAP(3) -> KMeans(2).

Why not ReID embeddings? ReID is trained to separate *individuals*; for teams we
want the opposite, a feature where two teammates look alike. A general
image-text encoder on the torso crop clusters by jersey colour/pattern.
Referees are excluded by class before clustering.
"""
from __future__ import annotations

import numpy as np
import torch

from hoops.device import pick_device
from hoops.reid.base import crop_boxes


class SigLIPEmbedder:
    def __init__(self, model: str = "google/siglip-base-patch16-224", device: str | None = None, batch: int = 32):
        from transformers import AutoModel, AutoProcessor

        self.device = pick_device(device)
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.batch = batch

    @torch.no_grad()
    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        feats = []
        for i in range(0, len(crops_bgr), self.batch):
            chunk = [c[:, :, ::-1] for c in crops_bgr[i : i + self.batch]]  # RGB
            inputs = self.proc(images=chunk, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**inputs)
            feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)


def torso_boxes(xyxy: np.ndarray) -> np.ndarray:
    """Central part of each player box (jersey, not legs/head/background)."""
    b = np.asarray(xyxy, float)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.stack([b[:, 0] + 0.2 * w, b[:, 1] + 0.15 * h, b[:, 2] - 0.2 * w, b[:, 1] + 0.55 * h], 1)


class TeamClassifier:
    def __init__(self, embedder: SigLIPEmbedder | None = None, device: str | None = None, n_teams: int = 2):
        self.embedder = embedder or SigLIPEmbedder(device=device)
        self.n_teams = n_teams
        self.reducer = None
        self.kmeans = None

    def fit(self, crops: list[np.ndarray]) -> TeamClassifier:
        import umap
        from sklearn.cluster import KMeans

        X = self.embedder(crops)
        self.reducer = umap.UMAP(n_components=3, random_state=0)
        Z = self.reducer.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=0).fit(Z)
        return self

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        X = self.embedder(crops)
        return self.kmeans.predict(self.reducer.transform(X))


def assign_teams_to_tracks(
    frames_iter,
    track_boxes: dict[int, list[tuple[int, np.ndarray]]],
    classifier: TeamClassifier,
    sample_every: int = 30,
    max_crops: int = 400,
) -> dict[int, int]:
    """Fit on a sample of crops across the clip, then majority-vote per track id.

    frames_iter: iterable of (frame_idx, image); track_boxes: id -> [(frame_idx, xyxy)].
    """
    wanted: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tid, lst in track_boxes.items():
        for f, b in lst:
            if f % sample_every == 0:
                wanted.setdefault(f, []).append((tid, b))
    crops, owners = [], []
    for f, img in frames_iter:
        if f not in wanted:
            continue
        for tid, b in wanted[f]:
            crops.append(crop_boxes(img, torso_boxes(b[None]))[0])
            owners.append(tid)
        if len(crops) >= max_crops:
            break
    if len(crops) < 2 * classifier.n_teams:
        return {}
    classifier.fit(crops)
    labels = classifier.predict(crops)
    votes: dict[int, list[int]] = {}
    for tid, lab in zip(owners, labels):
        votes.setdefault(tid, []).append(int(lab))
    return {tid: int(np.bincount(v).argmax()) for tid, v in votes.items()}
