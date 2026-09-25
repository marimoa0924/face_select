"""얼굴 검출/임베딩(InsightFace)과 '내 얼굴' 판별 로직."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC 원본을 직접 받을 때만 필요
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}


@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]
    det_score: float
    embedding: np.ndarray  # L2 정규화된 512차원 벡터


def decode_image(data: bytes, max_side: int = 1600) -> np.ndarray:
    """바이트 → EXIF 회전 보정된 BGR ndarray (InsightFace 입력 형식)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return np.asarray(img)[:, :, ::-1].copy()


class FaceEngine:
    def __init__(self, model: str = "buffalo_l", det_size: int = 640, min_face_px: int = 40,
                 min_det_score: float = 0.5, use_gpu: bool = False):
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model, allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
        self.min_face_px = min_face_px
        self.min_det_score = min_det_score

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]:
        faces = []
        for f in self.app.get(bgr):
            x1, y1, x2, y2 = f.bbox
            # 너무 작거나 흐린 얼굴은 임베딩 품질이 낮아 오탐의 주원인이므로 제외
            if min(x2 - x1, y2 - y1) < self.min_face_px or f.det_score < self.min_det_score:
                continue
            faces.append(DetectedFace((x1, y1, x2, y2), float(f.det_score), f.normed_embedding.astype(np.float32)))
        return faces


class ReferenceSet:
    """내 얼굴 기준 임베딩 모음. 여러 장(정면/측면/안경/시기별)을 넣을수록 정확해진다."""

    def __init__(self, embeddings: np.ndarray):
        if len(embeddings) == 0:
            raise ValueError("기준 얼굴 임베딩이 없습니다. reference/ 폴더에 내 사진을 넣어주세요.")
        self.embeddings = embeddings  # (N, 512)
        centroid = embeddings.mean(axis=0)
        self.centroid = centroid / np.linalg.norm(centroid)

    @classmethod
    def from_folder(cls, engine: FaceEngine, folder: str | Path) -> "ReferenceSet":
        embs, skipped = [], []
        for path in sorted(Path(folder).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            faces = engine.detect(decode_image(path.read_bytes()))
            if not faces:
                skipped.append(path.name)
                continue
            # 기준 사진에는 내 얼굴이 가장 크게 나왔다고 가정
            biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embs.append(biggest.embedding)
        if skipped:
            print(f"[경고] 얼굴을 찾지 못한 기준 사진: {', '.join(skipped)}")
        return cls(np.stack(embs) if embs else np.empty((0, 512), np.float32))

    def save(self, path: str | Path) -> None:
        np.save(path, self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceSet":
        return cls(np.load(path))

    def score(self, emb: np.ndarray, top_k: int = 3) -> float:
        """기준 임베딩들과의 코사인 유사도 중 상위 k개 평균과 중심점 유사도 중 큰 값."""
        sims = self.embeddings @ emb
        k = min(top_k, len(sims))
        top = float(np.sort(sims)[-k:].mean())
        return max(top, float(self.centroid @ emb))

    def best_match(self, faces: list[DetectedFace]) -> float:
        return max((self.score(f.embedding) for f in faces), default=0.0)
