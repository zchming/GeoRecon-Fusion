"""Similarity fitting for evaluation; no ground truth is used by inference."""
from dataclasses import dataclass
import numpy as np
from .camera import rotation


@dataclass
class Similarity:
    scale: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, points):
        return self.scale * np.asarray(points) @ self.R.T + self.t


def align_points(source, target, estimate_scale=True):
    """Umeyama least-squares source -> target. Reject <3 or collinear centers."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("alignment requires >=3 paired finite 3D points")
    a, b = source-source.mean(axis=0), target-target.mean(axis=0)
    if np.linalg.matrix_rank(a, tol=1e-10) < 2 or np.linalg.matrix_rank(b, tol=1e-10) < 2:
        raise ValueError("collinear or coincident alignment is underdetermined")
    u, singular, vh = np.linalg.svd(b.T @ a / len(a))
    signs = np.ones(3)
    signs[-1] = 1 if np.linalg.det(u @ vh) >= 0 else -1
    R = rotation(u @ np.diag(signs) @ vh)
    scale = float(np.dot(singular, signs) / np.mean(np.sum(a*a, axis=1))) if estimate_scale else 1.0
    if scale <= 1e-12:
        raise ValueError("degenerate fitted scale")
    return Similarity(scale, R, target.mean(axis=0)-scale*R @ source.mean(axis=0))
