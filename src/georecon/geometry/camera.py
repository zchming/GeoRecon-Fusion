"""OpenCV axes (right, down, forward), world-to-camera, camera Z-depth."""
from dataclasses import dataclass
import numpy as np


def finite_array(value, shape, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def rotation(value):
    matrix = finite_array(value, (3, 3), "R")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        raise ValueError("R must be a proper rotation, not a reflection")
    return matrix


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        self.K = finite_array(self.K, (3, 3), "K")
        self.R = rotation(self.R)
        self.t = finite_array(self.t, (3,), "t")
        if not np.allclose(self.K[2], [0, 0, 1], atol=1e-10, rtol=0) or self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("K must have positive focal lengths and bottom row [0,0,1]")
        if not np.isclose(self.K[1, 0], 0, atol=1e-10):
            raise ValueError("K must be upper triangular")
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("image dimensions must be positive integers")

    @property
    def center(self):
        return -self.R.T @ self.t

    @property
    def projection(self):
        return self.K @ np.column_stack((self.R, self.t))

    def world_to_camera(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be Nx3")
        return points @ self.R.T + self.t

    def project(self, points):
        """Return pixels, Z, and positive finite depth mask; does not clip image bounds."""
        camera_points = self.world_to_camera(points)
        z = camera_points[:, 2]
        valid = np.isfinite(camera_points).all(axis=1) & (z > 1e-12)
        pixels = np.full((len(z), 2), np.nan)
        homogeneous = camera_points[valid] @ self.K.T
        pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
        return pixels, z, valid

    def unproject(self, pixels, depth):
        pixels = np.asarray(pixels, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),):
            raise ValueError("pixels must be Nx2 and Z-depth must be N")
        if not np.isfinite(pixels).all() or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("unprojection requires finite pixels and positive Z-depth")
        rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(self.K).T
        return (rays * depth[:, None] - self.t) @ self.R

    def as_dict(self):
        return {"model": "PINHOLE", "K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist(),
                "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, value):
        if value.get("model") != "PINHOLE":
            raise ValueError("stage 1 only supports explicit undistorted PINHOLE cameras")
        return cls(**{key: value[key] for key in ("K", "R", "t", "width", "height")})


def resize_crop(camera, resized_wh, crop_xywh=None):
    """Pixel-center mapping: u' = sx*(u+.5)-.5-left (Pillow resize then crop)."""
    rw, rh = resized_wh
    if any(type(v) is not int or v <= 0 for v in (rw, rh)):
        raise ValueError("resize dimensions must be positive integers")
    left, top, width, height = crop_xywh or (0, 0, rw, rh)
    if any(type(v) is not int for v in (left, top, width, height)) or min(left, top) < 0 or min(width, height) <= 0 or left + width > rw or top + height > rh:
        raise ValueError("crop must be an integer rectangle inside resized image")
    sx, sy = rw / camera.width, rh / camera.height
    transform = np.array([[sx, 0, (sx-1)/2-left], [0, sy, (sy-1)/2-top], [0, 0, 1]])
    return Camera(transform @ camera.K, camera.R, camera.t, width, height), transform


def relative_pose(first, second):
    relative_R = second.R @ first.R.T
    return relative_R, second.t - relative_R @ first.t


def decompose_projection(projection, width, height):
    """RQ factorization of an undistorted 3x4 projection matrix, including its sign."""
    P = finite_array(projection, (3, 4), "projection")
    if abs(np.linalg.det(P[:, :3])) < 1e-12:
        raise ValueError("singular projection matrix")
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    q, r = np.linalg.qr(np.flipud(P[:, :3]).T)
    K = np.fliplr(np.flipud(r.T))
    R = np.flipud(q.T)
    signs = np.diag(np.where(np.diag(K) < 0, -1., 1.))
    K, R = K @ signs, signs @ R
    scale = K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    return Camera(K/scale, R, t, width, height)


def triangulate(first, second, first_pixels, second_pixels, min_angle_deg=0.1):
    """Linear two-view DLT for mathematical validation, not a teaching SfM pipeline."""
    p = np.asarray(first_pixels, dtype=float)
    q = np.asarray(second_pixels, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or q.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("correspondences must be equal finite Nx2 arrays")
    if not 0 < min_angle_deg < 180:
        raise ValueError("min_angle_deg must be in (0,180)")
    points = np.full((len(p), 3), np.nan)
    valid = np.zeros(len(p), dtype=bool)
    if np.linalg.norm(first.center-second.center) < 1e-12:
        return points, valid
    a, b = first.projection, second.projection
    for i, ((u, v), (x, y)) in enumerate(zip(p, q)):
        design = np.stack([u*a[2]-a[0], v*a[2]-a[1], x*b[2]-b[0], y*b[2]-b[1]])
        _, _, vh = np.linalg.svd(design)
        if abs(vh[-1, 3]) <= 1e-12:
            continue
        point = vh[-1, :3] / vh[-1, 3]
        ray1, ray2 = point-first.center, point-second.center
        angle = np.degrees(np.arccos(np.clip(np.dot(ray1, ray2)/(np.linalg.norm(ray1)*np.linalg.norm(ray2)), -1, 1)))
        if first.world_to_camera(point[None])[0, 2] > 0 and second.world_to_camera(point[None])[0, 2] > 0 and angle >= min_angle_deg:
            points[i], valid[i] = point, True
    return points, valid
