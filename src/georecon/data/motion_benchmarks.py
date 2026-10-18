"""Reference-format adapters; these are never imported by reconstruction."""
import numpy as np
from ..geometry.camera import Camera


def quaternion_matrix(xyzw):
    q = np.asarray(xyzw, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError('approximately unit finite quaternion required')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tum_reference(text, timestamp, k, maximum_gap=.02):
    rows = np.array([[float(v) for v in line.split()] for line in text.splitlines()
                     if line.strip() and not line.lstrip().startswith('#')], float)
    if rows.ndim != 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError('TUM timestamp tx ty tz qx qy qz qw required')
    row = rows[np.argmin(abs(rows[:, 0]-timestamp))]
    gap = abs(float(row[0])-timestamp)
    if gap > maximum_gap:
        return None, gap
    r = quaternion_matrix(row[4:]).T
    return Camera(np.array(k), r, -r@row[1:4], 640, 480), gap


def colmap_references(camera_text, image_text):
    cameras = {}
    for line in camera_text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        v = line.split()
        if v[1] != 'SIMPLE_RADIAL' or len(v) != 8:
            raise ValueError('frozen South Building SIMPLE_RADIAL model required')
        f, cx, cy, radial = map(float, v[4:])
        cameras[int(v[0])] = (np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]]),
                              np.array([radial, 0, 0, 0, 0]), int(v[2]), int(v[3]))
    lines = iter(image_text.splitlines()); result = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        v = line.split()
        if len(v) != 10:
            raise ValueError('COLMAP two-line image record required')
        qw, qx, qy, qz, tx, ty, tz = map(float, v[1:8])
        k, distortion, w, h = cameras[int(v[8])]
        r = quaternion_matrix([qx, qy, qz, qw])
        result[v[9]] = (Camera(k, r, np.array([tx, ty, tz]), w, h), distortion)
        next(lines)  # image observations, deliberately not used as features or GT tracks
    return result
