from itertools import combinations
import numpy as np
from ..data.protocol import validate_result
from ..geometry.camera import Camera, relative_pose
from ..geometry.alignment import align_points


def rotation_error_deg(estimated, reference):
    return float(np.degrees(np.arccos(np.clip((np.trace(estimated @ reference.T)-1)/2, -1, 1))))


def translation_direction_error_deg(estimated, reference):
    a, b = np.linalg.norm(estimated), np.linalg.norm(reference)
    if a <= 1e-12 or b <= 1e-12:
        return None
    # Oriented translation: reversed direction is 180 degrees, not zero.
    return float(np.degrees(np.arccos(np.clip(np.dot(estimated, reference)/(a*b), -1, 1))))


def statistics(values):
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None}


def evaluate_poses(prediction, reference):
    validate_result(prediction)
    validate_result(reference)
    if prediction["role"] != "prediction" or reference["role"] != "reference":
        raise ValueError("prediction and reference roles must be explicit")
    for key in ("scene_id", "input_manifest_sha256"):
        if prediction[key] != reference[key]:
            raise ValueError(f"prediction/reference mismatch: {key}")
    if prediction["provenance"]["data_kind"] != reference["provenance"]["data_kind"]:
        raise ValueError("synthetic and real results must not be mixed")
    ids = [r["image_id"] for r in reference["images"]]
    if ids != [r["image_id"] for r in prediction["images"]]:
        raise ValueError("results must preserve identical input image IDs and order")
    estimated = {r["image_id"]: Camera.from_dict(r["camera"]) for r in prediction["images"] if r["registered"]}
    truth = {r["image_id"]: Camera.from_dict(r["camera"]) for r in reference["images"]}
    rotations, translations, pairs = [], [], []
    for first, second in combinations(ids, 2):
        pair = {"image_ids": [first, second], "rotation_deg": None, "translation_direction_deg": None}
        if first not in estimated or second not in estimated:
            pair.update(status="unavailable", reason="unregistered_image")
        else:
            R, t = relative_pose(estimated[first], estimated[second])
            Rgt, tgt = relative_pose(truth[first], truth[second])
            angle = rotation_error_deg(R, Rgt)
            direction = translation_direction_error_deg(t, tgt)
            rotations.append(angle)
            if direction is not None:
                translations.append(direction)
            pair.update(status="evaluated" if direction is not None else "rotation_only", reason=None if direction is not None else "zero_baseline", rotation_deg=angle, translation_direction_deg=direction)
        pairs.append(pair)
    ate = {"status": "unavailable", "reason": "requires >=3 noncollinear registered cameras", "unit": reference["world_unit"], "rmse": None}
    if len(estimated) >= 3:
        ordered = [i for i in ids if i in estimated]
        centers = np.stack([estimated[i].center for i in ordered])
        gt_centers = np.stack([truth[i].center for i in ordered])
        try:
            alignment = align_points(centers, gt_centers)
            errors = np.linalg.norm(alignment.apply(centers)-gt_centers, axis=1)
            ate.update(status="evaluated", reason=None, rmse=float(np.sqrt(np.mean(errors**2))),
                       image_ids=ordered, per_camera_error=errors.tolist(),
                       alignment={"scope": "evaluation_only_registered_subset", "type": "Sim3", "scale": alignment.scale, "R": alignment.R.tolist(), "t": alignment.t.tolist()})
        except ValueError as exc:
            ate["reason"] = str(exc)
    return {"schema_version": 1, "scene_id": prediction["scene_id"], "method_id": prediction["method_id"],
            "data_kind": reference["provenance"]["data_kind"], "run_status": prediction["status"],
            "failure_reason": prediction["failure_reason"], "input_images": len(ids),
            "registered_images": len(estimated), "registration_ratio": len(estimated)/len(ids),
            "input_pairs": len(pairs), "rotation_evaluated_pairs": len(rotations),
            "translation_evaluated_pairs": len(translations), "pairs": pairs,
            "rotation_deg_registered_subset": statistics(rotations),
            "translation_direction_deg_registered_subset": statistics(translations), "camera_center_ate": ate,
            "dense_geometry": {"status": "not_evaluated", "reason": "pose evaluation only; dense geometry is not evaluated"}}
