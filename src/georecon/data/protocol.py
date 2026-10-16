"""Versioned JSON protocols; missing geometry is explicit, never fabricated."""
import hashlib
import json
import math
from pathlib import Path
from ..geometry.camera import Camera

CONVENTION = "opencv_world_to_camera_z_depth"
STATUSES = {"success", "partial", "no_reconstruction", "timeout", "oom", "process_error", "not_run"}


def read_json(path):
    def reject(value):
        raise ValueError(f"non-standard JSON numeric constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name:
        raise ValueError("artifact paths must be nonempty relative paths without '..'")
    root = Path(root).resolve()
    resolved = (root/path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path escapes scene root")
    return resolved


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("identifiers must be nonempty strings")


def validate_manifest(value):
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    if value["schema_version"] != 1 or value["kind"] not in {"synthetic", "real"} or value["split"] not in {"dev", "test", "smoke"}:
        raise ValueError("unsupported manifest version, kind or split")
    _identifier(value["scene_id"])
    if type(value["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for key in ("source", "license_note", "preprocessing", "budget"):
        if key not in value:
            raise ValueError(f"manifest missing {key}")
    preprocessing = value["preprocessing"]
    if preprocessing.get("operation") not in {"identity", "resize_then_crop"}:
        raise ValueError("unsupported preprocessing operation")
    if preprocessing["operation"] == "resize_then_crop":
        size, crop = preprocessing["resize_wh"], preprocessing["crop_xywh"]
        if len(size) != 2 or len(crop) != 4 or any(type(v) is not int for v in [*size, *crop]):
            raise ValueError("resize/crop must contain integer dimensions")
        if min(size) <= 0 or min(crop[:2]) < 0 or min(crop[2:]) <= 0 or crop[0]+crop[2] > size[0] or crop[1]+crop[3] > size[1]:
            raise ValueError("invalid resize/crop dimensions")
    if not value["images"]:
        raise ValueError("scene must contain images")
    ids = []
    for image in value["images"]:
        _identifier(image["image_id"])
        ids.append(image["image_id"])
        relative_file(Path.cwd(), image["path"])
        if not isinstance(image["sha256"], str) or len(image["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in image["sha256"]):
            raise ValueError("image SHA-256 must be 64 lowercase hex characters")
        if any(type(image[k]) is not int or image[k] <= 0 for k in ("width", "height")):
            raise ValueError("invalid image dimensions")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate image IDs")
    return value


def validate_result(value):
    if not isinstance(value, dict):
        raise ValueError("result must be a JSON object")
    if value["schema_version"] != 1 or value["role"] not in {"prediction", "reference"}:
        raise ValueError("unsupported result version or role")
    if value["coordinate_convention"] != CONVENTION or value["world_unit"] not in {"m", "mm", "arbitrary"}:
        raise ValueError("unsupported coordinate convention or unit")
    for key in ("scene_id", "method_id", "run_id"):
        _identifier(value[key])
    fingerprint = value["input_manifest_sha256"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("result requires input manifest SHA-256")
    if value["status"] not in STATUSES:
        raise ValueError("unknown run status")
    if value["status"] not in {"success"} and not value.get("failure_reason"):
        raise ValueError("non-success runs require a reason")
    if not isinstance(value["provenance"], dict) or not value["provenance"]:
        raise ValueError("result requires provenance")
    if value["provenance"].get("data_kind") not in {"synthetic", "real"}:
        raise ValueError("result must distinguish synthetic and real data")
    for key in ("config", "versions", "timing_seconds", "memory_bytes", "logs"):
        if key not in value:
            raise ValueError(f"result missing {key}; use null explicitly when unmeasured")
    if not isinstance(value["config"], dict) or not isinstance(value["versions"], dict) or not isinstance(value["logs"], list):
        raise ValueError("config/versions must be objects, logs must be a list")
    for group in ("timing_seconds", "memory_bytes"):
        if not isinstance(value[group], dict):
            raise ValueError(f"{group} must be an object")
        for number in value[group].values():
            if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0):
                raise ValueError(f"{group} values must be nonnegative finite numbers or null")
    ids, registered = [], 0
    if not value["images"]:
        raise ValueError("result requires explicit image registration states")
    for record in value["images"]:
        _identifier(record["image_id"])
        ids.append(record["image_id"])
        if type(record["registered"]) is not bool:
            raise ValueError("registered must be boolean")
        if record["registered"]:
            Camera.from_dict(record["camera"])
            registered += 1
        elif record.get("camera") is not None or not record.get("reason"):
            raise ValueError("unregistered images need null camera and a reason")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate image IDs")
    if value["status"] == "success" and registered != len(ids):
        raise ValueError("success requires all input images registered")
    if value["status"] == "partial" and not 2 <= registered < len(ids):
        raise ValueError("partial requires >=2 but fewer than all cameras")
    if value["status"] in {"not_run", "no_reconstruction"} and registered:
        raise ValueError("not_run/no_reconstruction cannot claim registered cameras")
    if value["role"] == "reference" and value["status"] != "success":
        raise ValueError("evaluation references must provide all cameras")
    for key in ("depth", "pointcloud", "tracks", "confidence"):
        artifact = value["artifacts"][key]
        if artifact is not None:
            if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("semantics"):
                raise ValueError("optional artifact needs relative path and semantics")
            relative_file(Path.cwd(), artifact["path"])
    return value


def result_template(scene_id, manifest_hash, image_ids, method_id, role="prediction", kind="synthetic"):
    return {"schema_version": 1, "role": role, "scene_id": scene_id, "method_id": method_id,
            "run_id": method_id, "input_manifest_sha256": manifest_hash,
            "coordinate_convention": CONVENTION, "world_unit": "arbitrary",
            "status": "not_run", "failure_reason": "method has not been run",
            "images": [{"image_id": i, "registered": False, "camera": None, "reason": "not_run"} for i in image_ids],
            "artifacts": {k: None for k in ("depth", "pointcloud", "tracks", "confidence")},
            "provenance": {"data_kind": kind}, "config": {}, "versions": {},
            "timing_seconds": {"total": None}, "memory_bytes": {"peak_gpu_allocated": None}, "logs": []}
