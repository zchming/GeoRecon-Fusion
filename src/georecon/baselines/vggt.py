"""Official VGGT raw-forward adapter. CPU-safe imports; no GT or BA dependency."""
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from .colmap import execute_worker, pixel_transform
from ..data.protocol import read_json, write_json, sha256, result_template, validate_result
from ..data.scene import load_scene
from ..geometry.camera import Camera
from ..provenance import code_provenance


def crop_geometry(width, height):
    """Pinned official crop: resize width to 518, round height to 14, crop height."""
    if min(width, height) <= 0:
        raise ValueError("positive input dimensions required")
    new_width = 518
    new_height = round(height*(new_width/width)/14)*14
    if new_height <= 0:
        raise ValueError("official crop produces zero height for this aspect ratio")
    top = (new_height-518)//2 if new_height > 518 else 0
    out_height = min(new_height, 518)
    sx, sy = new_width/width, new_height/height
    A = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2-top], [0, 0, 1.]])
    return {"resize_wh": [new_width, new_height], "crop_xywh": [0, top, new_width, out_height], "pixel_transform": A.tolist()}


def independent_crop(image):
    """Audit implementation, not the inference loader; compare against official pixels."""
    geometry = crop_geometry(image.width, image.height)
    resized = image.convert("RGB").resize(geometry["resize_wh"], Image.Resampling.BICUBIC)
    x, y, w, h = geometry["crop_xywh"]
    return np.asarray(resized.crop((x, y, x+w, y+h)), dtype=np.float32).transpose(2, 0, 1)/255


def convert_camera(K_native, extrinsic, original_wh, transform):
    # Official geometry.py uses np.arange pixel centers, not COLMAP's corner origin.
    return Camera(np.linalg.solve(transform, K_native), extrinsic[:, :3], extrinsic[:, 3], *original_wh)


def validate_config(config):
    if config["schema_version"] != 1 or config["official_commit"] != "a288dd0f14786c93483e45524328726ab7b1b4ce":
        raise ValueError("unsupported VGGT source version")
    if config["preprocessing"] != "official_crop_after_common_manifest_preprocessing" or config["device"] != "cuda" or config["autocast"] != "bfloat16" or config["parameter_dtype"] != "float32" or config["query_points"] is not None:
        raise ValueError("only pinned raw CUDA BF16 configuration without queries is supported")
    if config["postprocessing"] != "none; no BA, filtering, sampling or fusion":
        raise ValueError("this adapter only supports raw inference")
    if len(config["checkpoint_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in config["checkpoint_sha256"]):
        raise ValueError("checkpoint requires a SHA-256 pin")
    if type(config["seed"]) is not int or type(config["num_threads"]) is not int or config["num_threads"] <= 0 or not np.isfinite(config["timeout_seconds"]) or config["timeout_seconds"] <= 0:
        raise ValueError("invalid seed, thread or timeout configuration")
    return config


def verify_repository(repo, expected_commit):
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if git("rev-parse", "HEAD") != expected_commit or git("status", "--porcelain"):
        raise ValueError("official repository must be at pinned commit with no local changes")
    files = ["vggt/models/vggt.py", "vggt/models/aggregator.py", "vggt/utils/load_fn.py", "vggt/utils/pose_enc.py", "vggt/utils/geometry.py", "requirements.txt", "LICENSE.txt"]
    return {"commit": expected_commit, "files_sha256": {name: sha256(repo/name) for name in files}}


def validate_raw_arrays(path, frames, height, width):
    shapes = {"pose_enc": (frames, 9), "extrinsics": (frames, 3, 4), "intrinsics": (frames, 3, 3),
              "depth": (frames, height, width, 1), "depth_conf": (frames, height, width),
              "world_points": (frames, height, width, 3), "world_points_conf": (frames, height, width)}
    with np.load(path, allow_pickle=False) as arrays:
        for name, shape in shapes.items():
            value = arrays[name]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid raw VGGT {name}: expected finite float32 {shape}")
        if np.any(arrays["depth"] <= 0):
            raise ValueError("raw model produced nonpositive Z-depth; no silent filtering allowed")
    return {k: list(v) for k, v in shapes.items()}


def run_vggt(manifest_path, config_path, official_repo, checkpoint, output, preflight_only=False):
    start = time.perf_counter()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; attempts are never overwritten")
    config = validate_config(read_json(config_path))
    source = verify_repository(official_repo, config["official_commit"])
    manifest, images = load_scene(manifest_path, preprocess=True)
    if len({im.shape for im in images}) != 1:
        raise ValueError("this verified adapter requires equal common-preprocessed dimensions; official batch padding is not silently assumed")
    output.mkdir(parents=True, exist_ok=True)
    (output/"images").mkdir()
    mapping = []
    for index, (image, record) in enumerate(zip(images, manifest["images"])):
        name = f"{index:06d}.png"
        Image.fromarray(image).save(output/"images"/name)
        geometry = crop_geometry(image.shape[1], image.shape[0])
        A = np.array(geometry["pixel_transform"]) @ pixel_transform(manifest, record)
        mapping.append({"image_id": record["image_id"], "worker_name": name, "source_sha256": record["sha256"], "common_processed_sha256": sha256(output/"images"/name),
                        "original_wh": [record["width"], record["height"]], "common_wh": [image.shape[1], image.shape[0]],
                        "official_crop": geometry, "original_to_model_transform": A.tolist()})
    write_json(output/"input_mapping.json", mapping)
    write_json(output/"config.json", config)
    write_json(output/"request.json", {"official_repo": str(Path(official_repo).resolve()), "checkpoint": str(Path(checkpoint).resolve()), "preflight_only": preflight_only})
    write_json(output/"source.json", source)
    result = result_template(manifest["scene_id"], sha256(manifest_path), [r["image_id"] for r in mapping], config["method_id"], kind=manifest["kind"])
    result.update(config=config, run_id=output.name, provenance={**code_provenance(manifest["kind"]), "official_source": source,
                  "checkpoint_revision": config["checkpoint_revision"], "checkpoint_sha256": config["checkpoint_sha256"],
                  "gt_access": "none; no reference input", "camera_coordinates": "original image zero-based pixel centers"}, logs=["worker.log", "worker_state.json", "preflight.json", "process.json"])
    write_json(output/"prediction.json", validate_result(result))
    prep_seconds = time.perf_counter()-start
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS=str(config["num_threads"]))
    try:
        process = execute_worker([sys.executable, "-m", "georecon.baselines.vggt_worker", str(output)], config["timeout_seconds"], output/"worker.log", env)
    except OSError as exc:
        process = {"returncode": None, "timed_out": False, "error": str(exc), "wall_seconds": time.perf_counter()-start-prep_seconds}
    write_json(output/"process.json", process)
    try:
        state = read_json(output/"worker_state.json")
        if not isinstance(state, dict):
            raise ValueError("worker state must be an object")
    except (ValueError, OSError) as exc:
        state = {"error": f"cannot read worker state: {exc}"}
    expected_shape = [len(mapping), 3, mapping[0]["official_crop"]["crop_xywh"][3], 518]
    if process["returncode"] == 0 and not state.get("error"):
        try:
            audit = read_json(output/"preflight.json")
            if state.get("input_shape") != expected_shape or audit.get("input_shape") != expected_shape or audit["independent_loader_max_abs"] > 1e-7:
                raise ValueError("preprocessing shape/pixel audit mismatch")
            if preflight_only and state.get("status") != "preflight_complete":
                raise ValueError("worker did not complete preprocessing-only audit")
        except (ValueError, KeyError, OSError) as exc:
            state["error"] = f"invalid preprocessing audit: {exc}"
    if process["timed_out"]:
        result.update(status="timeout", failure_reason="raw inference worker exceeded configured budget")
    elif process["returncode"] != 0 or state.get("error"):
        result.update(status="oom" if state.get("failure_kind") == "oom" else "process_error", failure_reason=state.get("error", f"worker exited {process['returncode']}; see worker.log"))
    elif preflight_only:
        result["failure_reason"] = "preprocessing audit only; model not run"
    else:
        try:
            if state.get("status") != "complete" or state.get("checkpoint_sha256") != config["checkpoint_sha256"]:
                raise ValueError("worker did not finish verified local-checkpoint inference")
            shapes = validate_raw_arrays(output/"raw_predictions.npz", len(mapping), *state["input_shape"][-2:])
            payload = read_json(output/"export.json")
            if [r["image_id"] for r in payload["images"]] != [r["image_id"] for r in mapping]:
                raise ValueError("export changed input IDs or order")
            result.update(images=payload["images"], status="success", failure_reason=None)
            validate_result(result)
            digest = sha256(output/"raw_predictions.npz")
            for name, semantics in {"depth": "raw camera Z-depth on model-input grid, arbitrary reconstruction units; key depth", "pointcloud": "raw dense point-head world map, key world_points; not fused or filtered", "confidence": "raw depth_conf and world_points_conf on model-input grid; NOT calibrated probabilities"}.items():
                result["artifacts"][name] = {"path": "raw_predictions.npz", "sha256": digest, "semantics": semantics, "arrays": shapes, "coordinate_metadata": "preflight.json and native intrinsics/extrinsics in NPZ"}
        except (ValueError, KeyError, OSError) as exc:
            result.update(status="process_error", failure_reason=f"raw output validation failed: {exc}")
            result["images"] = [{"image_id": r["image_id"], "registered": False, "camera": None, "reason": "invalid_raw_export"} for r in mapping]
            result["artifacts"] = {k: None for k in result["artifacts"]}
    result["versions"] = state.get("versions", {})
    result["timing_seconds"] = {"input_validation_staging_and_source_check": prep_seconds, **state.get("timing_seconds", {}), "worker_wall": process["wall_seconds"], "total": time.perf_counter()-start}
    result["memory_bytes"] = {"peak_worker_rss_sampled": process.get("peak_worker_rss_sampled_bytes"), "peak_worker_rss_os": state.get("peak_rss_bytes"), "peak_gpu_allocated": None, "peak_gpu_reserved": None, **state.get("memory_bytes", {})}
    write_json(output/"prediction.json", validate_result(result))
    return result
