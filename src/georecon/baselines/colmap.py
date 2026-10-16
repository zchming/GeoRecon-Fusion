"""CPU COLMAP runner with an isolated worker, immutable inputs and no GT access."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from ..data.protocol import read_json, write_json, sha256, validate_result, result_template
from ..data.scene import load_scene
from ..geometry.camera import Camera, resize_crop
from ..provenance import code_provenance


def validate_config(config):
    if config["schema_version"] != 1 or config["pycolmap_version"] != "4.2.1":
        raise ValueError("this adapter requires the verified PyCOLMAP 4.2.1 API")
    if config["intrinsics"] != "estimate" or config["camera_model"] != "PINHOLE" or config["camera_sharing"] != "single":
        raise ValueError("only explicitly estimated shared PINHOLE intrinsics are supported")
    for key in ("num_threads", "sift_max_num_features", "min_model_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("timeout_seconds", "default_focal_length_factor"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive finite")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    return config


def pixel_transform(manifest, image):
    dummy = Camera(np.eye(3), np.eye(3), np.zeros(3), image["width"], image["height"])
    config = manifest["preprocessing"]
    if config["operation"] == "identity":
        return np.eye(3)
    return resize_crop(dummy, config["resize_wh"], config["crop_xywh"])[1]


def convert_camera(K_native, R, t, original_wh, transform):
    """COLMAP corner-based pixels -> zero-based centers -> original image pixels."""
    shift = np.array([[1., 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    K_original = np.linalg.solve(transform, shift @ np.asarray(K_native))
    return Camera(K_original, R, t, *original_wh)


def execute_worker(command, timeout, log_path, env=None):
    """Capture native stderr/stdout and terminate the entire worker process group."""
    start, timed_out, peak_rss = time.perf_counter(), False, 0
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while process.poll() is None:
            try:
                fields = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                peak_rss = max(peak_rss, next(int(line.split()[1])*1024 for line in fields if line.startswith("VmRSS:")))
            except (OSError, StopIteration):
                pass
            if time.perf_counter()-start > timeout:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.05)
        return {"returncode": process.wait(), "timed_out": timed_out, "wall_seconds": time.perf_counter()-start,
                "peak_worker_rss_sampled_bytes": peak_rss or None}


def validate_sparse_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        xyz, rgb, ids = data["xyz"], data["rgb"], data["point_ids"]
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or rgb.shape != xyz.shape or ids.shape != (len(xyz),):
            raise ValueError("invalid sparse point cloud shape or nonfinite coordinates")
        if rgb.dtype != np.uint8 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
            raise ValueError("invalid sparse RGB or duplicate point IDs")
    return len(xyz)


def run_colmap(manifest_path, config_path, output):
    start = time.perf_counter()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; previous attempts are never overwritten")
    config = validate_config(read_json(config_path))
    manifest, images = load_scene(manifest_path, preprocess=True)
    if len(images) < 2:
        raise ValueError("SfM requires at least two input images")
    if len({image.shape for image in images}) != 1:
        raise ValueError("shared-camera smoke configuration requires identical processed image sizes")
    output.mkdir(parents=True, exist_ok=True)
    (output/"images").mkdir()
    mapping = []
    for index, (image, record) in enumerate(zip(images, manifest["images"])):
        name = f"{index:06d}.png"
        Image.fromarray(image).save(output/"images"/name)
        mapping.append({"image_id": record["image_id"], "worker_name": name, "source_sha256": record["sha256"],
                        "processed_sha256": sha256(output/"images"/name), "original_wh": [record["width"], record["height"]],
                        "processed_wh": [image.shape[1], image.shape[0]], "pixel_transform": pixel_transform(manifest, record).tolist()})
    write_json(output/"input_mapping.json", mapping)
    write_json(output/"config.json", config)
    write_json(output/"manifest_snapshot.json", manifest)
    result = result_template(manifest["scene_id"], sha256(manifest_path), [r["image_id"] for r in mapping], config["method_id"], kind=manifest["kind"])
    result.update(config=config, provenance={**code_provenance(manifest["kind"]), "gt_access": "none; reference files never opened by adapter",
                                           "camera_coordinates": "original image, zero-based pixel centers", "backend": "official PyCOLMAP CPU SfM"},
                  logs=["worker.log", "worker_state.json", "process.json"], run_id=output.name)
    write_json(output/"prediction.json", validate_result(result))
    preprocessing_seconds = time.perf_counter()-start
    env = dict(os.environ, OMP_NUM_THREADS=str(config["num_threads"]), OPENBLAS_NUM_THREADS=str(config["num_threads"]))
    command = [sys.executable, "-m", "georecon.baselines.colmap_worker", str(output)]
    try:
        process = execute_worker(command, config["timeout_seconds"], output/"worker.log", env)
    except OSError as exc:
        process = {"returncode": None, "timed_out": False, "error": str(exc), "wall_seconds": time.perf_counter()-start-preprocessing_seconds}
    write_json(output/"process.json", process)
    state_error = None
    try:
        state = read_json(output/"worker_state.json") if (output/"worker_state.json").exists() else {}
        if not isinstance(state, dict):
            raise ValueError("worker state must be an object")
    except (ValueError, OSError) as exc:
        state_error = f"cannot read worker state: {exc}"
        state = {"error": state_error}
    if process["timed_out"]:
        result.update(status="timeout", failure_reason="worker exceeded configured wall-time budget; no incomplete model promoted")
    elif process["returncode"] != 0 or state_error:
        result.update(status="oom" if state.get("failure_kind") == "oom" else "process_error", failure_reason=state.get("error", process.get("error", f"worker exited {process['returncode']}; see worker.log")))
    else:
        try:
            payload = read_json(output/"export.json")
            if [r["image_id"] for r in payload["images"]] != [r["image_id"] for r in mapping]:
                raise ValueError("worker export changed input IDs/order")
            count = sum(r["registered"] for r in payload["images"])
            if count >= 2:
                validate_sparse_artifact(output/"sparse_points.npz")
            result["images"] = payload["images"]
            result.update(status="success" if count == len(mapping) else "partial" if count >= 2 else "no_reconstruction",
                          failure_reason=None if count == len(mapping) else "selected model did not register all inputs" if count >= 2 else "COLMAP returned no usable reconstruction")
            validate_result(result)
            if count >= 2:
                result["artifacts"]["pointcloud"] = {"path": "sparse_points.npz", "semantics": "sparse COLMAP xyz in reconstruction gauge; RGB uint8; not dense geometry", "sha256": sha256(output/"sparse_points.npz"), "world_unit": "arbitrary"}
        except (ValueError, KeyError, OSError) as exc:
            result.update(status="process_error", failure_reason=f"export validation failed: {exc}")
            result["images"] = [{"image_id": r["image_id"], "registered": False, "camera": None, "reason": "unusable_export"} for r in mapping]
            result["artifacts"]["pointcloud"] = None
    result["versions"] = state.get("versions", {})
    result["timing_seconds"] = {"input_validation_and_preprocessing": preprocessing_seconds, **state.get("timing_seconds", {}), "worker_wall": process["wall_seconds"], "total": time.perf_counter()-start}
    result["memory_bytes"] = {"peak_worker_rss_sampled": process.get("peak_worker_rss_sampled_bytes"), "peak_worker_rss_os": state.get("peak_rss_bytes"), "peak_gpu_allocated": None}
    write_json(output/"prediction.json", validate_result(result))
    return result
