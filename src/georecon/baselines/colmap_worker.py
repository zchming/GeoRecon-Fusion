"""Native COLMAP worker; receives only processed images, IDs and method options."""
import platform
from pathlib import Path
import resource
import sqlite3
import sys
import time
import traceback
import numpy as np
import PIL
from ..data.protocol import read_json, write_json
from .colmap import convert_camera


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serializable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def export_model(reconstruction, mapping, output):
    images_by_name = {image.name: image for image in reconstruction.images.values() if image.has_pose}
    exported, residuals, conversion_errors, positive, observation_count = [], [], [], 0, 0
    for item in mapping:
        image = images_by_name.get(item["worker_name"])
        if image is None:
            exported.append({"image_id": item["image_id"], "registered": False, "camera": None, "reason": "not_registered_in_selected_component"})
            continue
        native_camera = reconstruction.cameras[image.camera_id]
        if native_camera.model_name != "PINHOLE":
            raise ValueError("refuse to silently discard camera distortion")
        if [native_camera.width, native_camera.height] != item["processed_wh"]:
            raise ValueError("COLMAP camera dimensions differ from staged input")
        pose = image.cam_from_world()
        A = np.array(item["pixel_transform"])
        camera = convert_camera(native_camera.calibration_matrix(), pose.rotation.matrix(), pose.translation, item["original_wh"], A)
        if not np.allclose(camera.center, image.projection_center(), atol=1e-8):
            raise ValueError("world-to-camera export changed camera center")
        exported.append({"image_id": item["image_id"], "registered": True, "camera": camera.as_dict(), "reason": None})
        for observation in image.points2D:
            if not observation.has_point3D():
                continue
            point = reconstruction.points3D[observation.point3D_id].xyz
            camera_point = pose * point
            if not np.isfinite(camera_point).all():
                raise ValueError("nonfinite model point")
            observation_count += 1
            if camera_point[2] <= 0:
                continue
            positive += 1
            native_pixel = native_camera.img_from_cam(camera_point)
            pixels, _, valid = camera.project(point[None])
            if not valid[0]:
                raise ValueError("export changed point depth sign")
            from_export = A @ np.r_[pixels[0], 1]
            conversion_errors.append(float(np.max(np.abs(from_export[:2]+0.5-native_pixel))))
            residuals.append(float(np.linalg.norm(native_pixel-observation.xy)))
    if conversion_errors and max(conversion_errors) > 1e-7:
        raise ValueError("pixel-center/intrinsics export validation failed")
    points = sorted(reconstruction.points3D.items())
    np.savez(output/"sparse_points.npz", xyz=np.array([p.xyz for _, p in points], dtype=float).reshape(-1, 3),
             rgb=np.array([p.color for _, p in points], dtype=np.uint8).reshape(-1, 3), point_ids=np.array([i for i, _ in points], dtype=np.int64))
    return exported, {"scope": "same fitted observations; internal consistency, NOT independent reliability evidence",
                      "observations": observation_count, "positive_depth_observations": positive,
                      "reprojection_mean_processed_px": float(np.mean(residuals)) if residuals else None,
                      "reprojection_median_processed_px": float(np.median(residuals)) if residuals else None,
                      "native_to_export_max_abs_processed_px": max(conversion_errors) if conversion_errors else None}


def main(output):
    state = {"status": "starting", "timing_seconds": {}}
    def save():
        state["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write_json(output/"worker_state.tmp.json", serializable(state))
        (output/"worker_state.tmp.json").replace(output/"worker_state.json")
    save()
    try:
        start = time.perf_counter()
        import pycolmap as colmap
        config, mapping = read_json(output/"config.json"), read_json(output/"input_mapping.json")
        if colmap.__version__ != config["pycolmap_version"]:
            raise ValueError(f"PyCOLMAP version mismatch: {colmap.__version__}")
        state["versions"] = {"pycolmap": colmap.__version__, "python": platform.python_version(), "numpy": np.__version__, "Pillow": PIL.__version__, "platform": platform.platform()}
        state["cuda_build_support"] = colmap.has_cuda
        state["device"] = "CPU explicitly for feature extraction, matching and mapper BA"
        state["timing_seconds"]["backend_import"] = time.perf_counter()-start
        colmap.set_random_seed(config["seed"])
        extraction = colmap.FeatureExtractionOptions()
        extraction.num_threads = config["num_threads"]
        extraction.max_image_size = max(mapping[0]["processed_wh"])
        extraction.use_gpu = False
        extraction.sift.max_num_features = config["sift_max_num_features"]
        matching = colmap.FeatureMatchingOptions()
        matching.num_threads, matching.use_gpu = config["num_threads"], False
        reader = colmap.ImageReaderOptions()
        reader.camera_model = config["camera_model"]
        reader.default_focal_length_factor = config["default_focal_length_factor"]
        mapper = colmap.IncrementalPipelineOptions()
        mapper.num_threads = config["num_threads"]
        mapper.random_seed = config["seed"]
        mapper.mapper.random_seed = config["seed"]
        mapper.triangulation.random_seed = config["seed"]
        mapper.min_model_size = config["min_model_size"]
        mapper.ba_use_gpu = False
        mapper.max_runtime_seconds = config["timeout_seconds"]
        state["effective_options"] = {"extraction": extraction.todict(), "matching": matching.todict(), "reader": reader.todict(),
                                      "mapper": mapper.todict(), "camera_mode": "SINGLE", "device": "cpu"}
        database = output/"database.db"
        for stage, fn in [
            ("feature_extraction", lambda: colmap.extract_features(database, output/"images", image_names=[i["worker_name"] for i in mapping], camera_mode=colmap.CameraMode.SINGLE, reader_options=reader, extraction_options=extraction, device=colmap.Device.cpu)),
            ("exhaustive_matching", lambda: colmap.match_exhaustive(database, matching_options=matching, device=colmap.Device.cpu)),
        ]:
            state["status"] = stage
            save()
            start = time.perf_counter()
            fn()
            state["timing_seconds"][stage] = time.perf_counter()-start
            save()
        with sqlite3.connect(database) as db:
            state["database"] = {"images": db.execute("SELECT image_id,name,camera_id FROM images ORDER BY image_id").fetchall(),
                                 "keypoints": db.execute("SELECT image_id,rows FROM keypoints ORDER BY image_id").fetchall(),
                                 "verified_pairs": db.execute("SELECT pair_id,rows,config FROM two_view_geometries ORDER BY pair_id").fetchall(),
                                 "cameras": [{"camera_id": i, "prior_focal_length": bool(prior), "initial_params": np.frombuffer(params, dtype=np.float64).tolist()} for i, prior, params in db.execute("SELECT camera_id,prior_focal_length,params FROM cameras ORDER BY camera_id")]}
        state["status"] = "incremental_mapping"
        save()
        (output/"sparse").mkdir(exist_ok=True)
        start = time.perf_counter()
        models = colmap.incremental_mapping(database, output/"images", output/"sparse", options=mapper)
        state["timing_seconds"]["incremental_mapping_including_native_ba"] = time.perf_counter()-start
        # All components are retained; selection uses no ground truth or evaluated error.
        candidates = [{"model_id": int(i), "registered_images": m.num_reg_images(), "points3D": m.num_points3D()} for i, m in sorted(models.items())]
        state["models"] = candidates
        usable = [item for item in candidates if item["registered_images"] >= 2]
        start = time.perf_counter()
        if usable:
            chosen = sorted(usable, key=lambda m: (-m["registered_images"], -m["points3D"], m["model_id"]))[0]
            reconstruction = models[chosen["model_id"]]
            exported, validation = export_model(reconstruction, mapping, output)
            state.update(selected_model=chosen, validation=validation)
            reconstruction.write(output/"sparse"/str(chosen["model_id"]))
        else:
            exported = [{"image_id": i["image_id"], "registered": False, "camera": None, "reason": "no_usable_model"} for i in mapping]
            state["selected_model"] = None
        write_json(output/"export.json", {"images": exported})
        state["timing_seconds"]["export_and_validation"] = time.perf_counter()-start
        state["status"] = "complete"
        save()
    except Exception as exc:
        state.update(status="process_error", failure_kind="oom" if isinstance(exc, MemoryError) else "process_error", error=f"{type(exc).__name__}: {exc}")
        save()
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main(Path(sys.argv[1]))
