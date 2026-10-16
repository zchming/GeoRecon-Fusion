"""Deterministic mathematical fixture generation, explicitly not a baseline."""
from pathlib import Path
import platform
import time
import numpy as np
import PIL
from PIL import Image, ImageDraw
from .provenance import code_provenance
from .geometry.camera import Camera
from .data.protocol import read_json, write_json, sha256, result_template, validate_result, validate_manifest
from .data.scene import load_scene
from .evaluation.poses import evaluate_poses


def run_demo(config_path, output):
    start = time.perf_counter()
    config = read_json(config_path)
    output = Path(output)
    if (output/"manifest.json").exists():
        raise ValueError("output exists; select a new run directory to preserve provenance")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    width, height = config["image_wh"]
    K = np.array([[config["focal_px"], 0, (width-1)/2], [0, config["focal_px"], (height-1)/2], [0, 0, 1.]])
    centers = np.array(config["centers"], dtype=float)
    cameras = [Camera(K, np.eye(3), -center, width, height) for center in centers]
    points = rng.uniform([-0.8, -0.6, 3.5], [0.8, 0.6, 5.5], (config["points"], 3))
    images = []
    for index, camera in enumerate(cameras):
        image = Image.new("RGB", (width, height), (12, 18, 24))
        draw = ImageDraw.Draw(image)
        pixels, _, valid = camera.project(points)
        for (u, v), ok in zip(pixels, valid):
            if ok and 0 <= u < width and 0 <= v < height:
                draw.ellipse((u-1, v-1, u+1, v+1), fill=(240, 200, 60))
        name = f"images/{index:03d}.png"
        (output/"images").mkdir(exist_ok=True)
        image.save(output/name)
        images.append({"image_id": f"{index:03d}", "path": name, "sha256": sha256(output/name), "width": width, "height": height})
    manifest = {"schema_version": 1, "kind": "synthetic", "scene_id": config["scene_id"], "split": "smoke", "seed": config["seed"],
                "source": {"generator": "georecon.synthetic", "config": config}, "license_note": "Generated synthetic mathematical fixture, not real Benchmark data",
                "images": images, "preprocessing": {"operation": "identity"}, "budget": {"baseline_runs": 0}}
    write_json(output/"manifest.json", validate_manifest(manifest))
    manifest_hash = sha256(output/"manifest.json")
    reference = result_template(config["scene_id"], manifest_hash, [i["image_id"] for i in images], "synthetic_reference", "reference")
    reference.update(status="success", failure_reason=None, world_unit="m", provenance=code_provenance(), config=config,
                     versions={"python": platform.python_version(), "numpy": np.__version__, "Pillow": PIL.__version__})
    reference["images"] = [{"image_id": image["image_id"], "registered": True, "camera": camera.as_dict(), "reason": None} for image, camera in zip(images, cameras)]
    write_json(output/"reference.json", validate_result(reference))
    theta = config["similarity_yaw_radians"]
    Q = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    summaries = {}
    for variant in ("exact_sim3", "perturbed_partial", "no_reconstruction"):
        result = result_template(config["scene_id"], manifest_hash, [i["image_id"] for i in images], f"synthetic_{variant}")
        result.update(provenance=code_provenance(), config=config, versions=reference["versions"])
        if variant != "no_reconstruction":
            for index, (record, camera) in enumerate(zip(result["images"], cameras)):
                if variant == "perturbed_partial" and index == config["missing_camera_index"]:
                    record["reason"] = "intentionally_missing_synthetic_camera"
                    continue
                center = camera.center.copy()
                if variant == "perturbed_partial" and index == config["perturb_camera_index"]:
                    center += config["perturb_center"]
                center = config["similarity_scale"]*Q @ center + config["similarity_translation"]
                R = camera.R @ Q.T
                estimated = Camera(camera.K, R, -R @ center, width, height)
                record.update(registered=True, camera=estimated.as_dict(), reason=None)
            result.update(status="success" if variant == "exact_sim3" else "partial", failure_reason=None if variant == "exact_sim3" else "one camera intentionally omitted")
        else:
            result.update(status="no_reconstruction", failure_reason="intentional synthetic failure fixture")
        write_json(output/f"prediction_{variant}.json", validate_result(result))
        summary = evaluate_poses(result, reference)
        write_json(output/f"evaluation_{variant}.json", summary)
        summaries[variant] = summary
    load_scene(output/"manifest.json")
    write_json(output/"fixture_geometry.json", {"kind": "synthetic", "points_world": points.tolist()})
    write_json(output/"run.json", {"scope": "synthetic_fixture_generation_and_evaluation_not_model_inference", "config": config,
                                   "provenance": code_provenance(), "versions": reference["versions"], "total_seconds": time.perf_counter()-start})
    return {key: {"registration_ratio": value["registration_ratio"], "rotation_evaluated_pairs": value["rotation_evaluated_pairs"],
                  "translation_direction_deg": value["translation_direction_deg_registered_subset"], "ate_rmse": value["camera_center_ate"]["rmse"]} for key, value in summaries.items()}
