"""DTU official SampleSet smoke loader. No model execution or dense evaluation."""
import hashlib
import platform
import PIL
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from .protocol import read_json, write_json, sha256, relative_file, validate_manifest, validate_result, result_template
from .remote_zip import RangeZip
from ..geometry.camera import decompose_projection, resize_crop
from ..provenance import code_provenance


def prepare_dtu(config_path, output):
    config = read_json(config_path)
    output = Path(output)
    if (output/"manifest.json").exists():
        raise ValueError("output already prepared; use inspect-scene or choose a new output directory")
    members = {"SampleSet/ReadMe.txt": "source/ReadMe.txt"}
    for index in config["view_ids"]:
        members[f"SampleSet/MVS Data/Rectified/scan{config['scan']}/rect_{index:03d}_{config['light']}_r5000.png"] = f"images/{index:03d}.png"
        members[f"SampleSet/MVS Data/Calibration/cal18/pos_{index:03d}.txt"] = f"calibration/pos_{index:03d}.txt"
    start = time.perf_counter()
    remote = RangeZip(config["archive_url"], config["max_transfer_bytes"])
    if config.get("archive_size_bytes", remote.size) != remote.size or config.get("archive_etag", remote.etag) != remote.etag:
        raise ValueError("official archive identity changed; recheck source before updating pins")
    provenance = {"source_url": remote.url, "etag": remote.etag, "archive_size_bytes": remote.size,
                  "members": [], "scope": "real_scene_loader_only_not_benchmark", "config": config, "code": code_provenance("real")}
    with zipfile.ZipFile(remote) as archive:
        infos = [archive.getinfo(member) for member in members]
        if sum(m.file_size for m in infos) > config["max_transfer_bytes"]:
            raise ValueError("uncompressed member budget exceeded")
        for info in infos:
            raw = archive.read(info)  # zipfile also verifies member CRC32
            target = relative_file(output, members[info.filename])
            expected = config.get("member_sha256", {}).get(info.filename)
            actual = hashlib.sha256(raw).hexdigest()
            if expected and actual != expected:
                raise ValueError(f"source member hash changed: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            provenance["members"].append({"archive_member": info.filename, "local_path": members[info.filename],
                                          "bytes": len(raw), "sha256": actual, "crc32": f"{info.CRC:08x}"})
    provenance.update(transferred_bytes=remote.transferred, ranges=remote.ranges, download_seconds=time.perf_counter()-start)
    image_records, cameras, transforms = [], [], []
    for index in config["view_ids"]:
        name = f"images/{index:03d}.png"
        with Image.open(output/name) as image:
            width, height = image.size
        P = np.loadtxt(output/f"calibration/pos_{index:03d}.txt")
        camera = decompose_projection(P, width, height)
        # Source projection is only known up to scale; compare normalized matrices.
        original, rebuilt = P/np.linalg.norm(P), camera.projection/np.linalg.norm(camera.projection)
        residual = min(np.max(np.abs(original-rebuilt)), np.max(np.abs(original+rebuilt)))
        if residual > 1e-10:
            raise ValueError("projection decomposition does not reconstruct source calibration")
        processed, A = resize_crop(camera, config["resize_wh"], config["crop_xywh"])
        image_records.append({"image_id": f"{index:03d}", "path": name, "sha256": sha256(output/name), "width": width, "height": height})
        cameras.append(camera)
        transforms.append({"image_id": f"{index:03d}", "pixel_transform": A.tolist(),
                           "processed_camera": processed.as_dict(), "projection_reconstruction_max_abs": float(residual)})
    manifest = {"schema_version": 1, "kind": "real", "scene_id": config["scene_id"], "split": config["split"],
                "seed": config["seed"], "source": {"url": config["source_page"], "archive_url": remote.url, "scan": config["scan"], "light": config["light"]},
                "license_note": "DTU SampleSet ReadMe calls data citeware; cite Jensen et al., CVPR 2014. Raw data stays local.",
                "images": image_records, "preprocessing": {"operation": "resize_then_crop", "resize_wh": config["resize_wh"],
                    "crop_xywh": config["crop_xywh"], "pixel_centers": "u_out=sx*(u_in+0.5)-0.5-left", "implementation": "Pillow bilinear"},
                "budget": {"max_download_bytes": config["max_transfer_bytes"], "baseline_runs": 0}}
    write_json(output/"manifest.json", validate_manifest(manifest))
    reference = result_template(config["scene_id"], sha256(output/"manifest.json"), [r["image_id"] for r in image_records], "dtu_calibration", "reference", "real")
    reference.update(status="success", failure_reason=None, world_unit="mm", config=config,
                     provenance={"data_kind": "real", "reference_type": "official_projection_matrices_not_method_prediction", "source_url": remote.url, **code_provenance("real")},
                     versions={"python": platform.python_version(), "numpy": np.__version__, "Pillow": PIL.__version__})
    reference["images"] = [{"image_id": record["image_id"], "registered": True, "camera": camera.as_dict(), "reason": None} for record, camera in zip(image_records, cameras)]
    write_json(output/"reference.json", validate_result(reference))
    write_json(output/"camera_preprocessing_check.json", {"scope": "calibration_algebra_only_not_accuracy", "transforms": transforms})
    write_json(output/"download_provenance.json", provenance)
    pending = result_template(config["scene_id"], sha256(output/"manifest.json"), [r["image_id"] for r in image_records], "baseline_not_run", kind="real")
    write_json(output/"prediction_not_run.json", validate_result(pending))
    return {"scene_id": config["scene_id"], "images": len(image_records), "transferred_bytes": remote.transferred,
            "status": "loaded_calibration_only", "baseline_status": "not_run"}
