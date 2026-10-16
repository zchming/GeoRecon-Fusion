"""Local pinned VGGT forward. No downloads, queries, BA or filtering."""
import hashlib
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import numpy as np
from PIL import Image
import PIL
from ..data.protocol import read_json, write_json, sha256
from .vggt import independent_crop, convert_camera, validate_raw_arrays


def main(output):
    state = {"status": "starting", "timing_seconds": {}, "memory_bytes": {}}
    def save():
        state["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write_json(output/"worker_state.tmp.json", state)
        (output/"worker_state.tmp.json").replace(output/"worker_state.json")
    save()
    try:
        started = time.perf_counter()
        import torch
        import torchvision
        import huggingface_hub
        import einops
        import safetensors
        request, config = read_json(output/"request.json"), read_json(output/"config.json")
        mapping = read_json(output/"input_mapping.json")
        sys.path.insert(0, request["official_repo"])
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        state["versions"] = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
                             "numpy": np.__version__, "Pillow": PIL.__version__, "huggingface_hub": huggingface_hub.__version__, "einops": einops.__version__, "safetensors": safetensors.__version__}
        state["timing_seconds"]["runtime_and_official_utilities_import"] = time.perf_counter()-started
        torch.set_num_threads(config["num_threads"])
        torch.manual_seed(config["seed"])
        paths = [str(output/"images"/i["worker_name"]) for i in mapping]
        state["status"] = "preprocessing_audit"
        save()
        started = time.perf_counter()
        images = load_and_preprocess_images(paths, mode="crop")
        actual = images.numpy()
        expected = []
        for path in paths:
            with Image.open(path) as image:
                expected.append(independent_crop(image))
        expected = np.stack(expected)
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError("official preprocessing differs from independently derived resize/crop")
        if actual.dtype != np.float32 or actual.min() < 0 or actual.max() > 1:
            raise ValueError("official inputs must be float32 RGB in [0,1]")
        state["input_shape"] = list(images.shape)
        np.save(output/"model_input.npy", actual, allow_pickle=False)
        audit = {"scope": "pre-inference preprocessing audit; no GT", "input_shape": list(actual.shape), "dtype": str(actual.dtype), "range": [float(actual.min()), float(actual.max())],
                 "independent_loader_max_abs": float(np.max(np.abs(actual-expected))), "tensor_bytes_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                 "pixel_convention": "official geometry.py uses arange integer pixel centers; no COLMAP -0.5 shift", "mapping": mapping}
        write_json(output/"preflight.json", audit)
        state["timing_seconds"]["official_preprocessing_and_independent_audit"] = time.perf_counter()-started
        if request["preflight_only"]:
            state["status"] = "preflight_complete"
            save()
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("configured CUDA BF16 unavailable; no silent CPU/FP16 fallback")
        torch.cuda.manual_seed_all(config["seed"])
        state["device"] = {"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                           "free_total_bytes_before": list(torch.cuda.mem_get_info()), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark}
        state["status"] = "checkpoint_verification"
        save()
        started = time.perf_counter()
        digest = sha256(request["checkpoint"])
        state["timing_seconds"]["checkpoint_sha256"] = time.perf_counter()-started
        if digest != config["checkpoint_sha256"]:
            raise ValueError("local checkpoint SHA-256 mismatch; no automatic download")
        state["checkpoint_sha256"] = digest
        state["status"] = "model_loading"
        save()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        from vggt.models.vggt import VGGT
        model = VGGT().eval()
        weights = torch.load(request["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(weights, strict=True)
        del weights
        model = model.to("cuda")
        images = images.to("cuda")
        torch.cuda.synchronize()
        state["timing_seconds"]["model_construct_checkpoint_load_and_gpu_transfer"] = time.perf_counter()-started
        state["memory_bytes"].update(gpu_after_load_allocated=torch.cuda.memory_allocated(), gpu_after_load_reserved=torch.cuda.memory_reserved())
        state["status"] = "forward"
        save()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(images)  # All default heads, no query_points -> no tracking run.
        torch.cuda.synchronize()
        state["timing_seconds"]["synchronized_raw_forward"] = time.perf_counter()-started
        state["memory_bytes"].update(peak_gpu_allocated=torch.cuda.max_memory_allocated(), peak_gpu_reserved=torch.cuda.max_memory_reserved())
        state["status"] = "decode_export_and_validate"
        save()
        started = time.perf_counter()
        with torch.inference_mode():
            extrinsics, intrinsics = pose_encoding_to_extri_intri(prediction["pose_enc"].float(), images.shape[-2:])
        torch.cuda.synchronize()
        arrays = {key: prediction[key][0].float().cpu().numpy() for key in ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf")}
        arrays.update(extrinsics=extrinsics[0].float().cpu().numpy(), intrinsics=intrinsics[0].float().cpu().numpy(),
                      pose_enc_iterations=np.stack([value[0].float().cpu().numpy() for value in prediction["pose_enc_list"]]))
        np.savez(output/"raw_predictions.npz", **arrays)
        frames, _, height, width = images.shape
        state["array_shapes"] = validate_raw_arrays(output/"raw_predictions.npz", frames, height, width)
        exported, conversion_errors, unprojection_errors = [], [], []
        for index, item in enumerate(mapping):
            K, E = arrays["intrinsics"][index].astype(float), arrays["extrinsics"][index].astype(float)
            camera = convert_camera(K, E, item["original_wh"], np.array(item["original_to_model_transform"]))
            exported.append({"image_id": item["image_id"], "registered": True, "camera": camera.as_dict(), "reason": None})
            # Deterministic sparse samples: algebra cross-check only, never reliability evidence.
            x, y = np.meshgrid(np.linspace(0, width-1, 8, dtype=int), np.linspace(0, height-1, 6, dtype=int))
            native_pixels = np.c_[x.ravel(), y.ravel()]
            depths = arrays["depth"][index, y, x, 0].ravel().astype(float)
            native_points = (np.c_[native_pixels, np.ones(len(native_pixels))] @ np.linalg.inv(K).T)*depths[:, None]
            world = (native_points-E[:, 3]) @ E[:, :3]
            original_pixels, _, valid = camera.project(world)
            remapped = np.c_[original_pixels, np.ones(len(original_pixels))] @ np.array(item["original_to_model_transform"]).T
            if not valid.all():
                raise ValueError("camera export changed sampled depth signs")
            conversion_errors.append(float(np.max(np.abs(remapped[:, :2]-native_pixels))))
            official_world = unproject_depth_map_to_point_map(arrays["depth"][index:index+1], arrays["extrinsics"][index:index+1], arrays["intrinsics"][index:index+1])[0]
            unprojection_errors.append(float(np.max(np.abs(official_world[y, x].reshape(-1, 3)-world))))
        if max(conversion_errors) > 1e-3 or max(unprojection_errors) > 1e-3:
            raise ValueError("official/project coordinate algebra check failed")
        write_json(output/"export.json", {"images": exported})
        state["validation"] = {"scope": "internal algebra only, NOT independent accuracy or reliability evidence",
                               "camera_export_max_abs_model_px": max(conversion_errors), "official_depth_unprojection_max_abs_world": max(unprojection_errors),
                               "raw_model_keys": list(prediction), "tracks": "not_run: no query points", "raw_points": int(frames*height*width),
                               "confidence_ranges": {key: [float(arrays[key].min()), float(arrays[key].max())] for key in ("depth_conf", "world_points_conf")}}
        state["timing_seconds"]["decode_cpu_export_and_internal_checks"] = time.perf_counter()-started
        state["memory_bytes"].update(peak_gpu_allocated=torch.cuda.max_memory_allocated(), peak_gpu_reserved=torch.cuda.max_memory_reserved())
        state["status"] = "complete"
        save()
    except Exception as exc:
        state.update(status="process_error", failure_kind="oom" if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError" else "process_error", error=f"{type(exc).__name__}: {exc}")
        if "torch" in locals() and torch.cuda.is_available():
            state["memory_bytes"].update(peak_gpu_allocated=torch.cuda.max_memory_allocated(), peak_gpu_reserved=torch.cuda.max_memory_reserved())
        save()
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main(Path(sys.argv[1]))
