"""Read-only environment audit; no model/data downloads or baseline execution."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import urllib.request


def command(args, timeout=60, cwd=None):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", action="store_true", help="HEAD requests only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "stage0_environment_only", "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(),
              "wsl": "microsoft" in platform.release().lower(),
              "cpu_count_visible": os.cpu_count(), "disk_bytes": shutil.disk_usage(root)._asdict(),
              "commands": {}, "imports": {}}
    for name, cmd in {
        "os": ["cat", "/etc/os-release"], "cpu": ["lscpu"], "memory": ["free", "-b"],
        "gpu": ["nvidia-smi"], "cmake": ["cmake", "--version"],
        "ninja": ["ninja", "--version"], "compiler": ["c++", "--version"],
        "colmap": ["colmap", "-h"], "cuda_compiler": ["nvcc", "--version"],
        "pip_check": [sys.executable, "-m", "pip", "check"],
        "colmap_build_dependencies": ["pkg-config", "--modversion", "eigen3", "ceres-solver", "sqlite3"],
    }.items():
        report["commands"][name] = command(cmd)
    for name in ["numpy", "scipy", "cv2", "PIL", "torch", "torchvision", "matplotlib",
                 "pycolmap", "pyceres", "lightglue", "trimesh", "huggingface_hub", "einops", "safetensors"]:
        try:
            mod = importlib.import_module(name)
            report["imports"][name] = {"status": "ok", "version": getattr(mod, "__version__", "unknown")}
        except Exception as exc:
            report["imports"][name] = {"status": "unavailable", "reason": str(exc)}
    try:
        import torch
        torch.manual_seed(42)
        cuda = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
        if cuda["available"]:
            cuda.update(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                        free_total_bytes=torch.cuda.mem_get_info(), bf16_supported=torch.cuda.is_bf16_supported())
            x = torch.randn(128, 128)
            y = torch.randn(128, 128)
            torch.testing.assert_close((x.cuda() @ y.cuda()).cpu(), x @ y, rtol=1e-3, atol=1e-3)
            cuda["matmul_check"] = "PASS (128x128, rtol=atol=1e-3); not model inference"
        report["cuda"] = cuda
    except Exception as exc:
        report["cuda"] = {"error": str(exc)}
    with tempfile.TemporaryDirectory(prefix="georecon-stage0-") as d:
        tmp = Path(d)
        (tmp / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.16)\nproject(probe LANGUAGES CXX)\nadd_executable(probe main.cpp)\n')
        (tmp / "main.cpp").write_text('#include <iostream>\nint main(){std::cout << "compiler probe passed\\n";return 0;}\n')
        report["compiler_probe"] = [command(["cmake", "-S", d, "-B", str(tmp / "build"), "-G", "Ninja"])]
        if report["compiler_probe"][-1].get("returncode") == 0:
            report["compiler_probe"].append(command(["cmake", "--build", str(tmp / "build")]))
            if report["compiler_probe"][-1].get("returncode") == 0:
                report["compiler_probe"].append(command([str(tmp / "build/probe")]))
    official = root / "experiment03_depth_reprojection/vggt_official"
    report["existing_vggt"] = {"commit": command(["git", "rev-parse", "HEAD"], cwd=official),
                                "changes": command(["git", "status", "--short"], cwd=official)}
    checkpoint = root / "experiment03_depth_reprojection/model.pt"
    if checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        report["existing_checkpoint"] = {"path": str(checkpoint.relative_to(root)),
                                          "bytes": checkpoint.stat().st_size, "sha256": digest.hexdigest()}
    report["legacy_dependency_import_probe"] = command([
        sys.executable, "-c",
        "import sys,json,importlib; sys.path.insert(0,sys.argv[1]); "
        "mods=['huggingface_hub','einops','safetensors']; "
        "print(json.dumps({m:getattr(importlib.import_module(m),'__version__','unknown') for m in mods}))",
        str(root / "experiment03_depth_reprojection/deps")])
    if args.network:
        urls = ["https://github.com/facebookresearch/vggt", "https://colmap.github.io/install.html",
                "https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt",
                "https://www.eth3d.net/documentation", "https://roboimagedata.compute.dtu.dk/?page_id=36"]
        report["network_head_only"] = {}
        for url in urls:
            try:
                request = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(request, timeout=15) as response:
                    report["network_head_only"][url] = {"status": response.status}
            except Exception as exc:
                report["network_head_only"][url] = {"error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Audit saved to {args.output}; inspect per-check results (completion is not a blanket PASS).")


if __name__ == "__main__":
    main()
