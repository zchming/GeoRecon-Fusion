import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
from georecon.baselines.colmap import convert_camera, execute_worker, run_colmap, validate_config, validate_sparse_artifact
from georecon.data.protocol import read_json, write_json, sha256
from georecon.geometry.camera import Camera

ROOT = Path(__file__).resolve().parents[1]


def scene(root):
    root.mkdir()
    records = []
    for index in range(3):
        name = f"{index}.png"
        Image.new("RGB", (64, 48)).save(root/name)
        records.append({"image_id": str(index), "path": name, "sha256": sha256(root/name), "width": 64, "height": 48})
    manifest = {"schema_version": 1, "scene_id": "blank_synthetic_failure_test", "kind": "synthetic", "split": "smoke", "seed": 42,
                "source": "test fixture", "license_note": "generated", "preprocessing": {"operation": "identity"}, "budget": {}, "images": records}
    write_json(root/"manifest.json", manifest)
    # Deliberately invalid reference: an accidental reference read would fail.
    (root/"reference.json").write_text("DO NOT READ: ground truth is not an inference input")
    return root/"manifest.json"


class ColmapAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = scene(self.root/"input")
        self.config = ROOT/"configs/methods/colmap_cpu_smoke.json"

    def test_pixel_origin_resize_and_crop_known_numbers(self):
        # Native cx=100,cy=75 -> center-based 99.5,74.5; invert crop/half-size.
        A = np.array([[.5, 0, -10.25], [0, .5, -5.25], [0, 0, 1.]])
        native = np.array([[200., 0, 100], [0, 220, 75], [0, 0, 1]])
        camera = convert_camera(native, np.eye(3), np.zeros(3), [400, 300], A)
        assert_allclose(camera.K, [[400, 0, 219.5], [0, 440, 159.5], [0, 0, 1]])
        pixels = camera.project([[0, 0, 2]])[0][0]
        assert_allclose((A @ np.r_[pixels, 1])[:2]+.5, [100, 75])

    def test_known_intrinsics_not_silently_enabled(self):
        config = read_json(self.config)
        config["intrinsics"] = "known_gt"
        with self.assertRaises(ValueError):
            validate_config(config)
        config = read_json(self.config)
        config["timeout_seconds"] = float("inf")
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_timeout_kills_worker_and_preserves_log(self):
        log = self.root/"timeout.log"
        result = execute_worker([sys.executable, "-u", "-c", "import time; print('started',flush=True); time.sleep(30)"], .15, log)
        self.assertTrue(result["timed_out"])
        self.assertLess(result["wall_seconds"], 5)
        self.assertIn("started", log.read_text())

    def test_sparse_shape_nan_and_ids_are_validated(self):
        path = self.root/"points.npz"
        for xyz, ids in [(np.array([[np.nan, 0, 0]]), np.array([1])), (np.zeros((2, 3)), np.array([1, 1])), (np.zeros((1, 3)), np.array([1.5]))]:
            np.savez(path, xyz=xyz, rgb=np.zeros(xyz.shape, dtype=np.uint8), point_ids=ids)
            with self.assertRaises(ValueError):
                validate_sparse_artifact(path)

    def _fake_worker(self, status="ok", corrupt=False):
        def execute(command, timeout, log_path, env):
            out = Path(log_path).parent
            Path(log_path).write_text("simulated worker for adapter state test, not COLMAP measurement")
            write_json(out/"worker_state.json", {"versions": {"backend": "test_fake"}, "error": "simulated failure", "failure_kind": "oom" if status == "oom" else "process_error"})
            if status == "ok":
                mapping = read_json(out/"input_mapping.json")
                camera = Camera([[50, 0, 31.5], [0, 50, 23.5], [0, 0, 1]], np.eye(3), [0, 0, 0], 64, 48)
                rows = [{"image_id": i["image_id"], "registered": k < 2, "camera": camera.as_dict() if k < 2 else None, "reason": None if k < 2 else "not_registered"} for k, i in enumerate(mapping)]
                if corrupt:
                    rows[0]["image_id"] = "unexpected"
                write_json(out/"export.json", {"images": rows})
                np.savez(out/"sparse_points.npz", xyz=np.array([[0., 0, 3]]), rgb=np.zeros((1, 3), dtype=np.uint8), point_ids=np.array([1]))
            return {"returncode": 0 if status == "ok" else 1, "timed_out": status == "timeout", "wall_seconds": .01, "peak_worker_rss_sampled_bytes": None}
        return execute

    def test_partial_preserves_order_and_does_not_open_reference(self):
        with patch("georecon.baselines.colmap.execute_worker", side_effect=self._fake_worker()):
            result = run_colmap(self.manifest, self.config, self.root/"partial")
        self.assertEqual(result["status"], "partial")
        self.assertEqual([r["image_id"] for r in result["images"]], ["0", "1", "2"])
        self.assertIsNone(result["images"][2]["camera"])
        self.assertIsNone(result["artifacts"]["depth"])

    def test_timeout_oom_and_crash_are_saved(self):
        for state, expected in [("timeout", "timeout"), ("oom", "oom"), ("error", "process_error")]:
            with self.subTest(state=state), patch("georecon.baselines.colmap.execute_worker", side_effect=self._fake_worker(state)):
                result = run_colmap(self.manifest, self.config, self.root/state)
            self.assertEqual(result["status"], expected)
            self.assertTrue(all(not r["registered"] for r in result["images"]))
            self.assertEqual(read_json(self.root/state/"prediction.json")["status"], expected)

    def test_corrupt_export_is_failure_not_fake_success(self):
        with patch("georecon.baselines.colmap.execute_worker", side_effect=self._fake_worker(corrupt=True)):
            result = run_colmap(self.manifest, self.config, self.root/"corrupt")
        self.assertEqual(result["status"], "process_error")
        self.assertIn("IDs/order", result["failure_reason"])

    def test_truncated_worker_state_still_saves_failure(self):
        def corrupt(command, timeout, log_path, env):
            Path(log_path).write_text("simulated interrupted state write")
            (Path(log_path).parent/"worker_state.json").write_text('{"truncated":')
            return {"returncode": 0, "timed_out": False, "wall_seconds": .01}
        with patch("georecon.baselines.colmap.execute_worker", side_effect=corrupt):
            result = run_colmap(self.manifest, self.config, self.root/"bad_state")
        self.assertEqual(result["status"], "process_error")
        self.assertIn("cannot read worker state", result["failure_reason"])

    def test_existing_output_is_never_overwritten(self):
        out = self.root/"existing"
        out.mkdir()
        (out/"keep.txt").write_text("preserve")
        with self.assertRaises(ValueError):
            run_colmap(self.manifest, self.config, out)
        self.assertEqual((out/"keep.txt").read_text(), "preserve")

    @unittest.skipUnless(importlib.util.find_spec("pycolmap"), "requires separate official PyCOLMAP environment")
    def test_native_blank_scene_reports_no_reconstruction(self):
        output = self.root/"native_blank"
        process = subprocess.run([sys.executable, "-m", "georecon.cli", "run-colmap", "--manifest", str(self.manifest), "--config", str(self.config), "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 1, process.stderr)
        result = read_json(output/"prediction.json")
        self.assertEqual(result["status"], "no_reconstruction")
        self.assertTrue(all(r["camera"] is None for r in result["images"]))
