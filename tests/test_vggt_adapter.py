import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
from georecon.baselines.vggt import crop_geometry, independent_crop, convert_camera, validate_config, validate_raw_arrays, verify_repository, run_vggt
from georecon.data.protocol import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def arrays(frames=3, height=14, width=14):
    return {"pose_enc": np.ones((frames, 9), np.float32), "extrinsics": np.tile(np.c_[np.eye(3), np.zeros(3)], (frames, 1, 1)).astype(np.float32),
            "intrinsics": np.tile(np.eye(3), (frames, 1, 1)).astype(np.float32), "depth": np.ones((frames, height, width, 1), np.float32),
            "depth_conf": np.ones((frames, height, width), np.float32), "world_points": np.ones((frames, height, width, 3), np.float32), "world_points_conf": np.ones((frames, height, width), np.float32)}


class VggtAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_official_crop_rounding_and_portrait_crop(self):
        landscape = crop_geometry(800, 600)
        self.assertEqual(landscape["resize_wh"], [518, 392])
        self.assertEqual(landscape["crop_xywh"], [0, 0, 518, 392])
        assert_allclose(landscape["pixel_transform"], [[518/800, 0, (518/800-1)/2], [0, 392/600, (392/600-1)/2], [0, 0, 1]])
        self.assertEqual(crop_geometry(1000, 2000)["crop_xywh"], [0, 259, 518, 518])
        with self.assertRaises(ValueError):
            crop_geometry(10000, 1)

    def test_independent_rgb_pixels_and_crop(self):
        image = Image.new("RGB", (800, 600), (255, 0, 128))
        tensor = independent_crop(image)
        self.assertEqual(tensor.shape, (3, 392, 518))
        self.assertEqual(tensor.dtype, np.float32)
        assert_allclose(tensor[:, 0, 0], [1, 0, 128/255], atol=1e-7)

    def test_integer_pixel_origin_does_not_apply_colmap_half_pixel(self):
        K = np.array([[500., 0, 259], [0, 510, 196], [0, 0, 1]])
        E = np.c_[np.eye(3), [1, 2, 3]]
        A = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])
        camera = convert_camera(K, E, [1036, 784], A)
        assert_allclose(camera.K, [[1000, 0, 518.5], [0, 1020, 392.5], [0, 0, 1]])
        assert_allclose(camera.center, [-1, -2, -3])
        p = np.array([[0., 0, 5]])
        uv = camera.project(p)[0][0]
        direct = K @ (E @ np.r_[p[0], 1])
        assert_allclose((A @ np.r_[uv, 1])[:2], direct[:2]/direct[2])

    def test_raw_shapes_dtype_nonfinite_and_nonpositive_depth(self):
        path = self.root/"raw.npz"
        good = arrays()
        np.savez(path, **good)
        self.assertIn("depth", validate_raw_arrays(path, 3, 14, 14))
        for name, value in [("depth", np.zeros_like(good["depth"])), ("world_points", good["world_points"].astype(np.float64)), ("depth_conf", np.full_like(good["depth_conf"], np.nan)), ("pose_enc", np.zeros((2, 9), np.float32))]:
            bad = {**good, name: value}
            np.savez(path, **bad)
            with self.assertRaises(ValueError):
                validate_raw_arrays(path, 3, 14, 14)

    def test_no_queries_ba_or_unpinned_source(self):
        config = read_json(ROOT/"configs/methods/vggt_raw_smoke.json")
        for key, value in [("query_points", [[1, 2]]), ("postprocessing", "ba"), ("official_commit", "main"), ("timeout_seconds", float("inf"))]:
            invalid = {**config, key: value}
            with self.assertRaises(ValueError):
                validate_config(invalid)

    def test_source_git_commit_and_cleanliness_are_enforced(self):
        repo = self.root/"repo"
        repo.mkdir()
        for name in ["vggt/models/vggt.py", "vggt/models/aggregator.py", "vggt/utils/load_fn.py", "vggt/utils/pose_enc.py", "vggt/utils/geometry.py", "requirements.txt", "LICENSE.txt"]:
            p = repo/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("synthetic repository fixture; not official source")
        subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "synthetic fixture"], capture_output=True, check=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(verify_repository(repo, commit)["commit"], commit)
        with self.assertRaises(ValueError):
            verify_repository(repo, "0"*40)
        (repo/"LICENSE.txt").write_text("modified")
        with self.assertRaises(ValueError):
            verify_repository(repo, commit)

    def _manifest(self):
        p = self.root/"scene"
        p.mkdir()
        Image.new("RGB", (800, 600), "black").save(p/"image.png")
        write_json(p/"manifest.json", {"schema_version": 1, "kind": "synthetic", "scene_id": "vggt_fixture", "split": "smoke", "seed": 42, "source": "fixture", "license_note": "generated", "preprocessing": {"operation": "identity"}, "budget": {},
                                     "images": [{"image_id": "one", "path": "image.png", "sha256": sha256(p/"image.png"), "width": 800, "height": 600}]})
        (p/"reference.json").write_text("invalid reference sentinel: never load GT")
        return p/"manifest.json"

    def test_preflight_success_has_no_geometry_or_fake_metrics(self):
        manifest = self._manifest()
        def fake(command, timeout, log, env):
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            output = Path(log).parent
            write_json(output/"worker_state.json", {"status": "preflight_complete", "input_shape": [1, 3, 392, 518]})
            write_json(output/"preflight.json", {"input_shape": [1, 3, 392, 518], "independent_loader_max_abs": 0})
            return {"returncode": 0, "timed_out": False, "wall_seconds": .01}
        with patch("georecon.baselines.vggt.verify_repository", return_value={"commit": "synthetic"}), patch("georecon.baselines.vggt.execute_worker", side_effect=fake):
            result = run_vggt(manifest, ROOT/"configs/methods/vggt_raw_smoke.json", self.root, self.root/"nonexistent.pt", self.root/"out", True)
        self.assertEqual(result["status"], "not_run")
        self.assertTrue(all(value is None for value in result["artifacts"].values()))
        self.assertIsNone(result["images"][0]["camera"])

    def test_missing_preflight_cannot_report_success(self):
        manifest = self._manifest()
        def fake(command, timeout, log, env):
            write_json(Path(log).parent/"worker_state.json", {"status": "preflight_complete", "input_shape": [1, 3, 392, 518]})
            return {"returncode": 0, "timed_out": False, "wall_seconds": .01}
        with patch("georecon.baselines.vggt.verify_repository", return_value={}), patch("georecon.baselines.vggt.execute_worker", side_effect=fake):
            result = run_vggt(manifest, ROOT/"configs/methods/vggt_raw_smoke.json", self.root, self.root/"fake.pt", self.root/"bad", True)
        self.assertEqual(result["status"], "process_error")

    def test_oom_timeout_and_launch_failure_are_preserved(self):
        manifest = self._manifest()
        for kind in ("oom", "timeout", "launch"):
            def fake(command, timeout, log, env):
                if kind == "launch":
                    raise OSError("synthetic launch failure")
                write_json(Path(log).parent/"worker_state.json", {"error": "synthetic failure", "failure_kind": "oom"})
                return {"returncode": 1, "timed_out": kind == "timeout", "wall_seconds": .01}
            with patch("georecon.baselines.vggt.verify_repository", return_value={}), patch("georecon.baselines.vggt.execute_worker", side_effect=fake):
                result = run_vggt(manifest, ROOT/"configs/methods/vggt_raw_smoke.json", self.root, self.root/"fake.pt", self.root/kind)
            self.assertEqual(result["status"], kind if kind != "launch" else "process_error")
