import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from georecon.data.protocol import read_json, validate_manifest, validate_result, relative_file, write_json
from georecon.data.scene import load_scene
from georecon.evaluation.poses import evaluate_poses
from georecon.synthetic import run_demo

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        run_demo(ROOT/"configs/experiments/stage1_synthetic.json", self.root)
        self.reference = read_json(self.root/"reference.json")
        self.prediction = read_json(self.root/"prediction_exact_sim3.json")

    def test_exact_gauge_transform_and_real_order(self):
        report = evaluate_poses(self.prediction, self.reference)
        self.assertEqual(report["registration_ratio"], 1)
        self.assertEqual(report["input_pairs"], 6)
        self.assertLess(report["camera_center_ate"]["rmse"], 1e-12)
        self.assertLess(report["translation_direction_deg_registered_subset"]["mean"], 1e-5)
        manifest, images = load_scene(self.root/"manifest.json")
        self.assertEqual([r["image_id"] for r in manifest["images"]], ["000", "001", "002", "003"])
        self.assertEqual(images[0].shape, (120, 160, 3))

    def test_partial_keeps_missing_pairs(self):
        result = evaluate_poses(read_json(self.root/"prediction_perturbed_partial.json"), self.reference)
        self.assertEqual(result["registration_ratio"], 0.75)
        self.assertEqual(result["input_pairs"], 6)
        self.assertEqual(result["rotation_evaluated_pairs"], 3)
        self.assertEqual(sum(p["status"] == "unavailable" for p in result["pairs"]), 3)
        self.assertGreater(result["translation_direction_deg_registered_subset"]["mean"], 0.1)

    def test_failure_not_zero_error(self):
        result = evaluate_poses(read_json(self.root/"prediction_no_reconstruction.json"), self.reference)
        self.assertEqual(result["registered_images"], 0)
        self.assertEqual(len(result["pairs"]), 6)
        self.assertIsNone(result["rotation_deg_registered_subset"]["mean"])
        self.assertIsNone(result["camera_center_ate"]["rmse"])

    def test_no_two_camera_ate(self):
        result = copy.deepcopy(self.prediction)
        for record in result["images"][2:]:
            record.update(registered=False, camera=None, reason="missing")
        result.update(status="partial", failure_reason="only two registered")
        self.assertEqual(evaluate_poses(result, self.reference)["camera_center_ate"]["status"], "unavailable")

    def test_zero_baseline_rotation_only(self):
        reference, result = copy.deepcopy(self.reference), copy.deepcopy(self.prediction)
        for record in reference["images"]:
            record["camera"]["t"] = [0, 0, 0]
        for record in result["images"]:
            record["camera"]["t"] = [0, 0, 0]
        evaluation = evaluate_poses(result, reference)
        self.assertEqual(evaluation["translation_evaluated_pairs"], 0)
        self.assertEqual(evaluation["rotation_evaluated_pairs"], 6)

    def test_success_cannot_hide_missing_camera(self):
        result = copy.deepcopy(self.prediction)
        result["images"][0].update(registered=False, camera=None, reason="failed")
        with self.assertRaises(ValueError):
            validate_result(result)

    def test_unregistered_geometry_and_duplicate_ids_rejected(self):
        for mutation in ("fabricated_camera", "duplicate_id"):
            result = copy.deepcopy(self.prediction)
            if mutation == "fabricated_camera":
                result["images"][0]["registered"] = False
            else:
                result["images"][1]["image_id"] = result["images"][0]["image_id"]
            with self.assertRaises(ValueError):
                validate_result(result)

    def test_convention_intrinsics_and_negative_timing_rejected(self):
        for mutation in ("convention", "camera", "time"):
            result = copy.deepcopy(self.prediction)
            if mutation == "convention":
                result["coordinate_convention"] = "camera_to_world"
            elif mutation == "camera":
                result["images"][0]["camera"]["K"][0][0] = -1
            else:
                result["timing_seconds"]["total"] = -1
            with self.assertRaises(ValueError):
                validate_result(result)

    def test_scene_manifest_order_and_roles_must_match(self):
        for mutation in ("scene", "fingerprint", "order", "role"):
            result = copy.deepcopy(self.prediction)
            if mutation == "scene":
                result["scene_id"] = "wrong"
            elif mutation == "fingerprint":
                result["input_manifest_sha256"] = "0"*64
            elif mutation == "order":
                result["images"].reverse()
            else:
                result["role"] = "reference"
            with self.assertRaises(ValueError):
                evaluate_poses(result, self.reference)

    def test_synthetic_real_mixing_and_invalid_preprocessing_rejected(self):
        result = copy.deepcopy(self.prediction)
        result["provenance"]["data_kind"] = "real"
        with self.assertRaises(ValueError):
            evaluate_poses(result, self.reference)
        manifest = read_json(self.root/"manifest.json")
        manifest["preprocessing"] = {"operation": "resize_then_crop", "resize_wh": [80, 60], "crop_xywh": [60, 0, 30, 60]}
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_images_hash_and_dimensions_verified(self):
        path = self.root/"manifest.json"
        manifest = read_json(path)
        manifest["images"][0]["width"] += 1
        write_json(path, manifest)
        with self.assertRaises(ValueError):
            load_scene(path)
        manifest["images"][0]["width"] -= 1
        manifest["images"][0]["sha256"] = "0"*64
        write_json(path, manifest)
        with self.assertRaises(ValueError):
            load_scene(path)

    def test_image_resize_crop_executes(self):
        path = self.root/"manifest.json"
        manifest = read_json(path)
        manifest["preprocessing"] = {"operation": "resize_then_crop", "resize_wh": [80, 60], "crop_xywh": [5, 3, 60, 50]}
        write_json(path, manifest)
        _, images = load_scene(path, preprocess=True)
        self.assertEqual(images[0].shape, (50, 60, 3))

    def test_path_traversal_symlink_and_duplicate_manifest(self):
        for path in ("../escape", "/tmp/escape"):
            with self.assertRaises(ValueError):
                relative_file(self.root, path)
        (self.root/"escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            relative_file(self.root, "escape/file")
        manifest = read_json(self.root/"manifest.json")
        manifest["images"][1]["image_id"] = manifest["images"][0]["image_id"]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_nonstandard_json_and_repeat_run_rejected(self):
        path = self.root/"bad.json"
        path.write_text('{"value":NaN}')
        with self.assertRaises(ValueError):
            read_json(path)
        with self.assertRaises(ValueError):
            run_demo(ROOT/"configs/experiments/stage1_synthetic.json", self.root)

    def test_cli_evaluate_and_invalid_input_exit(self):
        output = self.root/"cli.json"
        command = [sys.executable, "-m", "georecon.cli", "evaluate-poses", "--prediction", str(self.root/"prediction_exact_sim3.json"),
                   "--reference", str(self.root/"reference.json"), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(read_json(output)["registration_ratio"], 1)
        command[command.index("--prediction")+1] = str(self.root/"missing.json")
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
