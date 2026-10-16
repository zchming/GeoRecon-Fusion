import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image
from georecon.data.dtu import prepare_dtu
from georecon.data.protocol import read_json, write_json
from georecon.data.scene import load_scene


class FakeRemote(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self.url, self.etag, self.size = url, '"fixture"', len(payload)
        self.transferred, self.ranges = 0, []


class DtuLoaderTests(unittest.TestCase):
    def test_official_layout_mapping_and_not_run_evaluation(self):
        # Synthetic local ZIP exercises DTU integration without internet or a real benchmark.
        image = io.BytesIO()
        Image.new("RGB", (16, 12)).save(image, format="PNG")
        K = np.array([[20., 0, 8], [0, 20, 6], [0, 0, 1]])
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("SampleSet/ReadMe.txt", "Synthetic fixture, not DTU data")
            for index, center in enumerate(([0., 0, 0], [1., 0, 0], [0., 1, 0]), 1):
                z.writestr(f"SampleSet/MVS Data/Rectified/scan1/rect_{index:03d}_3_r5000.png", image.getvalue())
                stream = io.StringIO()
                np.savetxt(stream, K @ np.column_stack((np.eye(3), -np.array(center))))
                z.writestr(f"SampleSet/MVS Data/Calibration/cal18/pos_{index:03d}.txt", stream.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"archive_url": "https://fixture.invalid/test.zip", "max_transfer_bytes": 100000,
                      "source_page": "https://fixture.invalid", "view_ids": [1, 2, 3], "scan": 1, "light": 3,
                      "resize_wh": [8, 6], "crop_xywh": [0, 0, 8, 6], "scene_id": "fixture", "seed": 42, "split": "dev"}
            write_json(root/"config.json", config)
            with patch("georecon.data.dtu.RangeZip", side_effect=lambda url, budget: FakeRemote(archive.getvalue(), url)):
                result = prepare_dtu(root/"config.json", root/"data")
            self.assertEqual(result["images"], 3)
            self.assertEqual(result["baseline_status"], "not_run")
            _, processed = load_scene(root/"data/manifest.json", preprocess=True)
            self.assertEqual(processed[0].shape, (6, 8, 3))
            reference = read_json(root/"data/reference.json")
            from georecon.evaluation.poses import evaluate_poses
            report = evaluate_poses(read_json(root/"data/prediction_not_run.json"), reference)
            self.assertEqual(report["run_status"], "not_run")
            self.assertEqual(report["input_pairs"], 3)
            self.assertIsNone(report["rotation_deg_registered_subset"]["mean"])
