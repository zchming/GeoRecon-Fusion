from pathlib import Path
import numpy as np
from PIL import Image
from .protocol import read_json, relative_file, sha256, validate_manifest


def load_scene(path, preprocess=False):
    """Verify exact image bytes and sizes; preserve manifest order, never glob-sort."""
    path = Path(path)
    manifest = validate_manifest(read_json(path))
    images = []
    for item in manifest["images"]:
        image_path = relative_file(path.parent, item["path"])
        if sha256(image_path) != item["sha256"]:
            raise ValueError(f"image hash mismatch: {item['image_id']}")
        with Image.open(image_path) as image:
            if image.size != (item["width"], item["height"]):
                raise ValueError(f"image dimension mismatch: {item['image_id']}")
            image = image.convert("RGB")
            if preprocess:
                config = manifest["preprocessing"]
                if config["operation"] == "resize_then_crop":
                    image = image.resize(config["resize_wh"], Image.Resampling.BILINEAR)
                    x, y, w, h = config["crop_xywh"]
                    if min(x, y) < 0 or min(w, h) <= 0 or x+w > image.width or y+h > image.height:
                        raise ValueError("crop outside image")
                    image = image.crop((x, y, x+w, y+h))
                elif config["operation"] != "identity":
                    raise ValueError("unsupported preprocessing operation")
            images.append(np.array(image))
    return manifest, images
