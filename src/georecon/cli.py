import argparse
import json
import sys
from .data.protocol import read_json, write_json
from .data.scene import load_scene
from .evaluation.poses import evaluate_poses


def main():
    parser = argparse.ArgumentParser(description="GeoRecon-Fusion stage 1 data/geometry checks")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("synthetic-demo", help="Synthetic checks, never real benchmark results")
    demo.add_argument("--config", required=True)
    demo.add_argument("--output", required=True)
    dtu = sub.add_parser("prepare-dtu", help="Download only 3 DTU views and calibration via bounded HTTP ranges")
    dtu.add_argument("--config", required=True)
    dtu.add_argument("--output", required=True)
    inspect = sub.add_parser("inspect-scene", help="Verify hashes, order, decode and preprocessing")
    inspect.add_argument("manifest")
    inspect.add_argument("--output")
    evaluate = sub.add_parser("evaluate-poses", help="Reference-only evaluation, separate from inference")
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "synthetic-demo":
            from .synthetic import run_demo
            result = run_demo(args.config, args.output)
        elif args.command == "prepare-dtu":
            from .data.dtu import prepare_dtu
            result = prepare_dtu(args.config, args.output)
        elif args.command == "inspect-scene":
            manifest, originals = load_scene(args.manifest)
            _, processed = load_scene(args.manifest, preprocess=True)
            result = {"scene_id": manifest["scene_id"], "kind": manifest["kind"], "image_ids": [r["image_id"] for r in manifest["images"]],
                      "original_shapes": [list(i.shape) for i in originals], "processed_shapes": [list(i.shape) for i in processed],
                      "checks": "hashes_dimensions_decode_preprocessing_passed", "baseline_status": "not_run"}
            if args.output:
                write_json(args.output, result)
        else:
            result = evaluate_poses(read_json(args.prediction), read_json(args.reference))
            write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, OSError) as exc:
        print(f"georecon: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
