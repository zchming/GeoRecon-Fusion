import argparse
import json
import sys
from .data.protocol import read_json, write_json
from .data.scene import load_scene
from .evaluation.poses import evaluate_poses


def main():
    parser = argparse.ArgumentParser(description="GeoRecon-Fusion geometry checks and explicit baseline runs")
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
    colmap = sub.add_parser("run-colmap", help="Official CPU COLMAP SfM, estimated intrinsics, no reference access")
    colmap.add_argument("--manifest", required=True)
    colmap.add_argument("--config", required=True)
    colmap.add_argument("--output", required=True)
    vggt = sub.add_parser("run-vggt", help="Pinned local VGGT raw forward, no BA or download")
    vggt.add_argument("--manifest", required=True)
    vggt.add_argument("--config", required=True)
    vggt.add_argument("--official-repo", required=True)
    vggt.add_argument("--checkpoint", required=True)
    vggt.add_argument("--output", required=True)
    vggt.add_argument("--preflight-only", action="store_true", help="Verify official preprocessing before any model inference")
    ba = sub.add_parser("run-vggt-ba", help="Pinned official VGGT BA, offline weights, same three-view smoke input")
    for name in ["manifest", "config", "official-repo", "checkpoint", "assets", "dinov2-repo", "lightglue-repo", "output"]:
        ba.add_argument("--"+name, required=True)
    reliability = sub.add_parser("analyze-reliability", help="GT-free geometric diagnostics only; no selection or fusion")
    for name in ["bundle", "config", "output"]:
        reliability.add_argument("--"+name, required=True)
    selection = sub.add_parser("select-scene", help="Frozen whole-scene selection/fallback only; no fusion")
    for name in ["bundle", "policy", "output"]:
        selection.add_argument("--"+name, required=True)
    teaching = sub.add_parser("run-teaching-sfm", help="Stage6 calibrated two-view SIFT/essential/DLT teaching chain, no BA")
    for name in ["input", "config", "output"]:
        teaching.add_argument("--"+name, required=True)
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
        elif args.command == "run-colmap":
            from .baselines.colmap import run_colmap
            result = run_colmap(args.manifest, args.config, args.output)
        elif args.command == "run-vggt":
            from .baselines.vggt import run_vggt
            result = run_vggt(args.manifest, args.config, args.official_repo, args.checkpoint, args.output, args.preflight_only)
        elif args.command == "run-vggt-ba":
            from .baselines.vggt_ba import run_vggt_ba
            result = run_vggt_ba(args.manifest, args.config, args.official_repo, args.checkpoint, args.assets, args.dinov2_repo, args.lightglue_repo, args.output)
        elif args.command == "analyze-reliability":
            from .reliability.analyze import run
            result = run(args.bundle, args.config, args.output)
        elif args.command == "select-scene":
            from .selection.run import run
            result = run(args.bundle, args.policy, args.output)
        elif args.command == "run-teaching-sfm":
            from .teaching.run import run
            result = run(args.input, args.config, args.output)
        else:
            result = evaluate_poses(read_json(args.prediction), read_json(args.reference))
            write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command in {"run-colmap", "run-vggt", "run-vggt-ba", "run-teaching-sfm"} and result["status"] not in {"success", "partial"} and not (args.command == "run-vggt" and args.preflight_only and result["status"] == "not_run"):
            raise SystemExit(1)
    except (ValueError, KeyError, OSError) as exc:
        print(f"georecon: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
