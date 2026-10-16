# Stage 1 interchange protocol (schema_version=1)

The executable validators live in `src/georecon/data/protocol.py`. This is the first, deliberately small version, not the final protocol for every COLMAP/VGGT output. Stage 1 accepts only undistorted `PINHOLE` cameras. Other camera models must be converted explicitly or rejected; distortion is never silently discarded.

## Scene input manifest

A manifest contains `scene_id`, `kind` (`synthetic`/`real`), `split`, `seed`, `source`, `license_note`, `preprocessing`, `budget`, and an **ordered** nonempty `images` list. Each image specifies a unique `image_id`, a relative `path`, the SHA-256 of its original bytes, and integer `width`/`height`.

The loader checks file hashes, decodes actual RGB pixels, and checks dimensions. Paths are rooted at the manifest directory; absolute paths, `..`, and symlinks escaping that root are rejected. It does not glob or reorder inputs. `kind=synthetic` is never a real Benchmark result. `split=dev` for this DTU smoke scene is a development reservation, not a completed Benchmark split.

Preprocessing supports `identity` or `resize_then_crop` using Pillow bilinear resize. Let `sx = resized_width/original_width` and likewise `sy`. For zero-based pixel centers:

```text
u' = sx * (u + 0.5) - 0.5 - left
v' = sy * (v + 0.5) - 0.5 - top
A = [[sx, 0, (sx-1)/2-left],
     [0, sy, (sy-1)/2-top],
     [0, 0, 1]]
K' = A K
```

The actual integer resize and crop sizes are saved. This convention is specific to the stage 1 loader; an official baseline's different preprocessing must retain its own explicit mapping. DTU's supplied projection matrices are retained in their native pixel convention; the algebra check is not independent proof of pixel-origin correctness against feature ground truth.

The inference-facing manifest contains **no GT poses or point cloud**. The separately written `reference.json` is for evaluation only. A baseline adapter should not load it unless the user approves an explicitly known-intrinsics protocol.

## Result/reference document

Required identifiers: `scene_id`, `method_id`, `run_id`, `role` (`prediction` or `reference`), and `input_manifest_sha256`. The fingerprint binds the result to image bytes, order, preprocessing and other manifest settings. Both results must match on scene, fingerprint, image IDs/order and real/synthetic kind before comparison.

`coordinate_convention=opencv_world_to_camera_z_depth` means right/down/forward axes, `X_camera = R X_world + t`, and camera center `C = -R.T t`. `K` is upper triangular with positive focal lengths and last row `[0,0,1]`; `R` must be a proper rotation. Width and height refer to that camera's pixel coordinates. `world_unit` is `m`, `mm` or `arbitrary`; depths use camera Z, not ray length. Extrinsics use a 3×3 `R` and length-3 `t`, not an ambiguously named 4×4 pose.

Each image includes `registered`. A registered image has a validated camera. An unregistered image must have `camera=null` and a reason. The run status is one of:

| Status | Meaning |
|---|---|
| success | All inputs registered; does not imply geometrically accurate |
| partial | At least two but fewer than all images registered |
| no_reconstruction | No registered cameras and a failure reason |
| not_run | No execution; null errors must never become zero |
| timeout / oom / process_error | Failure retained; any recoverable cameras can be reported explicitly |

All non-success states need a reason. `provenance.data_kind` is explicit. `config`, `versions`, `logs`, `timing_seconds` and `memory_bytes` are present even when empty/unmeasured; unmeasured numbers are null, not invented zeros. Generated synthetic results record code commit, source-file digest, Git dirty state, versions and their configuration. Their generation wall time is not labeled model inference time.

Optional `artifacts.depth`, `pointcloud`, `tracks` and `confidence` are null when unavailable. A present artifact descriptor needs a relative path and `semantics`. Stage 1 does **not** implement file-content schemas or dense evaluation for those artifacts; adapters must add shape/dtype/unit/coordinate checks before using them in later stages. A non-null descriptor alone is not proof of usable content.

JSON is strict: NaN/Infinity are rejected and missing metrics use null.

## Pose evaluation

The CLI requires separate `--prediction` and `--reference` files. It evaluates all input unordered image pairs, explicitly retaining missing pairs. Pair geometry is `R21=R2 R1.T`, `t21=t2-R21 t1`. Rotation error is in degrees; translation-direction error is the oriented angle in `[0,180]`, so reversed direction is 180 degrees. A zero baseline yields a null direction error, even if rotation is evaluable. This is not automatically any dataset's published AUC convention.

Means/medians are labeled **registered subset**, alongside input pair counts, evaluated pair counts, registration ratio and failure state. No pose AUC or scene ranking is reported in stage 1. The center ATE is an RMSE after source-to-reference Sim(3), only with at least three noncollinear registered centers. Alignment parameters, camera IDs and per-camera distances are saved; units follow the reference. Two-camera or collinear fits return unavailable, not zero. This GT alignment is evaluation only; it is not an inference fusion transform.

For DTU, `prediction_not_run.json` supplies no method estimates. Evaluating it exercises unavailable-state handling and returns zero coverage plus null errors. Comparing calibration against itself would be an algebra check, never a reconstruction accuracy result.
