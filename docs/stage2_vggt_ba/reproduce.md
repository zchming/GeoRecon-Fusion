# 阶段 2 官方 BA 复现

仅复现 DTU scan1 light3 的 001/002/003 三图，沿用 `data/dtu_smoke_pinned/manifest.json`，不扩展场景。输入与权重不进入 Git；数据准备见上一阶段报告。以下命令均从仓库根目录运行。

## 环境与源码

已验证环境为 Python 3.12.3、Torch 2.11.0+cu128、torchvision 0.26.0+cu128、NumPy 2.5.2。BA 的 PyCOLMAP 3.10.0 与既有 COLMAP baseline 的 4.2.1 **必须分环境**。本机使用 `/home/zchming/.venvs/georecon-vggt-ba`，通过 `.pth` 只读继承已有科学计算环境和 raw VGGT 环境；它不是完全独立安装的一整套 Torch。未修改上述已有环境。完整实测包清单见 `environment.json`。

新机器可创建独立环境并安装以下锁定列表；这是可复现安装说明，不代表本次重新下载了 CUDA 大包：

```bash
python3 -m venv .venv-vggt-ba
.venv-vggt-ba/bin/python -m pip install -r requirements-vggt-runtime-lock.txt -r requirements-vggt-ba-runtime-lock.txt
.venv-vggt-ba/bin/python -m pip install --no-deps -e .
```

已有官方 VGGT checkout 必须干净且固定在 `a288dd0f14786c93483e45524328726ab7b1b4ce`；既有 `experiment03_depth_reprojection/model.pt` 的 SHA 必须匹配方法配置。新增两个源码仓库如下，均留在被忽略的 `data` 下：

```bash
git clone https://github.com/jytime/LightGlue.git data/third_party/LightGlue
git -C data/third_party/LightGlue checkout 2f23ca2ea9638cecad7f7220795210fc6b8353c3
git clone https://github.com/facebookresearch/dinov2.git data/third_party/dinov2
git -C data/third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
.venv-vggt-ba/bin/python -m pip install --no-deps -e data/third_party/LightGlue
.venv-vggt-ba/bin/python -m pip check
python3 scripts/prepare_vggt_ba_assets.py --config configs/assets/vggt_ba.json --output data/vggt_ba_assets
```

首次下载已记录哈希，现在配置均已固定，不需要 `--record-first-use`。脚本不覆盖不匹配的文件，单份权重下载限 1 GiB。推理期间所有模型 URL 被白名单映射到已校验的本地文件，DINO Hub 映射到固定版本本地源码；未知请求直接失败。

## 验收运行

本机实际使用的解释器与下列路径不同，仅将 `.venv-vggt-ba/bin/python` 换成 `/home/zchming/.venvs/georecon-vggt-ba/bin/python`。输出目录必须是新的空目录。

```bash
.venv-vggt-ba/bin/python -m georecon.cli run-vggt-ba \
  --manifest data/dtu_smoke_pinned/manifest.json \
  --config configs/methods/vggt_ba_12gb_smoke.json \
  --official-repo experiment03_depth_reprojection/vggt_official \
  --checkpoint experiment03_depth_reprojection/model.pt \
  --assets data/vggt_ba_assets \
  --dinov2-repo data/third_party/dinov2 \
  --lightglue-repo data/third_party/LightGlue \
  --output outputs/stage2_vggt_ba/acceptance
.venv-vggt-ba/bin/python -m georecon.cli evaluate-poses \
  --prediction outputs/stage2_vggt_ba/acceptance/prediction.json \
  --reference data/dtu_smoke_pinned/reference.json \
  --output outputs/stage2_vggt_ba/acceptance/evaluation.json
```

另对同次 forward 保存的 `pre_ba_prediction.json` 用相同评价命令生成 `pre_ba_evaluation.json`。GT 只由评价器读取。`acceptance_commands.json` 保存实际执行的命令数组及评价耗时。数学/接口回归在原 PyCOLMAP 4.2.1 环境执行 `python -m unittest discover -s tests -q`；PyCOLMAP 3.10 原生接口与 GPU 跟踪/BA 由实际验收运行覆盖。

`vggt_ba_smoke.json` 保持官方模型常驻的调度；该调度本机已 OOM，不建议重复执行。`vggt_ba_12gb_smoke.json` 仅将不再使用的 VGGT 模型转到 meta 释放存储；不改图片、权重、分辨率、精度、查询点数、精细跟踪或 Ceres 参数。该配置是显式的本地显存调度适配，不声称逐行原样运行官方入口。
