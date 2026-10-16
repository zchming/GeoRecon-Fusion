# GeoRecon-Fusion

**Geometry-Reliability-Aware Hybrid Multi-View 3D Reconstruction**

基于视觉几何基础模型与经典几何融合的可靠性驱动多视图三维重建系统。

研究问题：经典 SfM/MVS 与视觉几何基础模型分别何时失败？推理时无需真值的可靠性判断能否支持选择或融合，并改善精度、完整度、成功率与计算成本之间的权衡？融合有效与否尚待实验验证。

## 当前状态：阶段 3 的首轮 DTU 相机位姿 Benchmark 完成

已接入 COLMAP SfM、VGGT raw 和 VGGT 官方跟踪/BA，并完成阶段 2 产物与 GT 位姿评价复核。阶段 3 在运行前锁定 scan1 开发集、scan6 测试集及四种条件（正常五图、减少至三图、扩大视角间隔、固定模糊），完成 **24 次单次对比**，59 项 CPU 数学/接口检查通过。逐实例结果、失败、全部输入对分母的 AUC、完整耗时与显存均留档。**这是两场景的小型 pose Benchmark，不是完整 DTU 官方稠密评测；尚未开展可靠性分析或融合。**

- [阶段 1 报告、真实执行记录与限制](docs/stage1/report.md)
- [阶段 2 COLMAP 报告、结果与复现](docs/stage2_colmap/report.md)
- [阶段 2 VGGT raw 报告、坐标核对与结果](docs/stage2_vggt/report.md)
- [阶段 2 官方 BA 报告、完整耗时/显存与失败记录](docs/stage2_vggt_ba/report.md)
- [阶段 3 首轮结果、失败分析与复现](docs/stage3/report.md)
- [阶段 3 运行前冻结协议](docs/stage3/protocol.md)
- [相机、输入、结果与评价协议](docs/protocol.md)
- [阶段 0 环境审计与方案](docs/stage0/report.md)
- [完整项目要求与阶段边界](docs/project_brief.md)

## 安装与合成检查

使用 Python 3.12 的独立 CPU 环境，从项目根目录执行：

```bash
python -m pip install -r requirements-stage1-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m unittest discover -s tests -v
georecon synthetic-demo --config configs/experiments/stage1_synthetic.json --output outputs/my_synthetic_run
```

合成图像和相机是人为构造的 fixture，不属于 Benchmark，也不是从图像恢复相机的算法。包含完整、部分注册和完全失败三个案例；误差缺失时为 null，不填零。输出保留输入、配置、源码摘要、日志信息和逐图像对结果。

## DTU 小场景

```bash
georecon prepare-dtu --config configs/datasets/dtu_smoke.json --output data/dtu_smoke
georecon inspect-scene data/dtu_smoke/manifest.json
georecon evaluate-poses --prediction data/dtu_smoke/prediction_not_run.json --reference data/dtu_smoke/reference.json --output outputs/dtu_notrun.json
```

准备命令从官方 SampleSet 按字节读取 3 张图像、3 份标定和 ReadMe，单次实际传输约 8.94 MB，上限 32 MiB；源不支持精确 Range 时直接停止，不下载整个归档。已存在的准备目录拒绝覆盖，请检查已有数据或换新目录。DTU 数据要求引用 Jensen et al., CVPR 2014；源地址与固定成员哈希见配置。

以上 `prediction_not_run.json` 是阶段 1 创建的占位状态文件，只验证不可评估状态；实际 COLMAP 结果由下面的命令另行生成。GT 标定独立存于 reference，输入 manifest 不含 GT 相机。原始数据与大型运行产物不进入 Git。

## COLMAP 三图 SfM

使用单独的 CPU 环境，避免影响 VGGT 及旧科研环境：

```bash
python3.12 -m venv .venv-colmap
source .venv-colmap/bin/activate
python -m pip install -r requirements-colmap-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m unittest discover -s tests -v
georecon run-colmap --manifest data/dtu_smoke/manifest.json --config configs/methods/colmap_cpu_smoke.json --output outputs/my_colmap_run
georecon evaluate-poses --prediction outputs/my_colmap_run/prediction.json --reference data/dtu_smoke/reference.json --output outputs/my_colmap_run/evaluation.json
```

仅接入官方 PyCOLMAP 4.2.1 CPU SfM：SIFT、全匹配、增量建图及其内置 BA。由 COLMAP 自估共享 PINHOLE 内参，不读取 DTU 标定参与推理。输出恢复原图尺寸与零基像素中心，保留每张输入的注册状态、所有候选模型、原生日志和成本。失败时保存记录并返回非零退出码；已有输出目录拒绝覆盖。

最终开发样本的三对图像平均相对旋转误差约 0.621°、平移方向误差约 1.191°；Sim(3) 对齐三相机中心后的 ATE 为 1.383 mm。仅一个三图场景，不能据此宣称泛化性能或优于其他方法。[完整记录及限制](docs/stage2_colmap/report.md)。

## VGGT 三图原始推理

使用具有已验证 CUDA/BF16 运行库的独立命令环境；官方 checkout 和本地 checkpoint 通过参数提供，不自动下载或修改。环境版本见 `requirements-vggt-runtime-lock.txt`。本机采用只读共享既有 Torch 的 overlay，完整新机器安装尚未实测，详见报告。

```bash
# 以下变量指向你准备的固定源码和本地原始权重
GEOR_VGGT_REPO=experiment03_depth_reprojection/vggt_official
GEOR_VGGT_CHECKPOINT=experiment03_depth_reprojection/model.pt

georecon run-vggt --manifest data/dtu_smoke/manifest.json --config configs/methods/vggt_raw_smoke.json --official-repo "$GEOR_VGGT_REPO" --checkpoint "$GEOR_VGGT_CHECKPOINT" --output outputs/my_vggt_preflight --preflight-only
georecon run-vggt --manifest data/dtu_smoke/manifest.json --config configs/methods/vggt_raw_smoke.json --official-repo "$GEOR_VGGT_REPO" --checkpoint "$GEOR_VGGT_CHECKPOINT" --output outputs/my_vggt_run
georecon evaluate-poses --prediction outputs/my_vggt_run/prediction.json --reference data/dtu_smoke/reference.json --output outputs/my_vggt_run/evaluation.json
```

官方 crop 实际输入 518×392，与 COLMAP 的 800×600 不同；所有尺寸和复合变换明确记录。统一相机恢复原图坐标，原始数组保留模型网格，不过滤、不融合，不接 BA。VGGT 的“相机可用”不等于经典 SfM 注册成功，置信度也不是已校准概率。

该单场景 VGGT 平均旋转误差约 0.392°、平移方向误差约 0.722°，评价专用 Sim(3) 后中心 ATE 为 0.424 mm。GPU forward 约 0.866 s，包含加载/校验的 runner 总耗时约 14.645 s；不能用纯 forward 与 COLMAP 整条链路直接比较。详细成本、输出与限制见 [VGGT 报告](docs/stage2_vggt/report.md)。

## 代码结构

```text
configs/                 # 固定小场景与合成配置
src/georecon/
  data/                  # 协议、图像加载、DTU 小样本获取
  geometry/              # 相机、投影、三角化、SE(3)/Sim(3)
  evaluation/            # 独立参考位姿评价
  baselines/             # 官方 COLMAP 与 VGGT raw 适配
  cli.py                 # 命令入口
  synthetic.py           # 明确标记的合成验证
tests/                   # CPU 数学与接口检查
docs/                    # 阶段报告、协议与小型审计记录
data/                    # 本地数据，Git 忽略
outputs/                 # 本地运行产物，Git 忽略
```

统一协议支持无畸变 PINHOLE，BA 的 SIMPLE_PINHOLE 无损展开为 fx=fy。COLMAP 稀疏点云、VGGT 原始深度/点图/置信度以及 BA 的优化点云/原始跟踪数组已导出校验；没有将原始深度标成 BA 后稠密结果。CPU 几何、COLMAP、VGGT raw 与官方 BA 使用分开的依赖锁；PyCOLMAP 4.2.1 与 BA 的 3.10.0 分环境。BA 复现与显存适配见[说明](docs/stage2_vggt_ba/reproduce.md)。

## 后续工作：等待明确授权

2. 阶段 2 当前授权子步骤已完成；COLMAP MVS 仅在另行授权且 GPU 构建可用时开展。
3. 阶段 3 首轮小型位姿 Benchmark 已完成；扩展场景、资源配置版本或稠密评测需另行明确范围。
4. 几何可靠性分析。
5. 选择/融合与消融。
6. 教学版 SfM（仅按明确指示提前）。
7. 复现、可视化和成果整理。

每阶段仅在用户明确授权后开始。官方已有跟踪、导出和 BA 属于 Baseline，不作为原创贡献。本机此前的学习实验保留原目录，本次未整体发布；历史环境说明保留在 `docs/stage0/legacy_environment_setup.md`，不作为当前功能清单。
