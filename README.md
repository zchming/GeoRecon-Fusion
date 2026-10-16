# GeoRecon-Fusion

**Geometry-Reliability-Aware Hybrid Multi-View 3D Reconstruction**

基于视觉几何基础模型与经典几何融合的可靠性驱动多视图三维重建系统。

研究问题：经典 SfM/MVS 与视觉几何基础模型分别何时失败？推理时无需真值的可靠性判断能否支持选择或融合，并改善精度、完整度、成功率与计算成本之间的权衡？融合有效与否尚待实验验证。

## 当前状态：阶段 1 完成

已实现 CPU 数据/相机协议、最小命令行、几何数学逻辑和位姿评价；33 项检查通过。已获取 DTU scan1 的 3 张图像和标定，完成加载与代数核查。**尚未运行 COLMAP/VGGT Baseline，没有真实重建精度或融合性能结果。**

- [阶段 1 报告、真实执行记录与限制](docs/stage1/report.md)
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

真实场景目前只有 `not_run` 预测，以上评测验证不可评估状态，不代表真实方法的精度或成功率。GT 标定独立存于 reference，输入 manifest 不含 GT 相机。原始数据与大型运行产物不进入 Git。

## 代码结构

```text
configs/                 # 固定小场景与合成配置
src/georecon/
  data/                  # 协议、图像加载、DTU 小样本获取
  geometry/              # 相机、投影、三角化、SE(3)/Sim(3)
  evaluation/            # 独立参考位姿评价
  cli.py                 # 命令入口
  synthetic.py           # 明确标记的合成验证
tests/                   # CPU 数学与接口检查
docs/                    # 阶段报告、协议与小型审计记录
data/                    # 本地数据，Git 忽略
outputs/                 # 本地运行产物，Git 忽略
```

只支持无畸变 PINHOLE；其他相机模型明确拒绝。可选深度/点云等产物尚未接入。当前依赖锁只适用于 CPU 数据/几何层，不是 VGGT/COLMAP 环境配置。

## 后续阶段：均未开始

2. COLMAP、VGGT 和官方 BA 的小样本 Baseline。
3. 固定 Benchmark 与失败分析。
4. 几何可靠性分析。
5. 选择/融合与消融。
6. 教学版 SfM（仅按明确指示提前）。
7. 复现、可视化和成果整理。

每阶段仅在用户明确授权后开始。官方已有跟踪、导出和 BA 属于 Baseline，不作为原创贡献。本机此前的学习实验保留原目录，本次未整体发布；历史环境说明保留在 `docs/stage0/legacy_environment_setup.md`，不作为当前功能清单。
