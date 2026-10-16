# 阶段 1：统一数据与最小评测链路

日期：2026-10-16（Asia/Shanghai）。用户指示“下一步”后开始阶段 1，并明确选择 DTU 标定与稠密几何优先路线。**本阶段已完成；没有开始阶段 2。**

## 实际完成与验收

| 范围 | 实际结果 |
|---|---|
| 最小可安装包与 CLI | `georecon-fusion` 0.1.0，可编辑安装成功；从 `/tmp` 导入成功，无须手动 PYTHONPATH；CPU 导入未加载 torch |
| 数据/相机/结果协议 | 版本 1，严格 JSON、输入顺序与哈希、注册/失败状态、独立 reference、缺失能力与未测量值显式 null |
| 数学逻辑 | 相机变换、投影/反投影、Z-depth、缩放/裁剪、相对位姿、两视图 DLT、正深度、退化检查、SE(3)/Sim(3) 对齐 |
| 合成加载与评测 | 4 相机固定 fixture，完整、扰动+部分注册、完全失败三种状态；全部执行，明确不是 Benchmark |
| 真实小场景 | DTU scan1，light=3，视图 001/002/003，原图 1600×1200；解码、SHA-256、顺序与 800×600 处理通过 |
| 相机标定核查 | 从官方 pos 文件分解 K/R/t，重建 P 的数值检查通过；与本机 OpenCV 独立分解结果交叉检查通过 |
| 自动检查 | 33 项 unittest 全部通过，实际日志 `tests.txt`；可编辑安装、pip check、CLI 加载/评测通过 |

实现提交：`b53f2c0`（阶段 1 的代码、配置、测试和协议说明）。验收产物保存该提交与 `source_tree_sha256`，后续报告提交不改变实现。保留的历史未跟踪文件会使 Git dirty 状态为 true；不将其伪装成 clean，源码摘要记录实际参与的包文件。数学测试在实现提交前运行，正式 CLI 冒烟在该提交后执行。

## 文件与职责

- `pyproject.toml`、`requirements-stage1-lock.txt`：仅 CPU 数据/几何层的安装与已测依赖；不沿用庞大的历史 GPU 环境锁文件。
- `src/georecon/data/`：manifest、结果校验、RGB 加载、DTU 小样本准备与有预算的 ZIP 按字节读取。
- `src/georecon/geometry/`：明确约定的 pinhole 几何及对齐，不是教学版 SfM 或 Baseline。
- `src/georecon/evaluation/poses.py`：独立真值评价，输出覆盖率、每对状态、角度误差、满足条件时的中心 ATE。
- `src/georecon/synthetic.py`、`cli.py`、`provenance.py`：固定 fixture、命令入口、源码/运行来源记录。
- `configs/datasets/dtu_smoke.json`、`configs/experiments/stage1_synthetic.json`：固定输入与参数，未写死本机路径。
- `tests/`：数学、协议、缺失/失败、命令行、DTU 布局及 HTTP range 行为的检查。
- [协议说明](../protocol.md)、本目录证据 JSON 和测试日志：审查与追溯。

原始图像、标定、官方 ReadMe、合成图像和完整运行结果留在被忽略的 `data/`、`outputs/`，不提交原始数据或大型产物。提交的是代码、配置、哈希与小型审计记录。

## DTU 数据获取与限制

来源是 [DTU 官方 MVS 页面](https://roboimagedata.compute.dtu.dk/?page_id=36) 链接的 [SampleSet.zip](https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip)，没有改用来历不明的镜像。归档实际长度 6,905,656,531 bytes；本阶段通过 HTTP Range 只读取目录及选定的 7 个成员：三张 Rectified scan1 图像、cal18 下三个 pos 文件、SampleSet ReadMe。

最终固定哈希复查那次准备传输 **8,936,447 bytes**，上限 32 MiB；首次获取后，进行了固定成员 SHA-256 的再次下载验证，另有少量目录/说明探测。没有下载整个 6.9 GB 包、其他场景、扫描点云、权重或模型。下载记录包含 ETag、归档长度、请求区间、成员名、CRC32、内容哈希和实际耗时。配置中的成员哈希来自本次获取后的固定记录，**不是发布者额外提供的独立校验签名**。

本地 ReadMe 将数据称为 citeware，要求引用 Jensen et al., “Large Scale Multi-view Stereopsis Evaluation”, CVPR 2014；它说明 Rectified 为径向畸变校正图像，供应的 P 矩阵用于对应扫描坐标系。保留原说明，不重新许可或发布数据。light=3 依官方说明选择；仅保留到 dev 作为加载冒烟，没有建立正式 Benchmark。

数据路径为 `data/dtu_smoke_pinned/`，首次未固定哈希的获取保留在 `data/dtu_smoke/` 用于追溯。最终原图 SHA-256、处理尺寸、标定代数检查与未运行状态见 [dtu_smoke_evidence.json](dtu_smoke_evidence.json)。读取 raw `pos_*.txt` 使用 RQ 分解，允许 P 的全局正负缩放；与 OpenCV `decomposeProjectionMatrix` 的 K/R/相机中心差异均小于 1e-8。

**上述检查验证加载和代数实现，不是三维重建精度。** 没有独立标注对应点来判定 DTU 原始像素原点约定，不把 P 分解再重建的低残差当独立正确性证据。参考相机单独存于 `reference.json`；manifest 不携带 GT K/位姿，不供未来方法静默获得额外信息。尚未下载几何真值，不运行稠密评价。

## 已执行的合成结果：不是 Benchmark

固定 seed=42、4 相机、40 个三维点。每张图像是真正生成并经过解码校验的 PNG。预测是人为构造的数学 fixture，**没有声称从图像估计相机**。

| 人工构造条件 | 覆盖 | 有效旋转图像对 / 全部对 | 平移方向误差均值 | Sim(3) 后中心 RMSE |
|---|---|---|---|---|
| 仅改变全局尺度/旋转/平移 | 4/4 | 6/6 | 约 1.42e-7 度（浮点误差） | 约 4.49e-16 合成米 |
| 对一相机中心加入配置指定扰动，缺失一相机 | 3/4 | 3/6 | 约 1.70586 度，仅已注册子集 | 约 0.0200482 合成米，仅已注册子集 |
| 人工完全失败 | 0/4 | 0/6 | null | null |

这只是证明实现能识别规范不变性、扰动与缺失；不能说明 COLMAP/VGGT 性能。全部逐对状态、对齐参数、参考单位和来源见 [synthetic_evidence.json](synthetic_evidence.json)。中心对齐要求至少三个非共线相机，两个或共线相机返回不可评估。平移反向误差按 180 度记录，不使用某些数据协议的无向歧义约定。

DTU 没有方法预测：实际评测入口读取 `prediction_not_run.json`，输出 `not_run`、0/3 注册、全部 3 对不可评估、误差 null。这是状态链路验证，不能当作“真实场景性能为零”。

## 运行命令

在项目根目录，用 Python 3.12 环境执行。为独立 CPU 开发创建新 venv，再安装以下依赖即可；本机实际验证的是现有科研 venv 中的安装，不宣称已验证空白机器及所有平台。

```bash
python -m pip install -r requirements-stage1-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m unittest discover -s tests -v

# 合成数据与端到端评价；输出目录不得已有 manifest
georecon synthetic-demo --config configs/experiments/stage1_synthetic.json --output outputs/stage1/my_synthetic_run

# 下载 DTU 小样本：只读选定成员，32 MiB 上限，不支持 Range 时直接拒绝
georecon prepare-dtu --config configs/datasets/dtu_smoke.json --output data/dtu_smoke_new
georecon inspect-scene data/dtu_smoke_new/manifest.json --output outputs/stage1/my_dtu_inspection.json

# 此阶段只有 not_run 预测；验证失败状态，不产出真实准确率
georecon evaluate-poses --prediction data/dtu_smoke_new/prediction_not_run.json --reference data/dtu_smoke_new/reference.json --output outputs/stage1/my_dtu_notrun.json
```

已有本机数据无需再次下载，可直接将命令中的 `data/dtu_smoke_new` 改为 `data/dtu_smoke_pinned` 做检查。本机命令入口位于 `/home/zchming/.venvs/visual-geometry/bin/georecon`，对应 Python 是同目录的 `python`。该绝对路径只用于运行记录，不在算法默认配置中。实际执行的命令、返回码、完整 stdout/stderr 见 [acceptance_commands.json](acceptance_commands.json)；测试命令及日志见 [test_run.json](test_run.json) 和 [tests.txt](tests.txt)。

## 已知限制与未完成事项

1. 只支持无畸变 PINHOLE；原始畸变、多种 COLMAP 相机模型和官方预处理适配尚未实现，不能静默丢畸变。
2. 深度、点云、轨迹、置信度只有可选描述协议，尚未实现数组内容校验与评价；null 不伪造数据。
3. 没有 COLMAP/VGGT adapter、官方 BA、特征匹配、PnP、完整 SfM、可靠性或融合；两视图 DLT 仅用于数学验证，未提前实施阶段 6。
4. 没有稠密扫描真值与官方几何评价、正式划分、AUC/场景排行/消融，没有性能提升结论。
5. 下载器要求支持精确 Range 的官方源；归档 ETag/长度或成员哈希变化会停止，需人工核查更新，不退化为大文件下载。失败后的局部数据保留，准备成功的输出目录拒绝覆盖。
6. 33 项测试均为 CPU、合成或本地 HTTP fixture；真实 DTU 的额外检查单独记录。OpenCV 只用于本机独立交叉核验，不是最小包运行依赖。
7. NumPy/Pillow 版本与当前机器固定；后续 VGGT 的旧依赖仍需独立环境。阶段 0 的 COLMAP/CUDA MVS 阻塞没有在本阶段解决。

## 下一阶段建议：现在不开始

建议阶段 2 先验证 COLMAP 安装和最小 SfM，再接 VGGT 原始推理与官方 BA。采用同一批图像清单，单独记录各方法预处理、原始结果和成本。

开始前仍需明确相机协议：主比较建议各方法自估内参；若用 DTU 标定 K，应作为显式 known-intrinsics 配置单独报告，不能混入主比较。该选择以及 Baseline 环境安装在下一次明确授权后处理。本阶段到此停止。
