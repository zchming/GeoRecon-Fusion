# 阶段 2 子步骤：COLMAP 三图 SfM 接入与验证

日期：2026-10-16（Asia/Shanghai）。本次用户明确授权 COLMAP 小样本流程、验证和 GitHub 推送，并确认“COLMAP 自估内参，DTU 标定只用于评价”。**此子步骤已完成，阶段 2 整体尚未完成。** 未接入 VGGT、VGGT 官方 BA、MVS、Benchmark、可靠性或融合。

## 1. 实际交付与验收

| 验收范围 | 实际结果 |
|---|---|
| 官方实现与环境 | 独立 venv 安装官方 PyCOLMAP 4.2.1，CPU wheel、has_cuda=False；没有改动原科研环境、安装系统 CLI 或驱动 |
| SfM 接入 | SIFT 特征、全匹配、增量建图及 COLMAP 原生局部/全局 BA；不是新实现的 SfM 或原创算法 |
| 输入固定 | 沿用阶段 1 的 DTU scan1/light3/001、002、003，原图 1600×1200，按原 manifest 处理为 800×600；没有下载新数据/权重 |
| 内参公平性 | 单共享 PINHOLE，从图像尺寸初始化，焦距自行优化；数据库 prior_focal_length=False，无 GT K/位姿输入 |
| 输出协议 | 完整保留输入 ID/顺序、注册/失败状态；将相机恢复到原图分辨率和项目像素约定；导出经过内容校验的稀疏点云 |
| 坐标与文件验证 | 原生模型落盘后重新读取，R/t/相机中心验证通过；点 ID、XYZ、RGB 与原生模型逐项一致；投影转换最大差异约 2.27e-13 处理后像素 |
| 失败处理 | 超时终止进程组；模拟 OOM/异常/损坏输出/截断状态文件均保留失败记录；实际空白合成图像返回无法重建，CLI 返回 1 |
| 检查 | 隔离环境中 43 项测试全部通过；安装与 pip check 通过；真实三图 SfM 与独立评价均返回 0 |

源代码实现提交为 `6569611`，随后在 `1d0bc8a` 修复中断写状态文件时的记录问题。最终验收运行对应 **`1d0bc8a`**，同时保存实际源码摘要。历史未跟踪的学习实验会使 working_tree_dirty 为 true；没有伪称 clean。文档提交不修改这些实现。

## 2. 官方来源和安装选择

采用 [COLMAP 官方 PyCOLMAP 接口](https://colmap.github.io/pycolmap/index.html)，接口调用核对了 [4.2.1 官方例程](https://github.com/colmap/colmap/blob/4.2.1/python/examples/example.py)，并对安装后的函数签名和 options 实际检查。对应发行 tag 4.2.1，官方源码 commit `bd1fcf654d2dd8fefa1466999c190a246f83f4b9`；使用的是官方发布 wheel，不声称本机从该源码自行构建。

完整 wheel 文件名、PyPI 发布哈希、版本、例程和许可证入口在 [official_sources.json](official_sources.json)。已安装 wheel 含 COPYING.txt，其哈希单独记录。[COLMAP 4.2.1 许可](https://github.com/colmap/colmap/blob/4.2.1/COPYING.txt) 对库本身采用 new BSD，依赖另有许可；本项目不重新分发二进制。

本机独立解释器：`/home/zchming/.venvs/georecon-colmap/bin/python`。Python 3.12.3、NumPy 2.5.2、Pillow 12.3.0、PyCOLMAP 4.2.1；CPU 为 i7-13700KF，WSL 可见 24 逻辑 CPU，本次配置线程上限为 8。完整 CPU、内存、依赖与 pip check 输出在 [environment.json](environment.json)。CPU wheel 无 CUDA 支持，不代表本机 PyTorch GPU 环境不可用；本次也不宣称 MVS 已就绪。

## 3. 方法与输入配置

方法配置为 `configs/methods/colmap_cpu_smoke.json`，不是根据 GT 评价误差挑选的配置：

- seed=42，CPU extraction/matching/mapper BA，线程上限 8，SIFT max_num_features=8192。
- 处理后图像 800×600，无额外 COLMAP 图像缩放。共享 PINHOLE 是同一 DTU 相机的小样本假设，不作为通用多相机配置。
- 初始焦距因子 1.2，对应处理后 fx=fy=960、cx=400、cy=300；不读取 EXIF 或标定初始化。PNG 重编码保留 RGB 像素但不复制 EXIF。
- 焦距按官方 mapper 默认优化，主点按默认固定；完整其余有效 options 保存于每次 worker_state，而不凭记忆重写默认值。
- min_model_size 从官方默认 10 显式改为 3，以适配三图验证；没有更改 RANSAC/误差阈值来迎合 GT。
- 整个 worker 上限 1200 s，包含提取、匹配和建图；输出目录不覆盖。

原始输入 manifest 的 `budget.baseline_runs=0` 是阶段 1 的历史准备记录，未静默修改其哈希；本次执行预算以方法配置与逐次运行记录为准。其 split=dev 只标记开发冒烟，没有锁定正式 Benchmark。

若 COLMAP 返回多个独立模型，保存全部模型，按“注册数最多、点数最多、model ID 最小”的固定规则选择一个，不使用 GT，也不合并不同坐标系来虚增覆盖。当前只有一个模型。

## 4. 坐标与导出验证

[COLMAP 官方像素约定](https://colmap.github.io/database.html) 是左上像素中心 `(0.5,0.5)`；项目的 OpenCV 数学协议用 `(0,0)`。先将原生 K 的 cx/cy 减 0.5，再用阶段 1 已保存的 resize/crop 矩阵 A 逆变换：`K_original = inv(A) @ shift(-0.5,-0.5) @ K_native`。

R/t 仍是 camera-from-world，未反转外参；相机中心 `-R.T t` 与原生 projection_center 对照。原生 keypoint 坐标保留在数据库，不改写；内部重投影验证在原生坐标计算，另检查导出到原图、再回处理后原生坐标的结果一致。

最终验证覆盖 2071 个点观测，全部为正深度；导出投影差异最大约 2.27e-13 px。稀疏 NPZ 的 point_ids、xyz 和 RGB 与重新加载的原生模型完全一致。见 [validation.json](validation.json)。

这些是拟合同一观测后的内部一致性检查，**不是独立正确性或可靠性证据**；没有实现阶段 4 的可靠性信号。相机精度另由独立 DTU reference 评价。阶段 1 关于 DTU 原生像素原点缺少独立特征标注的限制仍保留，本次不输入其 K，也不据此宣称像素级标定准确度。

## 5. 真实小样本结果

以下仅为 **一个 DTU 开发场景、三张图像** 的 smoke result，不是固定 Benchmark、场景总体成功率或跨方法排行。GT 在独立 `evaluate-poses` 命令中读取，不参与模型选择、建图或推理对齐。

| 指标 | 最终验收实测 |
|---|---|
| 注册图像 | 3/3，覆盖 100% |
| 稀疏点 / 观测 | 690 / 2071 |
| 相对旋转误差，3 个输入无序图像对 | 均值 0.621119°，中位数 0.681488° |
| 有向平移方向误差，3 个图像对 | 均值 1.190776°，中位数 1.257965° |
| 相机中心 ATE | 三个非共线中心做评价专用 Sim(3) 后 RMSE=1.383421 mm |
| 内部重投影残差 | 均值 0.172704 px，中位数 0.120094 px，800×600 像素单位；不是 GT 误差 |

只用三个中心拟合 Sim(3) 约束很弱，ATE 不能外推到更大场景或物体表面精度。单目原始模型单位是 arbitrary；毫米只来自独立评价的参考坐标与尺度对齐。没有计算稠密 Accuracy/Completeness/F-score，也没有将稀疏点云当作稠密重建。

逐对误差、对齐参数和每相机中心误差均保存在 [runs.json](runs.json)。成功只表示此次全部输入注册且输出通过验证，不等于任意输入都能重建成功。

## 6. 成本与全部尝试记录

最终运行 `outputs/stage2_colmap/final_validation` 的实际计时：

| 阶段 | 秒 |
|---|---|
| 输入校验与预处理 | 0.320881 |
| backend 导入 | 0.036213 |
| SIFT 提取 | 0.503107 |
| 全匹配与几何验证 | 0.201448 |
| 增量建图，含原生 BA | 0.198300 |
| 导出与内部验证 | 0.049646 |
| worker 总 wall time | 1.104280 |
| runner 总耗时 | 1.430424 |

总耗时包括输入处理、子进程启动、采样和结果整理，不是单纯 mapper 时间；不包含安装、测试或独立 GT 评价。末尾结果 JSON 落盘的微小时间不纳入已写入的 total。worker OS 峰值 RSS 为 529,268,736 bytes，另以 50 ms 采样记录 528,723,968 bytes；这是 CPU 进程内存，**不是显存**。GPU 未使用，显存指标为 null，未编造零显存测量。少量输入、CPU 缓存和系统负载会影响计时，不作为普适速度结论。

三次真实运行全部保留，选最终运行是因为对应最终修复后的实现，未按 GT 误差择优：

1. `attempt01`：初版接入验证，3/3、690 点。
2. `acceptance`：固定初版实现验收，配置未改变，3/3、690 点。
3. `final_validation`：修复中断状态写入后再验收，配置未改变，3/3、690 点。

三次的相机精度结果一致，各次成本和源码摘要分别记录。全部原生日志保存在 `attempt01_native_log.txt`、`acceptance_native_log.txt`、`final_validation_native_log.txt`。日志显示官方 mapper 初始找不到合适图像对，随后**由官方自身放宽初始化约束**成功；没有隐藏这段过程，也没有在项目中根据 GT 修改参数。

## 7. 检查、文件与运行命令

主要新增：`src/georecon/baselines/colmap.py`、`colmap_worker.py`、方法配置、隔离依赖锁、`tests/test_colmap_adapter.py`，以及本目录证据。CLI 增加 `run-colmap`。原生数据库、二进制模型、处理图像和 NPZ 保留 `outputs/`，不进入 Git；小型预测/计时/评价快照与日志供审查。

43 项检查的实际日志见 [tests.txt](tests.txt) 与 [test_run.json](test_run.json)。包括已有阶段 1 检查、新像素中心转换、损坏点云、部分注册、超时进程终止、模拟 OOM/异常、截断状态、拒绝覆盖和真实 PyCOLMAP 对空白合成图像返回失败。**模拟 OOM 只测试记录分支，未声称本机发生过真实 OOM。** 此次真实 DTU 尝试没有失败；官方内部初始化失败/回退不从日志删去。

在项目根目录使用 Python 3.12 的独立环境：

```bash
python3.12 -m venv .venv-colmap
source .venv-colmap/bin/activate
python -m pip install -r requirements-colmap-lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m unittest discover -s tests -v

# 如果本机已有阶段 1 数据，无须重新下载
georecon run-colmap --manifest data/dtu_smoke_pinned/manifest.json --config configs/methods/colmap_cpu_smoke.json --output outputs/stage2_colmap/my_run
georecon evaluate-poses --prediction outputs/stage2_colmap/my_run/prediction.json --reference data/dtu_smoke_pinned/reference.json --output outputs/stage2_colmap/my_run/evaluation.json
```

新机器可先按 README 的 `prepare-dtu` 只获取该三图样本，之后使用对应 manifest 路径。输出目录必须为空；重跑换新目录，失败保留原目录。运行成功/部分注册时 CLI 返回 0，无法重建/超时/进程失败时返回 1，参数或输入错误返回 2。本机实际命令入口为 `/home/zchming/.venvs/georecon-colmap/bin/georecon`，这是运行记录，不是算法路径。

首次安装与验收命令见 [acceptance_commands.json](acceptance_commands.json)，最终命令的 stdout/退出状态见 [final_commands.json](final_commands.json)。本次真实推理没有读取 GT：输入目录只暂存处理后的 PNG、ID 映射和方法参数；数据库也确认初始焦距无 prior。GT 文件只由独立评价调用读取。

## 8. 限制与阶段停止点

- 当前只实现单共享、无畸变 PINHOLE、自估内参和 CPU SfM；拒绝其他相机模式，不默默丢弃畸变。只在 WSL/Linux 验证进程组超时及内存统计。
- 完整 COLMAP 轨迹留在原生模型，尚未规范化为项目的 tracks 数组；depth/confidence 为 null。小型接口没有承诺通用数据集适配或大规模性能。
- 超时不会把未验证的中间模型当成功结果；保留原目录以供排查。原生硬崩溃/外部 SIGKILL 无法可靠区分 OOM，记 process_error；只有明确 MemoryError 才记 oom。
- CPU wheel 不支持当前所需 CUDA MVS；系统 COLMAP CLI 仍未安装。未来若要 MVS，应另行核实构建和硬件，不能由这次 SfM 成功推断。
- 未运行 VGGT、VGGT 官方 BA、额外 BA 方法或数据扩张；COLMAP 自身 mapper 的 BA 是其标准 SfM 的一部分，不计作新贡献。
- 没有固定 Benchmark、跨方法性能结论、可靠性、选择/融合、教学 SfM 或前端工作。

**本次完成 COLMAP 子步骤后停止。** 下一步建议仍在阶段 2 内，按明确指令接入 VGGT 原始推理，先核对其预处理和坐标，之后再接官方优化流程；现在不开始。
