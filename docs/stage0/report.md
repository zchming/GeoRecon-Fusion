# 阶段 0：环境检查与方案落地

检查日期：2026-10-16（Asia/Shanghai）。仅执行环境审计、已有结果复核、官方接口与来源核查、方案设计。没有安装 COLMAP、下载新数据或权重、训练、重新推理模型或实现阶段 1–7。方案中的参数均为提案，不是已运行结果。

## 1. 结论与本阶段验收

**本机可开展 CPU 几何开发和小规模 GPU 推理准备，完整 Baseline 尚不具备直接运行条件。** 阶段 0 的报告交付完成，不等于所有环境组件均已就绪。

| 验收项 | 状态与证据 |
|---|---|
| 工作目录、系统、依赖、资源和编译能力 | 已检查；`environment.json` 保存实际命令、返回码和输出 |
| 既有实验和结果盘点 | 已审阅，未重跑；`legacy_evidence.json` 保存来源路径、哈希和原始内容快照 |
| 官方代码、版本、权重、许可和接口 | 已核查；`official_checks.json` 与第 4 节；运行兼容性未验证 |
| 数据候选与最小实验方案 | 已提出，未下载，等待用户选择 |
| 风险及后续阶段验收标准 | 本报告第 8–9 节 |

本阶段新增或修改：`README.md`、`.gitignore`、`scripts/check_stage0_environment.py`、`docs/project_brief.md`、`docs/stage0/` 下的报告和证据文件。旧 README 备份在 `legacy_environment_setup.md`，其中“暂未安装 VGGT”等历史说法不再代表现状。

## 2. 已有项目盘点

实际目录 `/home/zchming/projects/VisionGeometryLab`，展示名使用 GeoRecon-Fusion；不移动或删除旧文件。Git 已关联用户指定仓库，本阶段仅提交明确列出的审计交付物。

| 本机既有内容 | 用途和边界 |
|---|---|
| `verify_environment.py`、环境锁文件、Notebook | 历史准备记录，不等于已验证的跨机器安装包 |
| `experiment03_depth_reprojection/` | 约 4.9 GiB，含官方源码、权重、示例图、两帧预测与学习分析 |
| `experiment03_camera_geometry/`、`experiment05_camera_rays/` | 相机、反投影、射线的学习实验，不是正式 Baseline 比较 |
| `experiment03_vggt_tensor_anatomy.py` 等 | 张量分析，未迁入统一接口 |

两帧输入为官方 VGGT 仓库的 LLFF fern `001.png` 与 `002.png`，旧记录 shape 为 `[1,2,3,392,518]`，BF16 autocast、FP32 参数。历史推理 1.265346504 s，峰值 allocated 5,694,090,752 bytes（5.303 GiB）、reserved 6,450,839,552 bytes（6.008 GiB），旧脚本总耗时 8.609223836 s。**这是已有文件记录，本阶段未重跑，不能作为新测量或普遍预算。** 另一个学习脚本记录 0.8426 s，计时范围不同，不能混成重复实验统计。

旧误差参照为模型自身 Track Head，且有置信度/遮挡启发式筛选，没有独立真值。它只反映模型输出间一致性，不能证明真实精度、概率校准或融合收益。旧验证 JSON 的 PASS 是历史记录，不代表本次重新验收全部学习实验。其内容、来源与哈希已快照，旧脚本、图像和大型产物仍保留本机，不在本阶段整体上传。

## 3. 实测环境

[environment.json](environment.json) 包含 UTC 时间戳与逐项证据。资源空闲量为一次采样。

| 项目 | 已确认事实 |
|---|---|
| 系统 | WSL2，Ubuntu 24.04.4 LTS，x86_64，内核 6.18.54.1-microsoft-standard-WSL2 |
| CPU / 内存 | i7-13700KF，WSL 可见 24 逻辑 CPU，约 15 GiB RAM、4 GiB swap；不是宿主全部资源 |
| 磁盘 | 当前文件系统可用约 939 GiB；WSL 虚拟盘视图不保证 Windows 物理盘有等量空闲 |
| Python | `/home/zchming/.venvs/visual-geometry/bin/python`，3.12.3 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128，CUDA runtime 12.8 |
| GPU | RTX 4070 SUPER，12,282 MiB 总显存，compute capability 8.9，BF16 可用 |
| 当前空闲显存 | torch 查询 11,587,813,376 bytes，约 10.79 GiB |
| 驱动 | nvidia-smi 615.78.05，Windows KMD 617.42，CUDA UMD 显示 13.4；不能与 torch runtime 混淆 |
| 实际导入 | NumPy 2.5.2、SciPy 1.18.1、OpenCV 5.0.0、Pillow 12.3.0、Matplotlib 3.11.2 |
| GPU 检查 | 128×128 GPU/CPU 矩阵乘法对照通过，rtol=atol=1e-3；没有运行模型 |
| 编译能力 | GCC 13.3.0、CMake 3.28.3、Ninja 1.11.1；临时 C++ 工程配置、构建、执行返回码均为 0 |
| pip check | 返回 0，No broken requirements found |
| COLMAP / CUDA 编译器 | PATH 未发现 colmap 或 nvcc；未确认任何可运行 COLMAP 构建，MVS 未就绪 |
| 构建依赖发现 | pkg-config 未找到 eigen3 / ceres-solver / sqlite3；不等于穷尽所有安装位置 |

基础 venv 不含 pycolmap、pyceres、LightGlue、trimesh、huggingface_hub、einops、safetensors。后三者在旧实验的 `deps/` 中，子进程显式加入该目录后可导入，版本分别为 2.2.0、0.8.2、0.8.0；旧实验成功不能当成标准环境开箱可用。

网络 HEAD 检查：GitHub、COLMAP 文档、固定 VGGT 权重 URL 和 DTU 页面返回 200；ETH3D 本机超时，另外一次 curl 出现 TLS 错误。HEAD 成功不保证完整大文件下载成功。既有权重本次重新流式计算 SHA-256，与旧记录一致；未重新下载。

## 4. 官方实现核查

### VGGT

官方仓库：[facebookresearch/vggt](https://github.com/facebookresearch/vggt)。本地与远端 HEAD 都是 `a288dd0f14786c93483e45524328726ab7b1b4ce`，本地 `git status --short` 为空。后续固定此 commit，不静默更新 main。

[固定版本 demo_colmap.py](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/demo_colmap.py) 确有 `--use_ba`，包含 VGGSfM 跟踪、COLMAP 导出与 `pycolmap.bundle_adjustment`。这些是官方 Baseline，不是项目原创算法。

[基础依赖](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/requirements.txt) 固定 torch 2.3.1、torchvision 0.18.1、numpy 1.26.1；当前机器更高版本的学习推理成功不能证明完整 demo 兼容。[Demo 依赖](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/requirements_demo.txt) 另列 pycolmap 3.10.0、pyceres 2.3、jytime/LightGlue。阶段 2 再验证隔离环境，不直接降级现有 venv。

[代码许可](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/LICENSE.txt) 是自定义 VGGT License，不是 MIT。代码与权重要分开登记。[原始 VGGT-1B 模型卡](https://huggingface.co/facebook/VGGT-1B) 标示 CC-BY-NC-4.0；商业权重是另一 checkpoint，不自动替换当前模型。

既有权重固定来源：`https://huggingface.co/facebook/VGGT-1B/resolve/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt`。文件 5,026,874,952 bytes，SHA-256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`。不进入 Git。

固定源码的直接检查结果（尚未运行这些接口）：

| 项目 | 行为及接入要求 |
|---|---|
| 输入顺序 | 读取 scene_dir/images 下通配文件，未显式排序；接入时用 manifest 固定顺序并记录适配 |
| 图像预处理 | 加载为 1024 方形，再以 518×518 推理；与旧实验 crop 后 392×518 不同 |
| 相机 | OpenCV camera-from-world；导出重缩放内参与像素到原图体系，需单独核验 |
| 无 BA 导出 | PINHOLE、非 shared camera，confidence 阈值默认 5.0，最多随机导出 100,000 点，seed=42 |
| BA 默认 | SIMPLE_PINHOLE、max_reproj_error=8 px、vis_thresh=0.2、query_frame_num=8、max_query_pts=4096、fine_tracking=True |
| BA 附加资源 | VGGSfM tracker、DINOv2 排序、ALIKED/SuperPoint；新权重的固定 revision、哈希、许可与访问性留待阶段 2 核查，不假定已下载 |
| 已知限制 | 源码有 masks、radial distortion、iterative BA 等 TODO，不写作完整通用 SfM |
| 输出核验 | BA 后 reconstruction 已优化，但快捷 points.ply 使用之前的 points_3d 变量；应从优化后的 reconstruction 取评测点，不能凭 PLY 名称认定为 BA 输出 |

原始 forward 输出和官方过滤/导出分别登记；不能将过滤后的点集说成全部原始预测。当前未运行官方 demo 或 BA。

### COLMAP

官方仓库：[colmap/colmap](https://github.com/colmap/colmap)。本次 [稳定发行页面](https://github.com/colmap/colmap/releases/tag/4.2.1) 为 4.2.1，tag 指向 `bd1fcf654d2dd8fefa1466999c190a246f83f4b9`，作为后续候选版本，**不是已安装版本**。滚动在线文档与稳定版可能不同，运行参数须用最终安装版本的帮助验证。

[固定 CLI 文档](https://github.com/colmap/colmap/blob/4.2.1/doc/cli.rst) 确认稀疏路线 `feature_extractor → exhaustive_matcher → mapper` 和稠密路线 `image_undistorter → patch_match_stereo → stereo_fusion`。[固定安装文档](https://github.com/colmap/colmap/blob/4.2.1/doc/install.rst) 区分 GPU 构建，发行版包不默认带 CUDA/HIP；本机 NVIDIA MVS 需验证 CUDA 构建，不能由 PyTorch CUDA 可用推断。

[4.2.1 COPYING.txt](https://github.com/colmap/colmap/blob/4.2.1/COPYING.txt) 对库本身采用 new BSD，依赖另有许可。本阶段只检查文档，不重新分发其源码/二进制。C++ 工具链通过不等于 COLMAP/Ceres/CUDA 构建通过。当前无权重（经典方法不需要模型 checkpoint），运行配置尚未生成。

建议 COLMAP CLI 独立安装，VGGT BA 按其 pycolmap 3.10.0 接口隔离，不把新 COLMAP Python API 直接塞入旧 demo。没有修改系统驱动或安装 Toolkit。

## 5. 数据集候选：均未下载

| 候选 | 适用任务 | 获取、规模和局限 |
|---|---|---|
| DTU MVS 2014 | 标定/相机协议核对，稠密 Accuracy、Completeness；位姿可对照标定 | [官方入口](https://roboimagedata.compute.dtu.dk/?page_id=36) 提供标定、扫描真值、观测 mask 与评价代码。SampleSet 约 6.3 GB，完整图像包大得多。建议先用 SampleSet 一个场景的 3–5 图；具体成员、使用条款与实际包大小下载前再查 |
| ETH3D 高分辨率 multi-view training | 室内/室外稠密几何，官方容差和可见性协议 | [概览](https://www.eth3d.net/overview)、[评价说明](https://www.eth3d.net/documentation)；training 提供参照，hidden test 不能当本地有真值。本机访问失败，单场景下载大小、数据许可仍待确认，不作为当前必达依赖 |
| CO3Dv2 | 对象序列位姿、相对位姿误差与注册覆盖 | [官方仓库](https://github.com/facebookresearch/co3d) 提供约 8.9 GB 的单序列子集集合，并非单个序列 8.9 GB；不下载 5.5 TB 全集。相机/点云是重建型参照，不当作独立激光级真值；需披露潜在 SfM 方法偏向 |

DTU 更适合标定/扫描几何，CO3D 更适合对象位姿主线。公开可访问不等于允许任意再分发；确认候选后再核对具体许可、引用、大小与范围，不上传原始图像。本阶段未确认模型预训练数据与候选测试场景无重叠，不能承诺严格未见场景泛化。

**建议用户确认 DTU 单场景开发冒烟，或 CO3D 位姿优先路线。** 已有 fern 双图可作为无需新下载的接口冒烟，但没有本项目已核实的独立真值，不能充当正式 Benchmark。这里没有锁定正式测试集。

## 6. 最小可执行方案：后续阶段提案

### 资源和预算

先用 2 张图校验解码、顺序、预处理与导出，再尝试 3–5 张。不承诺固定显存能容纳完整配置。VGGT 初步沿官方 518×518、BF16 autocast/FP32 参数路线；BA 的 1024 跟踪可能明显增加资源。旧 392×518 双图峰值不能直接外推。

先登记官方默认 BA 为待测配置；若 OOM，保存失败，再在 dev 尝试官方说明的 `max_query_pts=2048, query_frame_num=5`，作为独立预算版本，不覆盖默认结果。改变视图数、分辨率、点数都要记录新配置并统一重跑可比输入。无需训练或付费资源。

建议初始本机保护预算：单场景每方法 20 分钟、预留约 1 GiB GPU 空间、CPU 线程先用 8；它们是工程起点，不是实测性能。阶段 2 测量后再冻结正式预算。MVS 在 CUDA 构建和最小运行通过前标为“未运行”，不另加算法替代。

### 阶段 1 的协议设计

输入 manifest 至少有 scene ID、来源/许可、相对路径/SHA-256/图像顺序、原始尺寸、预处理矩阵、seed、split 和预算。配置与代码分离，不写死本机路径。

结果协议至少有 scene/method/run ID、代码 commit/dirty 状态、模型 revision/hash、逐图注册状态/原因、相机模型与畸变、K、world-to-camera `[R|t]`、原图/处理后尺寸、坐标轴、Z-depth/range、尺度和单位；深度、点云、轨迹、置信度可选。缺失能力明确 unavailable/null，不能伪造恒等相机或零误差。另存配置、日志、阶段/总耗时、allocated/reserved 与设备侧显存采样。

合成数学检查覆盖 `Xc=R Xw+t`、`C=-Rᵀt`、正深度、投影/反投影、内参缩放和裁剪、三角化及 SE(3)/Sim(3) 对齐；合成检查不属于 Benchmark。

阶段 2 在用户确认的小场景按同一输入清单接入 COLMAP SfM、VGGT、官方 BA，先查坐标/尺度再比较。各方法自估内参或已知标定 K 属于影响结论的协议问题，数据确定后须用户确认，不能只给某一方法额外 GT K。推理不输入 GT 位姿/点云；评价单独读取 GT。

### 指标和公平比较

小场景先报告逐图注册状态/比例、相对旋转误差（deg）、平移方向误差（deg）、失败原因、耗时和显存。平移方向不是相机中心距离；近零基线标记不可评估。评测图像对使用明确的输入无序对或最终数据官方协议；缺失对不能从覆盖/失败统计中消失，已注册子集误差要连同覆盖率报告。

至少 3 个非退化相机中心再使用有意义的 Sim(3) 中心 ATE；不在两帧上报告可被拟合对齐消去的中心误差。单目尺度由独立评价模块对齐 GT，推理对齐不得访问 GT。后续位姿 AUC 可按 max(rotation, translation-direction) 定义，但阈值、积分和缺失对规则都须阶段 3 在看到 test 前固定。

稠密评价只有满足数据协议时才启用 Accuracy、Completeness、P/R/F-score；记录方向、单位、阈值、采样、mask 和对齐，优先官方实现。DTU 单位随其标定协议，CO3D 尺度不直接当毫米或米。数据确认前不随意填写距离阈值，不跨数据集直接排名。

### 阶段 3 的 Benchmark 草案

建议先 6 个独立场景，3 dev / 3 test，仅为探索性小样本，不宣称领域代表性。首个调试场景仅归 dev；同一物理场景、近重复和全部扰动必须归同一 split。最终场景/图像 ID、重叠和模型训练数据风险在阶段 3 锁定。

先正常输入与数量减少（例如 N=5/3），明确减少数量不等于低重叠。低重叠需独立匹配/已知邻接支持的可见重叠量，不能仅改变 stride 就命名。合成条件草案：相同 5 图、seed=42，Gaussian kernel=7/sigma=1.5；曝光增益=0.6 或 1.4，裁剪至有效 RGB 范围。参数只在阶段 3 冻结。低纹理来自有证据的真实场景，另行报告。任何离线用 GT 构造的输入协议都必须披露且对方法一致，不能流入可靠性推理。

保存 success、partial、no_reconstruction、timeout、OOM、process_error 等状态。建议 operational success 为全部输入注册且输出有限合法、未超时；partial 为至少 2 张但未全注册；几何正确性另报，不能把能输出等同正确。正式运行前冻结判定。逐场景原始表先保存，再算均值、中位数、覆盖与失败；失败计入分母，不只汇报成功子集。冷加载、预处理、forward、BA 和总 wall time 分开；混合方法计入全部分支、对齐和可靠性/融合耗时。

## 7. 可靠性与融合边界

阶段 4 建议先考虑独立匹配观测的残差与匹配支持/覆盖两类信号，保留 unknown，再由 dev 失败分析决定是否加入深度一致性。独立匹配也会错，需数量、内点率和空间分布；由同一观测拟合并评估的低残差不能当独立正确性证据，可留出观测检验。

不把模型自投影恒等一致性当可靠性，不把 confidence 当概率；只在 dev 校准量纲和规则。阶段 5 先场景选择/回退，再约束过滤，有证据才加融合复杂度。不平均未对齐的相机或点云；证据不足或无法推理对齐时回退/未知，不借测试 GT 解锁融合。

## 8. 风险、失败原因和未完成项

1. COLMAP 未安装，CUDA MVS 未验证；C++ smoke test 不是 COLMAP 构建证明。当前 `sudo -n` 需要密码，未安装系统依赖或改驱动。
2. 本机 NumPy 2 与官方 numpy<2、旧 pycolmap 与新 COLMAP 有兼容风险；阶段 2 验证独立环境，不破坏旧环境。
3. 官方 BA 新增 VGGSfM/DINOv2/特征模型，权重来源、许可和资源还需固定；1024 跟踪峰值未测。
4. 数据未选、未下载、未划分，ETH3D 直连不稳定，DTU 官方入口的最小完整包也不算很小。
5. 既有学习误差不是 GT 准确率；没有 COLMAP 对比、正式 Benchmark 或融合结果，不能写性能提升。
6. crop 与 square 混用会产生假残差；官方 BA 快捷 PLY 不能直接视为优化后结构，阶段 2 必须核验。
7. WSL 主存约 15 GiB，既有权重约 4.68 GiB，加载复制可能吃紧；扩张前核对宿主磁盘余量。
8. 数据规模、重建型参照与预训练重叠会限制结论；如融合无效如实保留，不增加额外模型或前端。

## 9. 后续验收与停止点

**等待用户明确授权阶段 1；现在不实施。** 建议阶段 1 验收：最小项目包/配置、相机及 manifest/result 协议、关键数学合成检查、小场景读取和独立评价入口、缺失/失败集成检查。若没有已批准的真实场景，只能完成合成链路，真实场景验收必须明确保留未完成。

阶段 2 另行授权：固定代码/依赖/权重 manifest，同一小场景各 Baseline 原始输出、预处理/坐标核验和实际成本；允许记录失败，不默认全部成功。阶段 3 才正式 Benchmark；阶段 4–5 才可靠性/融合；教学版 SfM 仅在明确要求后提前。

下次需用户选择：DTU 标定/稠密优先，还是 CO3D 位姿优先。候选、范围和条款确认前不下载。内参公平比较方案在具体数据确定后确认。

## 10. 可直接执行的复查命令

在本机项目根目录执行；解释器路径只作为本机运行记录，算法不依赖该路径：

```bash
/home/zchming/.venvs/visual-geometry/bin/python scripts/check_stage0_environment.py --output /tmp/georecon-stage0.json
# 可选网络 HEAD，不传输权重文件
/home/zchming/.venvs/visual-geometry/bin/python scripts/check_stage0_environment.py --network --output /tmp/georecon-stage0-network.json
git remote -v
```

实际执行了 `--network --output docs/stage0/environment.json` 审计。脚本做小矩阵 CUDA 运算、临时 C++ 编译、既有权重 SHA-256，不做模型 forward；临时编译目录自动清理，缺依赖如实记录。版本查询与官方固定文件的 HTTP 状态/哈希见 `official_checks.json`，旧记录及固定 VGGT 文件哈希见 `legacy_evidence.json`。阶段 0 报告完成后停止。
