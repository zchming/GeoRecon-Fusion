# 安装、运行与复现边界

所有命令从仓库根目录执行。已验证目标为Linux/WSL、Python 3.12；本页不承诺所有平台兼容。旧命令及历史文件未改动，已有输出请保留，新运行使用不同目录。

**先选择复现目标：** 无数据的新用户先运行CPU协议与评价示例；有已知K图像再运行教学SfM/BA；复核研究结论则需要对应历史缓存、源码和权重。本页提供已有入口，本次成果整理只静态核对命令、资源条件与链接，没有执行示例或新实验。基础安装和示例完整运行不等于真实重建性能已复现。

## 安装与执行矩阵

| 任务 | 安装 / 前提 | 无模型离线运行？ |
|---|---|---|
| 合成协议与评价示例 | `python -m pip install -e .`（NumPy、Pillow） | 安装后可以；无需历史数据 |
| 已知K教学双视图、PnP、增量、BA | `python -m pip install -e '.[teaching,optimization]'`；无畸变图像与明确提供的K | 算法不需模型；真实示例需本地输入 |
| 完整CPU单元测试 | `python -m pip install -e '.[teaching,optimization,colmap]'` | 可以；测试使用临时fixture、mock与本地HTTP服务，不下载模型或DTU |
| COLMAP CPU SfM | `python -m pip install -r requirements-colmap-lock.txt`，再`python -m pip install -e . --no-deps --no-build-isolation` | 需要本地manifest和图像，不需要模型权重 |
| VGGT raw | 独立CUDA环境、[运行依赖](../requirements-vggt-runtime-lock.txt)、固定官方源码及本地checkpoint | 不能只靠基础安装；推理需要GPU和权重 |
| VGGT官方跟踪/BA | 独立环境，PyCOLMAP 3.10.0；额外tracker/DINOv2/特征权重与源码 | 需要全部本地资源；勿与PyCOLMAP 4.2.1基线混装 |
| SIFT＋LightGlue预检及受控实验 | 独立Torch/LightGlue环境、固定源码和SIFT权重；真实对照另需图像/缓存与冻结配置 | 完整模型运行不属于CPU基础安装；历史实验条件见各报告 |
| Stage7D匹配/轨迹支持诊断 | NumPy、完整Stage7C缓存和冻结代码基线；不调用模型/几何求解 | 缓存齐全后可离线，不是纯Git克隆可运行的真实数据示例 |
| 原阶段报告/验收重放 | 完整历史docs、data、outputs、源码版本及锁 | 不是clone后即可运行的独立示例 |

`pyproject.toml`的extras是当前允许版本范围；根目录requirements文件记录历史实测版本，不是同一环境可以混装的总锁。部分历史venv使用`.pth`共享既有运行库。当前CPU CI安装声明范围内可用版本，不将它冒称历史逐包环境。

## 独立完整示例：协议与位姿评价

此例不下载数据、不调用模型。图像与预测均为数学fixture，不是从图像恢复相机。安装命令及结果解释见[README Quick Start](../README.md#quick-start完整cpu协议与评价示例)。

```bash
georecon synthetic-demo --config configs/experiments/stage1_synthetic.json --output outputs/quickstart
georecon inspect-scene outputs/quickstart/manifest.json
georecon evaluate-poses --prediction outputs/quickstart/prediction_exact_sim3.json --reference outputs/quickstart/reference.json --output outputs/quickstart/rechecked.json
```

输出含manifest、生成图像、reference、三种prediction/evaluation以及run.json。exact_sim3是完整预测，perturbed_partial为人为缺失/扰动，no_reconstruction为人为失败；缺失误差为null，不填零。示例的意义是完整跑通数据与评价，不是测模型精度。

可检查`evaluation_exact_sim3.json`的完整注册和接近浮点零的对齐误差，`evaluation_perturbed_partial.json`的缺失相机，以及`evaluation_no_reconstruction.json`的无重建状态；`rechecked.json`是显式再评价输出。请使用尚不存在的输出文件及新的示例目录；生成器会拒绝已有manifest，但不保证任意非空目录安全。先激活对应Python 3.12环境，`georecon`来自本项目安装后的CLI入口。

## 真实教学SfM：需要已有本地输入

以下路径是本项目已有DTU输入布局，Git不含图片或准备产物。输入缺失时不要把运行失败当算法结果，也不要自动运行历史prepare脚本：它们还依赖更早的本地数据与验收锁。自备输入应遵循[相机协议](protocol.md)、[双视图格式与中间量](stage6/walkthrough.md)和相应加载器的严格字段检查。

双视图已知K入口已接入CLI：

```bash
georecon run-teaching-sfm --input data/stage6_pairs/dtu_scan1_001_003/input.json --config configs/teaching/stage6_two_view.json --output outputs/my-two-view
```

PnP/增量与BA使用现有模块入口；不能用双视图CLI代替：

```bash
python -m georecon.teaching.incremental_run --input data/stage6_incremental/dtu_scan1_reduced3/input.json --config configs/teaching/stage6_incremental.json --output outputs/my-incremental
python -m georecon.teaching.ba_run --input data/stage6_incremental/dtu_scan1_reduced3/input.json --baseline outputs/my-incremental --config configs/teaching/stage6_ba.json --output outputs/my-ba
```

增量需要3—5幅图像、每幅已知K和固定种子；当前不自动换种子或重试失败视图。BA读取同一输入与增量产物哈希，至少3台已注册相机。双视图/增量不完整、BA跳过或候选拒绝均可返回非零状态；必须查看prediction、steps或ba.json，不能只看是否产生文件。BA接受但达到预算未收敛有单独标签。

上述示例采用任意重建尺度、固定几何规范。真实R/t和毫米尺度仅用于独立评价；不要把DTU参考真值传入推理，也不要与自估内参基线混作同条件排名。阶段范围字段保留原运行定义，不代表后续模块尚未实现。

## COLMAP与VGGT基线

已有manifest与图片可直接运行COLMAP，不必重新获取数据：

```bash
georecon run-colmap --manifest data/dtu_smoke_pinned/manifest.json --config configs/methods/colmap_cpu_smoke.json --output outputs/my-colmap
georecon evaluate-poses --prediction outputs/my-colmap/prediction.json --reference data/dtu_smoke_pinned/reference.json --output outputs/my-colmap/evaluation.json
```

新用户获取DTU数据需先阅读[数据准备与来源](stage1/report.md#dtu-数据获取与限制)。`georecon prepare-dtu`会访问网络；它不属于Quick Start或CI。本轮工程整改未执行下载、COLMAP真实重建或任何新模型运行。

VGGT需先准备干净、固定版本的官方checkout与本地权重；占位变量必须替换为自己的资源路径：

```bash
georecon run-vggt --manifest data/dtu_smoke_pinned/manifest.json --config configs/methods/vggt_raw_smoke.json --official-repo "$GEOR_VGGT_REPO" --checkpoint "$GEOR_VGGT_CHECKPOINT" --output outputs/my-vggt-preflight --preflight-only
# 真正推理需要另外运行run-vggt（去掉--preflight-only），并使用新的输出目录。
```

`--preflight-only`只检查预处理，不证明模型推理成功。[VGGT raw报告](stage2_vggt/report.md)列出固定commit、checkpoint SHA和已验证环境；完整官方BA安装与命令见[BA复现说明](stage2_vggt_ba/reproduce.md)。官方BA三图调度适配曾完成，冻结五图配置仍OOM，不能因GPU标称12GB便承诺可运行任意视图数。

历史批量运行器及LightGlue预检包含固定venv、缓存和输出位置，应按历史报告恢复相应环境；它们尚未改造成通用Benchmark入口。不要为重跑历史验收而覆盖旧结果、修改哈希锁或把新结果写回旧报告。

## 研究结果与只读诊断复核

| 需要核查的证据 | 已有入口及前提 |
|---|---|
| 受控A/B/C匹配及首轮失败 | [Stage7A报告](stage7a_results.md)：冻结图像/预处理、共享特征、固定LightGlue源码和本地权重；不单凭基础安装宣称可复现 |
| 种子选择与PnP组合 | [Stage7B报告](stage7b_initialization.md)：DTU等五来源缓存、相机适配与原几何配置；自估K的COLMAP仅作另协议参考 |
| AP3P留出确认 | [Stage7C报告及完整命令](stage7c_confirmation.md)：原冻结配置、Stage7A/B依赖和新对象缓存；其准备入口仍访问网络，不是离线示例 |
| 支持损失、不可评估与停止门 | [Stage7D报告](stage7d_support_diagnostic.md)：960份record/map、特征与匹配缓存及冻结源码；有全部缓存才能独立复核计数 |

Stage7D入口需要在原仓库布局下恢复清单对应缓存（旧索引含本地绝对路径；搬迁后不可静默改写并冒称同一冻结运行）。以下命令**仅供具备原布局/哈希的维护者复核，不是新用户Quick Start**；输出选用冻结父目录下新的子目录，禁止覆盖已有结果：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/diagnose_stage7d_support.py \
  --config configs/experiments/stage7d_support_diagnostic_v1.json \
  --data outputs/stage7c_data_v1 --study outputs/stage7c_confirmation_v1 \
  --output outputs/stage7d_support_diagnostic_v1/review_new
```

入口只读取既有关键点、匹配索引、观测身份和record，先检查哈希，诊断表封存后再关联旧评价，不调用GT几何或新求解器。检查`decision.json`、`session.json`、`unavailable.csv`及`features_seal.json`，不要把0个几何求解或程序正常退出写成性能通过。原`failure.json`及两次完整输出保持原位；大表/缓存不在Git，公开的是分组摘要与代表性失败解释。

## CPU CI与跳过解释

[工作流](../.github/workflows/cpu-tests.yml)使用Python 3.12，安装`.[teaching,optimization,colmap]`，先检查依赖，再运行全部单元测试。无模型、DTU或历史缓存输入。GitHub Actions本身将在提交/推送或手动触发后才能验证，本地通过不等于已在远端运行。

- verbose日志保留每项测试的执行/跳过原因；摘要分别统计通过、跳过、失败事件和错误事件。
- 最新发布提交`b7fdf33`的[实际CI](https://github.com/zchming/GeoRecon-Fusion-Geometry-Reliability-Aware-Hybrid-Multi-View-3D-Reconstruction/actions/runs/39127655433)报告252项：246通过、6项独立LightGlue环境依赖跳过，0失败/错误。其中5项为输入契约，1项为官方RootSIFT读取保护测试；跳过不算通过。
- 同一提交发布前的本地主环境为251通过、1项原生PyCOLMAP依赖跳过；本地具备LightGlue而CI具备PyCOLMAP，不能合并成“全部252项通过”。本次文档整理没有重跑上述测试，也没有新CI。
- 只装基础依赖时还会跳过OpenCV、SciPy或PyCOLMAP相关测试，必须按日志解释，不能称全部通过。
- 依赖安装失败、测试导入错误、断言失败或零测试均不能报成功。

测试示例仍使用原算法的数学fixture和退化控制；测试数不代表独立场景数或性能实验数。

## 演示资源来源

README两张PNG由现有运行预览原样复制，未裁剪、修饰、重算匹配或调用生成模型。DTU原图及大规模点云仍不进入Git。

| 展示文件 | 本地历史来源 | SHA256 |
|---|---|---|
| [sift-001-003.png](assets/sift-001-003.png) | outputs/stage6_two_view_v1/dtu_scan1_001_003/matches.png | 314eeaade723e5da09642e2afb0adb050f3c77187d3430c29ec3be5d898c23a8 |
| [sift-001-020-failure.png](assets/sift-001-020-failure.png) | outputs/stage6_two_view_v1/dtu_scan1_001_020/matches.png | 5fce7f1607d8d3b67db10c2b0208b86bf6d6a8277ef518d06dfd3b6c8cd0406e |

关键点与显示抽样规则来自[preview()](../src/georecon/teaching/run.py)：最多80条等间隔匹配，青色为最终接受点对应、绿色为其余E内点、红色为非E内点。宽间隔匹配不足时不估计E，图中红色不能被解释为GT判错。数值来源为[原双视图摘要](stage6/summary.json)与[教学验收](stage6_acceptance/report.md)。当前没有新增三维点云渲染图；已有点坐标在本地产物中，不能用匹配图冒充三维重建截图。

## 第三方归属与数据

| 组件 | 来源与归属 |
|---|---|
| COLMAP/PyCOLMAP | [官方项目](https://github.com/colmap/colmap)；本项目适配其SfM与优化，不原创其求解器 |
| VGGT及官方跟踪/BA | [官方源码](https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce)；模型、跟踪器、特征权重等来源与条款见[接入报告](stage2_vggt_ba/report.md#输入依赖与权重) |
| LightGlue | [官方项目](https://github.com/cvg/LightGlue)；本项目已完成SIFT权重预检、受控匹配与后续SfM对照，使用第三方预训练模型；[版本与权重](sift_lightglue_preflight/report.md)、[受控验证](stage7a_results.md)。VGGT BA使用的fork与此处官方配置不同 |
| 数值工具 | OpenCV特征/估计器、SciPy优化器、NumPy/Pillow；教学链与实验流程由本项目组织实现 |
| DTU | [官方数据页面](https://roboimagedata.compute.dtu.dk/?page_id=36)；引用Jensen等《Large Scale Multi-view Stereopsis Evaluation》，CVPR 2014；数据获取、引用要求及原说明见[数据报告](stage1/report.md) |

本页只梳理归属，不为第三方代码、数据或权重重新许可，也未选择项目许可证。已实现能力、负结果和未证实的新颖性边界见[贡献清单](contributions.md)。
