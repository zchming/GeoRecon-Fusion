# SIFT＋LightGlue 兼容性预检（2026-10-17）

结论：**在当前两组 DTU 图像上，输入适配、官方权重和 GPU 推理接口可用，允许以后提出受控的前端验证方案；尚未证明匹配准确率、位姿、注册率或鲁棒性提升。** 本次不实施后端接入、不开展正式对比、不改融合、不进入下一阶段。没有下载新数据。

## 历史瓶颈复核

只读审计已有8个 COLMAP 数据库、预测、评价及日志，数据库读前读后哈希一致。逐对计数与来源哈希见 [bottleneck_audit.json](bottleneck_audit.json)。

- scan1 spaced3：020–001/020–010 原始匹配19/22，验证后均0；001–010原始1003、验证894，配置6（PLANAR_OR_PANORAMIC）。scan6对应22/26→0，以及941→857、配置6。020在验证图上孤立；已有日志还显示剩余种子初始化失败。
- 这是“弱匹配边被几何验证拒绝＋剩余边初始化歧义”的组合，不能把全部失败归因于描述子或数量。配置6是估计器分类，不是场景真实退化的证明；图像编号间隔也不是重叠率测量。
- 教学版已知K宽间隔001–020仅4条互检匹配，达不到原20条门限；正常与模糊对分别有427/101条。这里只引用历史结果，没有重跑旧匹配器作性能排名。
- 模糊五图的自估内参 COLMAP 可完整注册，却仍有约14°平均旋转误差及约45°平移方向误差，低残差也未保证正确位姿。前端、内参和几何尚未因果分离。
- 已有 VGGT BA OOM 属于其他流程，不声称此次小匹配器解决其资源问题。保留旧失败与本轮失败。

## 依赖、权重与可复现边界

官方源码固定为 [cvg/LightGlue eb42fee](https://github.com/cvg/LightGlue/tree/eb42fee2d71449efb0aa5c10549752b5d75384d8)。源码位于本机缓存，未放入仓库。代码 LICENSE 为 Apache-2.0；权重使用官方发布来源，不另行推断其法律范围。

新建独立环境 `/home/zchming/.venvs/georecon-lightglue`，通过 `base-runtime.pth` 引用既有 visual-geometry 的 site-packages，复用 Torch/CUDA；原环境没有安装或升级包。新装 LightGlue editable、Kornia 0.8.3、kornia-rs 0.2.0。实际 Python3.12、Torch2.11.0+cu128、OpenCV5.0.0.93、NumPy2.5.2。完整环境与安装过程见 environment_freeze.txt、dependency_install.txt、pip_check.txt。**环境依赖本机基础环境，并非独立锁文件。**

官方 [SIFT 权重](https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth) 下载至 Torch hub 缓存：47,632,573 bytes；SHA256 `5b52b8d9982d43532dc042606b346bb9594c9f5a4bd6f64362c63866287b4ac0`。下载＋CPU载入4.125秒，不含安装。检查253个状态项并按官方逻辑重命名旧键；无意外键、无缺失可训练参数；唯一缺失为官方构造生成的 `confidence_thresholds` buffer。权重、图像、vendor和输出NPZ均在Git外。

OpenCV执行CPU SIFT；LightGlue执行GPU匹配。该路径无需PyCOLMAP。使用Torch原生SDPA，无需第三方flash_attn；未编译模型。官方 `flash=True, mp=False` 并不表示所有内部运算均FP32：其CUDA注意力路径会转换半精度；此次未剖析实际选择的底层注意力内核。

## 输入和坐标约定

核对 [SIFT源码](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/sift.py)、[匹配器](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/lightglue.py)、[预处理](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/utils.py)：

| 项目 | 本轮冻结输入 / 检查 |
|---|---|
| 数据 | 既有scan1 001–003、001–020，原800×600 PNG，逐文件SHA校验 |
| 检测 | 旧流程同款 OpenCV SIFT(nfeatures=4096)，8线程、seed42、OpenCL关闭；不裁剪、不缩放、不删点 |
| XY | float32 `[1,N,2]`，顺序x,y；保留OpenCV `.pt`，与历史NPZ逐点完全一致；不擅加±0.5 |
| 描述子 | `[1,N,128]`，原SIFT→官方RootSIFT：L1归一、截断1e-6、开方、L2归一；检查有限及单位范数 |
| 尺度/方向 | `[1,N]`，分别使用KeyPoint.size与角度转弧度；不是octave，不是度数 |
| 图像尺寸 | `[1,2]`，顺序W,H，显式传入；不使用相机内参或GT |
| 输出 | 原特征索引、分数；范围、唯一性、双向matches0/1一致性及坐标回取均检查 |
| 空特征 | 检查空白图产生空描述子，调用匹配器前显式跳过；不宣称上游空输入安全 |

匹配器归一化是 `(xy-size/2)/(max(size)/2)`，不修改导出的像素坐标。官方提取器还原缩放坐标使用 `(xy+0.5)/scale-0.5`；本轮禁用缩放，恒等还原。上游OpenCV分支仍有角点/中心约定TODO，因此仅验证本地OpenCV路径一致性，**不声称与PyCOLMAP坐标通用等价**。

官方SIFT默认与旧提取不相同：contrastThreshold 0.0066667 vs 0.04、nOctaveLayers 4 vs 3；默认nms_radius=0仍有去重；默认长边resize1024会改变本例尺寸。单独运行官方提取器，明确 `resize=None`，原图输出3464点；未用于两个匹配烟测。该数字不能用于提取器优劣比较。旧特征4096是目标而非严格上限，本轮实际均不超限；脚本超限时失败，不暗中截断。

**适配旧SIFT仍改变了描述子为RootSIFT。** 未来若比较匹配器，必须共享同一组关键点、RootSIFT及元数据，才能排除提取器/度量混杂；不能把旧L2 ratio匹配和新结果的数量差直接归因于LightGlue。

## 单次接口与资源验证

均为batch1、原800×600、官方SIFT匹配权重、9层上限、depth_confidence=.95、width_confidence=.99、filter_threshold=.1、flash=True、mp=False。输入已在运行脚本中固定，没有按结果调整。没有调用E/RANSAC、PnP、BA或真值评价。

| 输入对 | 关键点数 | 输出匹配（未验证） | CPU提取＋适配合计 | H2D | 单次GPU匹配 | 峰值allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|
| 001–003 | 3809 / 4096 | 1134 | .108s | .00159s | .429s | 282.4 / 380 MiB |
| 001–020 | 3809 / 1357 | 28 | .082s | .00074s | .030s | 209.1 / 382 MiB |

RTX4070 SUPER，总显存12282MiB；设置本进程allocator上限11GiB。模型缓存载入＋GPU迁移.222秒；整卡运行后快照1158/1160MiB。Torch分配/保留峰值是同步后实测；整卡快照**不是连续采样峰值**，不覆盖其他程序瞬态显存，也不保证未来更大输入资源。没有OOM。单次耗时含冷内核/自适应退出（7/5层），不构成速度Benchmark。完整配置、来源哈希、耗时和NPZ哈希见 [smoke.json](smoke.json)。

首次执行在模型加载后，因配置为SimpleNamespace而脚本尝试dict转换失败；原始日志见 attempt01_namespace_failure.txt。修复为vars后同配置重跑；未换数据、分辨率、权重或算法参数。最终成功日志为smoke_stdout.txt。

## 验证与复现

```bash
PYTHONPATH=src /home/zchming/.venvs/georecon-reliability/bin/python scripts/audit_matching_bottlenecks.py
PYTHONPATH=src /home/zchming/.venvs/georecon-lightglue/bin/python scripts/preflight_sift_lightglue.py
PYTHONPATH=src /home/zchming/.venvs/georecon-lightglue/bin/python -m unittest discover -s tests -p test_sift_lightglue_preflight.py
```

运行依赖本机既有数据、历史NPZ和缓存。脚本先验源码commit和权重SHA，读取历史产物仅做一致性核对；模型推理输入不包含历史匹配、内参或GT。重跑会覆盖本预检成功报告与本地烟测NPZ，失败日志应另存。没有自动安装依赖或获取新数据。5项输入契约测试覆盖尺寸顺序、弧度、描述子范数、非有限值和维度拒绝；核心环境跳过可选依赖测试。回归发现152项：核心环境146通过/6跳过，COLMAP环境107通过/45跳过；结合独立LightGlue环境5项通过，152项不同测试全部得到交叉覆盖。日志分别为regression_tests.txt、colmap_regression_tests.txt、contract_tests.txt。43个历史源文件哈希复核不变，src/configs无改动，详见verification.json。

## 累计贡献与局限

已实现可复现的COLMAP/VGGT链、固定小样本评价、无真值几何诊断、选择/过滤/融合/联合优化及教学SfM与BA；保留退化、OOM和无收益结果。502点oracle及源视图内留出诊断帮助否定当前安全修正信号的充分性。本轮增加历史匹配图只读审计、官方学习匹配器的输入/权重契约和低显存兼容性证据。

尚无稳定融合收益、准确率/鲁棒性提升或算法新颖性证据。两个scan的小样本、已见开发样本、单次资源烟测及未验证的28条宽间隔匹配都不支持泛化结论。当前仅达到**前端候选兼容性预检通过**；后续正式受控验证须另获阶段授权，不能据此宣称LightGlue已修复重建。
