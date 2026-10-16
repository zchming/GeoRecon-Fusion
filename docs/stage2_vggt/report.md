# 阶段 2 子步骤：VGGT 原始推理接入

日期：2026-10-16（Asia/Shanghai）。用户明确要求先核对预处理、坐标和统一输出，再使用同一 DTU 三图运行、验证并推送；暂不接入官方 BA。**本子步骤完成，阶段 2 的官方优化流程仍未执行，未进入阶段 3。**

## 1. 实际交付与验收

| 验收项 | 实际结果 |
|---|---|
| 官方实现固定 | 复用干净的官方 checkout `a288dd0f14786c93483e45524328726ab7b1b4ce`，保存关键源文件哈希，未改动其代码 |
| 权重固定 | 复用既有原始 VGGT-1B，完整 SHA-256 与指定 revision 对应；严格加载，无新权重下载 |
| 公共输入一致 | DTU scan1/light3 的 001、002、003；原始哈希、顺序和公共 800×600 PNG 字节与既有 COLMAP 运行全部一致 |
| 官方预处理 | 实际调用官方 crop loader；与独立 PIL/NumPy 实现逐像素一致，差异为 0，张量 `[3,3,392,518]` |
| 坐标/协议 | 保留 native 相机，导出相机回原图 1600×1200；复合缩放裁剪映射检查、CPU 官方 pose 解码复查均通过 |
| 原始输出 | 相机、Z-depth、depth confidence、point-head world map、point confidence，以及 camera head 的迭代输出均保留；不滤点、不融合、不采样输出 |
| 真实 GPU 验证 | 三张图均有有限合法相机，完整原始数组形状/类型检查通过，无 OOM、无降级、无 offload 或模型训练 |
| 检查 | 52 项 CPU 数学/协议/接口检查通过；GPU forward 和独立位姿评价另有真实记录；运行环境 pip check 通过 |

实现提交 `950d102`，真实 GPU 验收对应此提交及实际 `source_tree_sha256`。随后仅整理文档，不更改已验证的模型实现。历史未跟踪文件仍保留，所以 Git dirty 状态如实为 true，不伪称 clean。预处理预检先于真实 forward，预检时没有加载模型或伪造相机。

## 2. 官方实现、权重和环境

对照 [固定版本官方 Quick Start](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/README.md)，使用 `VGGT().eval()`、严格加载本地 checkpoint、BF16 autocast 与 `model(images)`；保留默认 heads，不传 query_points，因此 track head 不执行。相机由官方 `pose_encoding_to_extri_intri` 解码。这不是 `demo_colmap.py` 的方形预处理/导出流程，也没有调用 BA 或其他优化器。

原始权重来源为 [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)，revision `860abec7937da0a4c03c41d3c269c366e82abdf9`，文件 5,026,874,952 bytes，SHA-256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`。本次在加载前重新计算完整哈希，再用 `weights_only=True, mmap=True` 本地读取；未调用 from_pretrained 或下载方法。HF offline 模式在 worker 启动时启用。

[源码许可](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/LICENSE.txt) 与原始权重许可分别记录：源码是自定义 VGGT License，原始模型卡标示 CC-BY-NC-4.0。本次没有切换商业 checkpoint，也没有重新分发第三方源码或权重。

运行解释器为 `/home/zchming/.venvs/georecon-vggt/bin/python`。它是**独立命令环境、只读共享既有科学计算库的 overlay**：共享 `visual-geometry` 的 Torch/NumPy/Pillow，新环境本地安装 hub/einops/safetensors 及解析依赖，没有改写原 venv。它不是完全独立复制的 Torch 环境；完整依赖与这一限制见 [environment.json](environment.json)。

实际版本：Python 3.12.3，torch 2.11.0+cu128，torchvision 0.26.0+cu128，CUDA runtime 12.8，NumPy 2.5.2，Pillow 12.3.0，huggingface_hub 2.2.0，einops 0.8.2，safetensors 0.8.0。GPU 为 RTX 4070 SUPER。官方历史 requirements 固定更旧的 torch 2.3.1/NumPy 1.26.1；这里是本机已实际验证的新运行组合，不宣称复现论文的原环境或所有设备兼容性。

`requirements-vggt-runtime-lock.txt` 固定直接运行依赖；环境证据另存所有可见依赖版本。此次没有重新安装 CUDA 大包、改驱动、创建云资源或训练。

## 3. 预处理的逐项核对

为保持与 COLMAP 的公共输入一致，先用同一 manifest 做原图 1600×1200 → 800×600 的 Pillow bilinear 处理，保存的三个 PNG 与 COLMAP 的 staged 输入 SHA-256 完全相同。然后实际调用官方 `load_and_preprocess_images(..., mode="crop")`：

1. RGB，float32，范围 `[0,1]`，不额外做 ImageNet normalization。
2. 宽固定 518，按比例计算高度并舍入到 14 的倍数，故 600×800 → 392×518。
3. 官方使用 Pillow bicubic；本场景不发生高度裁剪。对于本 adapter 支持的等尺寸输入，另有公式计算裁剪位移；混合尺寸的官方 batch padding 尚未适配，明确拒绝。
4. 真正推理的 tensor 来自官方 loader；独立 PIL/NumPy 分支只做逐像素验证，不替代它。预检最大差异为 0。

实际记录完整复合像素变换 `A_total = A_official_crop @ A_common`，采用像素中心 resize 公式 `u' = sx*(u+0.5)-0.5-left`；高度按实际舍入后的尺寸计算 sy，不偷用宽度缩放因子。

VGGT 官方 [geometry.py](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/utils/geometry.py) 使用 `np.arange(W), np.arange(H)` 作为像素网格；因此不应用 COLMAP 特有的 -0.5 平移。虽然 pose 解码的主点是 W/2、H/2，也不据此擅自更改官方坐标。导出只做 `K_original=inv(A_total) @ K_native`，R/t 原样保留，都是 world-to-camera。

**分辨率差异仍须披露：COLMAP 真正推理 800×600；VGGT 真正推理 518×392。** 相同图像/公共输入并不等于完全相同的模型分辨率；本阶段没有按误差调节这两个配置。

预检和正式运行的 tensor 哈希、全部 A 矩阵、原图/公共/模型尺寸见 [runs.json](runs.json)。校验原始 ID、原始字节及公共 PNG 与 COLMAP 的一致性见 [validation.json](validation.json)。

## 4. 统一输出与坐标验证

原始数组保存到本地 `raw_predictions.npz`：

| key | 形状 / 解释 |
|---|---|
| pose_enc | `[3,9]`，最后一次相机预测 |
| pose_enc_iterations | `[4,3,9]`，原模型输出的各次 camera head 预测，不是项目新增 BA |
| extrinsics / intrinsics | `[3,3,4]` / `[3,3,3]`，模型输入网格上的原生相机 |
| depth / depth_conf | `[3,392,518,1]` / `[3,392,518]`，相机 Z 深度及原始分数 |
| world_points / world_points_conf | `[3,392,518,3]` / `[3,392,518]`，point head 原始世界坐标图及分数 |

所有保存数组为 float32，实际检查有限性、维度和正深度；任何坏数组会保留文件、将运行标记失败，不偷偷替换非有限值或滤掉低置信度区域。输出保留全部 **609,168 个网格位置**，不是经验证的 609,168 个正确三维点，也不能与 COLMAP 的 690 个稀疏点按数量排名。

统一 `prediction.json` 的相机处于原图像素系；深度/点图/置信度描述明确仍在模型网格，并引用 native 相机及坐标 metadata，**没有假装将深度重采样回原图**。深度与点图均为任意重建尺度。tracks 为 null，因为没有 query points；confidence 不做概率校准。

协议的 `registered=True` 对 VGGT 表示“存在通过数值校验的预测相机”，不是基于特征轨迹的 SfM 注册。本次采用相机输出覆盖 3/3 的说法，不能宣称 VGGT 通过了 COLMAP 式注册。此差异在机器可读 [same_inputs_summary.json](same_inputs_summary.json) 也明确记录。

坐标检查结果：

- 从原图导出相机投影再映射回模型网格，144 个固定深度采样点的最大差异为约 **2.756e-5 px**；FP32 旋转非精确正交会留下小数值残差，没有用 SVD 改写模型旋转。
- 与官方深度反投影工具比较，最大世界坐标差异约 **2.655e-8**（任意尺度单位）。
- 文件重新打开后，CPU 官方 pose decoder 与保存的 GPU 解码一致；原生 R/t 保持不变，复合 A 作用于原图 K 可还原 native K；最后一次 camera 迭代等于 pose_enc。

上述均为**内部代数一致性**，不能作为真实精度或可靠性证据。仅在检查时抽取点，输出没有采样或过滤；没有实现阶段 4 的信号或融合。

原始 depth_conf 范围 `[1,92.9455]`，point confidence 范围 `[1,60.7755]`；显然不是 `[0,1]` 的已校准正确概率，全部原值保留。

## 5. 同一 DTU 三图的独立位姿评价

使用与 COLMAP 相同的 reference 和三个无序输入图像对；GT 仅由独立评价命令读取，未参与预处理、模型推理或输出对齐。相对误差直接计算；ATE 的 Sim(3) 是评价专用变换，不回写推理点图。

| 位姿指标 | 既有 COLMAP | 本次 VGGT raw |
|---|---|---|
| 可用相机 | 3/3 经典注册 | 3/3 预测相机可用，语义不同 |
| 相对旋转误差均值 | 0.621119° | 0.392139° |
| 有向平移方向误差均值 | 1.190776° | 0.721815° |
| 三中心评价专用 Sim(3) 后 ATE RMSE | 1.383421 mm | 0.423804 mm |

这只是**单个开发场景的结果并列展示**，没有固定 Benchmark、困难条件统计或泛化结论，不据此宣称方法整体优劣。仅三个中心的对齐约束较弱；ATE 不代表表面几何精度。各方法必要的预处理、硬件和输出能力不同，不能把这些结果包装成融合收益。VGGT 预训练场景重叠尚未审计，不能声称严格未见数据泛化。

逐对值、每相机中心误差、对齐参数完整保存在 `runs.json`，没有只保留有利图像对。没有稠密 GT 指标、跟踪、官方 BA、消融或参数搜索。

## 6. 实测成本与运行记录

仅执行一次真实三图 GPU forward；此前一次运行仅做预处理预检，没有加载模型。两次记录、源码摘要和完整日志均保留。

| 阶段 | 实测秒 |
|---|---|
| 父进程输入校验、公共预处理与官方源码检查 | 0.324492 |
| 运行库/官方工具导入 | 2.480862 |
| 官方预处理及独立像素核对 | 0.115160 |
| 完整本地 checkpoint SHA-256 | 3.160998 |
| 构造模型、严格加载和 GPU 转移 | 6.442739 |
| 同步计时 raw forward | 0.866305 |
| 解码、CPU 导出和内部检查 | 0.062577 |
| worker wall time | 14.303525 |
| runner 总耗时 | 14.644613 |

total 包括子进程等待、模型加载与本项目的输入/权重校验，不包含安装、测试或独立 GT 评价；末尾结果 JSON 落盘的微小时间未计入已写入的 total。新进程加载模型，未控制操作系统文件缓存，不能宣称严格冷缓存实验。不能用 VGGT 的纯 forward 时间与 COLMAP 整条链路耗时直接比较。既有 COLMAP runner 单次约 1.430424 s、CPU；本次 VGGT runner 约 14.644613 s、GPU，两者的测量范围详见各自报告。

GPU 保持 FP32 参数，采用 BF16 autocast；未 CPU offload、未减少视图数、未降低配置、没有 OOM。PyTorch allocator 峰值 allocated 为 **5,887,922,176 bytes（5.484 GiB）**，reserved 为 **7,067,402,240 bytes（6.582 GiB）**；不是驱动级整卡峰值，不包含其他进程。worker OS 峰值 CPU RSS 为 10,906,980,352 bytes（10.158 GiB），加载使用 mmap，这与推理 CPU offload 不同。CPU 内存也可能成为后续扩张的限制，不能由这次成功承诺更大输入一定能跑。

原生日志只有官方旧 autocast API 的 FutureWarning，已保留，未为消除警告修改固定官方源码。没有真实失败需要隐藏；测试中模拟的 OOM/异常只验证状态分支，不是虚构的真实运行。

## 7. 主要文件与可执行命令

新增 `src/georecon/baselines/vggt.py`、`vggt_worker.py`、`configs/methods/vggt_raw_smoke.json`、`requirements-vggt-runtime-lock.txt`、`tests/test_vggt_adapter.py`；CLI 新增 `run-vggt` 和 `--preflight-only`。复用通用进程超时/内存采样工具，不加载 PyCOLMAP 或其 BA。

本机直接运行，需选尚未存在或为空的输出目录：

```bash
# 预检：不加载权重或推理模型
/home/zchming/.venvs/georecon-vggt/bin/georecon run-vggt --manifest data/dtu_smoke_pinned/manifest.json --config configs/methods/vggt_raw_smoke.json --official-repo experiment03_depth_reprojection/vggt_official --checkpoint experiment03_depth_reprojection/model.pt --output outputs/stage2_vggt/my_preflight --preflight-only

# 原始推理：本地权重，无 BA/下载/过滤
/home/zchming/.venvs/georecon-vggt/bin/georecon run-vggt --manifest data/dtu_smoke_pinned/manifest.json --config configs/methods/vggt_raw_smoke.json --official-repo experiment03_depth_reprojection/vggt_official --checkpoint experiment03_depth_reprojection/model.pt --output outputs/stage2_vggt/my_run

# 独立 GT 评价
/home/zchming/.venvs/georecon-vggt/bin/georecon evaluate-poses --prediction outputs/stage2_vggt/my_run/prediction.json --reference data/dtu_smoke_pinned/reference.json --output outputs/stage2_vggt/my_run/evaluation.json
```

新机器可建立独立 Python 3.12 环境，按运行依赖锁安装 CUDA wheels，再 `pip install -e . --no-deps --no-build-isolation`。自行准备并固定官方 checkout 和已授权 checkpoint，作为 `--official-repo` / `--checkpoint` 显式传入；不在算法中写死本机路径。干净新机器的完整安装尚未实测，本次证明的是已披露 overlay 环境。原始 checkpoint 与原始数据不上传 Git。

52 项检查在独立 COLMAP CPU 环境执行：

```bash
/home/zchming/.venvs/georecon-colmap/bin/python -m unittest discover -s tests -v
```

该环境能执行原有 native COLMAP 空白图像检查；VGGT 的 CPU 单测不需导入 torch 或权重。GPU 推理和官方预处理检查单独记录，未混称为 52 次 GPU 测试。日志见 [tests.txt](tests.txt)，命令及退出状态见 [test_run.json](test_run.json) 和 [acceptance_commands.json](acceptance_commands.json)。

## 8. 限制、未完成项和停止点

- 只支持本次验证过的等尺寸输入、固定官方 crop、CUDA BF16/FP32 参数；缺少兼容能力时明确失败，不自动改 CPU、FP16、分辨率或图像数量。
- depth/point confidence 原值没有校准；point head 与 depth 反投影不被强行当成同一预测，也没有加权融合。
- 原始输出中的相机可用不等于经典注册成功或独立几何可靠。原始点图的完整像素覆盖不等于表面完整度。
- 单个三图场景不能给出场景成功率、规模能力、泛化或融合结论。MVS、扫描真值评价、困难条件和固定测试集均未执行。
- 暂未接入官方 BA、跟踪或额外优化；没有修改官方源码、训练模型、扩大数据或增加界面。

**完成此次 VGGT raw 子步骤后停止。** 若用户后续授权，可仍在阶段 2 内单独接入官方 BA，核实新增依赖/权重、预处理及成本；现在不开始，也不进入 Benchmark 或可靠性分析。
