# 阶段 2：VGGT 官方 BA 三图验证

本子步骤完成：新增依赖、权重和预处理已核实；固定版本官方跟踪与 Ceres BA 已在同一组 DTU 三图跑通；统一输出、完整运行成本和失败记录已保存。源码验收提交为 `38eccc0`，实际源码摘要和执行命令见 `acceptance_commands.json`。仍停留在阶段 2；没有开展 Benchmark、可靠性分析、融合、训练或 COLMAP MVS。

## 输入、依赖与权重

输入仍是 DTU scan1/light3 的 001、002、003，原图 1600×1200，公共输入 800×600。manifest SHA-256 为 `5753c59b26dc9389bbb2dc7a356144f98381d97aeccd40d77baa38652b609a00`；逐图公共 PNG 哈希与上一轮 raw 完全相同。GT 不传入 runner，仅评价时读取 reference。

依据[固定版本官方 demo](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/demo_colmap.py)及[依赖清单](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/requirements_demo.txt)，新增独立 BA 环境，使用 PyCOLMAP 3.10.0、PyCeres 2.3、Hydra 1.3.2、OmegaConf 2.3.0、Kornia 0.8.3、Trimesh 4.12.2 和固定版本 LightGlue/DINOv2。基础 Torch 2.11.0+cu128 等继续使用已验证的现代运行时，与官方历史 requirements 并非逐包一致；本次原生接口及真实 GPU 链路已验证，`pip check` 通过。没有替换既有 PyCOLMAP 4.2.1 baseline 环境。详见 `requirements-vggt-ba-runtime-lock.txt`、`environment.json` 与[复现说明](reproduce.md)。

原 VGGT-1B 权重复用本地 5,026,874,952 字节文件，SHA `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`。新增如下权重，完整 URL、固定版本、SHA 和来源记录见 `asset_preparation.json` 与 `configs/assets/vggt_ba.json`。

| 权重 | 字节 | 许可/来源说明 |
|---|---:|---|
| VGGSfM v2 tracker | 195346695 | HF 固定 revision；CC-BY-NC-4.0 |
| DINOv2 ViT-B/14 register | 346393545 | 官方 HTTPS；Apache-2.0 |
| ALIKED n16 | 2738091 | 固定源码 revision；BSD-3-Clause |
| SuperPoint v1 | 5206086 | 官方发布文件；限制性非商业内部研究许可 |

VGGSfM SHA 可与 HF LFS 元数据核对；另外三份 SHA 是首次取得官方文件后记录的字节摘要，不冒充发布方提供的校验值。源码、权重和数据不随 Git 重新分发。尤其不能将 LightGlue 的 Apache-2.0 许可误用于 [SuperPoint 权重](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE)。VGGT 原有模型的非商业条件保持适用。

## 预处理、坐标和输出核对

官方 BA 入口先将 800×600 图片上下各补 100 黑边，再 BICUBIC 缩放至 1024×1024；跟踪在此网格进行。VGGT 相机/深度推理再通过 bilinear、`align_corners=False` 缩至 518×518。独立 PIL 实现与官方 1024 输入的最大像素差为 **0**。上一轮 raw 的 518×392 crop 与此不同，不能把两轮差值全部归因于 BA。

默认参数保持：4096 查询点上限、8 请求查询帧（三图实际选择三帧）、ALIKED+SuperPoint、fine tracking 开启、可见阈值 0.2、最大重投影误差 8、独立 SIMPLE_PINHOLE 相机、seed 42。原脚本的无序 glob 在适配层固定为 manifest 顺序；否则第一帧、轨迹与 GT 映射可能错位。

跟踪坐标由官方转换器原样插入 PyCOLMAP，**不额外应用通用 COLMAP 的 −0.5 修正**。官方将 K 前两行直接乘 1024/518，保留该规则。统一输出从 BA 后、官方重命名/缩放前的 1024 模型取得 K、世界到相机 R/t；再逆变换实际 PIL 像素中心映射及公共缩放，回到原图坐标。SIMPLE_PINHOLE 的单焦距无损展开为协议 PINHOLE 的 fx=fy。投影代数核对最大差 `3.41e-13` 个方形网格像素。

保存以下产物（大型文件仅在本地 `outputs/stage2_vggt_ba/acceptance`）：

- `prediction.json`：3/3 优化后相机，原图坐标，任意尺度世界坐标；未提供“BA 优化后稠密深度”的虚假能力。
- `optimized_points.npz`：8220 个真正的 `reconstruction.points3D` 优化后点；与原生模型二进制读回的坐标逐字节数值一致。
- `tracks.npz`：15318 条原始候选轨迹及原始 visibility/confidence，仅作接口产物；非概率校准或可靠性结论。最终模型有 19509 条点观测。
- `square_initial.npz` 与 `pre_ba_prediction.json`：本次方形预处理的原始 forward 输出，明确不是优化后深度。
- `native_before_ba` / `native_after_ba`：原生 BA 前后模型；`sparse` 保留官方最终导出，包括其主点强制回图像中心的操作。

统一输出的 K 逆变换保留实际坐标关系，与官方最后一步重设主点的导出约定区别已在 preflight 中记录；R/t 不受该缩放步骤影响。官方 `sparse/points.ply` 使用优化前初始化点，**没有将它当成 BA 后点云**。Git 中这些 JSON 是验收证据快照，其中大型产物的相对路径以原本地运行目录为根，不表示二进制数据已上传。

## 三图结果

| 同一次方形预处理 | 平均相对旋转误差 | 平均平移方向误差 | Sim(3) 后中心 ATE |
|---|---:|---:|---:|
| BA 前原始相机 | 0.777335° | 0.759734° | 0.543879 mm |
| 官方跟踪 + BA | 0.498176° | 1.175405° | 0.431491 mm |

三对相机均评价，Sim(3) 仅在评价器中拟合，没有反馈到推理。旋转和中心 ATE 改善，平移方向误差变差；本例不支持“BA 全面提升”。这是单场景三图接入验收，不是 Benchmark 或方法优越性证据。没有评价稠密几何精度。Ceres 报告的优化 cost 从 1.02308 px 降至 0.123078 px，36 次迭代、Convergence；这是目标函数拟合信息，不等于独立准确率。优化后实际观测平均二维距离为 0.152692 px，也不等于对 GT 的误差。

## 完整耗时与内存

下面使用最终 acceptance，单位秒，GPU 段调用同步计时；**包含项不能重复相加**，逐项原始值见 `prediction.json` / `worker_state.json`。纯 forward 不与其他方法整条链路直接比速度。

| 项目 | 秒 |
|---|---:|
| 输入 staging、清单及源码核对 | 0.303456 |
| 运行库导入与 CUDA 初始化 | 0.672221 |
| 五份 checkpoint SHA 校验 | 2.395124 |
| 官方 demo 导入 | 1.005107 |
| VGGT 模型构造 | 4.510222 |
| 1024 预处理与独立像素核对 | 0.150331 |
| 同步 VGGT 相机/深度 forward | 7.236723 |
| forward + 初始输出保存 + 模型释放（含上项） | 7.290381 |
| 跟踪含模型初始化、特征、精细跟踪及输出 | 21.644259 |
| Ceres BA | 0.444361 |
| 官方 demo 全链路（含上述模型/推理/跟踪/BA） | 35.490645 |
| 整个 worker（含导入、权重核验及退出） | 40.101777 |
| runner 端到端总耗时（含 worker、staging、导出验证） | **40.449275** |

模型反序列化及 GPU 传输、原生转换/重命名/导出、进程启动/退出等开销均包含在完整墙钟内；嵌套 stage 和每份权重加载也单独留痕，未单独分段的剩余开销不被遗漏于总时间。GT 评价不在推理计时中，两个评价命令的完整进程耗时见 `acceptance_commands.json`。首次四份新增权重并行下载 **38.304902 s**，单独计入准备成本，不混入本地推理时间。依赖安装及源码克隆未统一计时，明确记为 null；不把此处推理总时长称为从空机器部署的总耗时。

| 内存项目 | 字节 | GiB |
|---|---:|---:|
| PyTorch GPU allocated 全程峰值 | 10077113856 | 9.385 |
| PyTorch GPU reserved 全程峰值 | 10192158720 | 9.492 |
| nvidia-smi 整卡使用量采样峰值，含桌面/驱动 | 10867441664 | 10.121 |
| worker OS ru_maxrss 峰值 | 11007148032 | 10.251 |
| worker RSS 采样峰值 | 10577215488 | 9.851 |

GPU 使用本机 RTX 4070 SUPER（物理 12878086144 字节，约 11.99 GiB）；PyTorch allocator 限额 10.5 GiB，为其他占用留出空间。GPU 整卡每 100 ms 采样，CPU RSS 每 50 ms 采样，采样峰值可能漏掉瞬时峰值；allocator 高水位和 OS ru_maxrss 另行报告，三者不是同一口径。神经网络运行在 GPU；PyCOLMAP 3.10 的官方 Ceres 默认 BA 在 CPU，0.444 s，不替换算法以伪称全 GPU。

## 显存适配、失败与验收

官方原调度保留完整 VGGT 模型并同时创建跟踪器，实际在精细跟踪时 OOM。`vggt_ba_12gb_smoke.json` 明确记录：forward 后用 meta 张量释放后续不再使用的 VGGT 参数存储，约 0.050 s，无 CPU 大拷贝；随后照原参数运行官方跟踪/BA。未减少图像、查询点或精细跟踪。固定版本官方源码 checkout 始终保持干净；适配发生在本项目入口及审计 hook。

所有尝试保存在 `runs.json` 和各自 worker 日志中，没有覆盖失败：attempt01 为无序 glob 核对失败；attempt02 为旧 SuperPoint 格式不支持 mmap；attempt03 为原模型常驻调度 OOM；attempt04_12gb 首次完整成功；acceptance 为提交后最终验收。前几次为开发工作树，最终 acceptance 的源码提交及源码字节摘要已固定。

官方 cuDNN benchmark 开启、deterministic 关闭；即使 seed 固定，运行也不保证位级一致。attempt04 为 8200 点，最终验收为 8220 点，均完整留痕，不挑选最优一次；报告使用最终验收数字。该调度适配保持算法配置，但不宣称运行数字必然逐位重现。

57 项 CPU 检查通过，覆盖既有回归、奇数黑边与像素中心、相机反变换、资源配置拒绝、权重 pin、OOM/timeout 的显式失败状态及不读取 GT；真实 GPU 和 PyCOLMAP 3.10 原生操作由本次验收覆盖。有限数组、点云/轨迹 SHA、相同输入哈希、源码摘要、原生点云读回均通过，见 `validation.json`。没有训练、自动调参或删去失败记录。

至此停止本阶段子步骤。下一步只建议先确认阶段 2 验收与后续范围；固定 Benchmark、失败分析及可靠性模块均需新的明确指令。
