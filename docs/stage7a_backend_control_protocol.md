# Stage 7A 后端四组受控方案 v1

**状态：设计冻结、待审；未批准实现或运行。当前留出安全替换路线停止。** 本轮仅做本地 API/安装包元信息与上游源码只读核对，制定本协议与[声明式 JSON](../configs/experiments/stage7a_backend_control_v1.json)。JSON 不是现有运行入口的配置。本轮没有调用 E/H 求解、三角化、特征/匹配或推理，没有进入 Stage 7B、提交或推送。基线提交 `42458b65aa679208117e76fbfcebdd517c88c940`；不修改旧协议与负结果。

## 1. 决策与研究问题

[留出诊断](stage7a_heldout_geometry.md)中，9 个 C 改善 B 的对识别 8 个，但 4 个 C 伤害 B 的对仍有 2 个被主指标错误偏向；网格均衡错误偏向其中 3 个。34/46 对可成对评价、12 对未知，包含约 63° 严重错误。停止当前指标的安全替换开发，不再调 cap、盐、来源权重或堆例外。诊断工具和全部未知/失败保留。

新问题：**固定匹配输入，改变 E 的稳健估计能否兑现已有匹配收益，并控制错误接受？** 现有证据没有证明失败由求解器造成；这是待证假设。接入已有估计器只能算工程改进，不能认领原创算法或稳定泛化。

## 2. 本地接口、坐标与阈值核实

只导入本地 cv2、读取 docstring/类型声明/构建信息、实例化并读取 UsacParams；没有调用 findEssentialMat 或其他几何求解接口。实验解释器实际复用原环境中的 cv2；未安装、更换依赖。

- Python 3.12.3；NumPy 2.5.2；opencv-python **5.0.0.93**，cv2 **5.0.0**。构建为 `5.0.0-dirty`，pthreads、LAPACK YES、Eigen NO；OpenCL 编译可用，未来执行必须关闭。cv2 二进制 SHA256 为 `f7746177b49e0368c73142ab92d3d5a4a9ce989aecce7178d7e05c58679349c9`。
- 本地同时暴露 `USAC_MAGSAC=38`、`SCORE_METHOD_MAGSAC=2`、`LOCAL_OPTIM_SIGMA=4`、`SAMPLING_UNIFORM=0` 和 UsacParams。枚举存在不证明求解调用已经成功。
- 参数重载的 Python 顺序是 `findEssentialMat(points1, points2, cameraMatrix1, cameraMatrix2, dist_coeff1, dist_coeff2, params[, mask])`，返回 E、mask。不要混用另一重载的 `K1,dist1,K2,dist2` 顺序。
- 本地默认参数为 confidence=.99、maxIterations=5000、threshold=1.5、sampler=UNIFORM、score=MSAC、loMethod=INNER_LO、loSampleSize=14、loIterations=5、isParallel=False、neighborsSearch=GRID、randomGeneratorState=0、final_polisher=COV_POLISHER、final_polisher_iterations=3。**默认 UsacParams 不是 MAGSAC。**

源码核对固定于 [opencv-python 构建 tag 93 的 opencv 子模块](https://github.com/opencv/opencv-python/tree/93)，指向 OpenCV `40738fb16ceddb5fb3fea747585f7ce6abb0605b`（5.0.0 tag）。只临时读取源码，不引入 vendor 或新数据。对应本地版本的上游来源已定位；`dirty` 构建不能据此证明 wheel 的所有 C++ 源码逐字相同。

| 证据源码（均在该 commit） | 本次核对的语义 | SHA256 |
|---|---|---|
| [five-point.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/geometry/src/five-point.cpp) | 参数重载、旧 E 的平方 Sampson 误差与内参处理 | `9b0ae89ad250ce9675c88630477475c36b589ba62fea598502183051d035f305` |
| [usac/ransac_solvers.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/geometry/src/usac/ransac_solvers.cpp) | 参数传递、随机状态、MAGSAC 预设、阈值和内部细化 | `e98fe63ea3e11dce4ed761f1a6da6ef1ca1a21b31d727b3282b1c623c3d118cc` |
| [usac/utils.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/geometry/src/usac/utils.cpp) | 双 K 校准含 skew，阈值除双相机平均焦距 | `9c68402324f794d98365cf95f56df1f3cdd629d6f3fac7553bc7e333294a4461` |
| [usac/estimator.cpp](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/geometry/src/usac/estimator.cpp) | Sampson 误差平方、内部 float32 运算 | `c95836c4a1e6e90d7c37516bc3884e2afabf6eb44d69e732718c793f9e6768e9` |

官方 [USAC 教程](https://docs.opencv.org/4.13.0/de/d3e/tutorial_usac.html)有助解释 MAGSAC 和种子，但其关于 Essential 误差为 symmetric geometric distance 的描述与此版本源码 `SAMPSON_ERR` 不一致，不能直接套用；以固定版本源码为设计依据，实际执行契约待单独授权验证。

定义 `fbar=mean(fx1,fy1,fx2,fy2)`。旧 E 使用完整 K 逆得到 float64 归一化点，传 I 与 `tau=1.5/fbar`；新 E 固定传**原缓存像素 float64 连续副本、完整 K1/K2、两个 None 畸变槽**，params.threshold=1.5。上游内部同样换算到 tau，再平方比较 Sampson 残差。本地缓存相机平均焦距约 1443.877，tau 约 0.00103887；无需增加/减少 0.5 像素，不改 800×600 图像或 K。

1.5 是未平方的公共像素名义容差，不是 sigma，也不是任意各向异性内参下严格等距的像素误差。新旧模型评分、局部优化、内部精度和内点掩码不能视为完全相同。E 的 MAGSAC 评分上界另由内部默认 2 换算，当前为 `2/fbar`，且有 `10*tau` 上限；sigma 权重还使用内部默认尺度。**不能把旧归一化点＋I 的调用机械换成 USAC 标志**，否则这些隐含尺度会改变。

新后端只选一版：MAGSAC 评分＋Sigma Consensus 局部优化，UNIFORM 采样；confidence=.999、maxIterations=10000、threshold=1.5、loSampleSize=50、loIterations=10、isParallel=False、neighborsSearch=GRID、randomGeneratorState=42/43/44。这匹配上游 E 的 USAC_MAGSAC 主要预设，并通过参数接口显式控制种子；不使用 PROSAC 或匹配分数排序。`cv2.setRNGSeed` 不能替代 USAC 自身 randomGeneratorState。

此 commit 的参数转接**没有传递**暴露的 final_polisher/final_polisher_iterations 字段；不设置这些字段或声称其生效。保留 ModelImpl 的 MAGSAC final polisher 和内部 final_lsq_iters=7。该内部细化属于所测第三方 E 估计包，不是新增教学 BA。比较对象是整套 E 估计器，无法归因到单独评分函数。

## 3. 四组、输入锁与唯一变量

| 组 | 原缓存匹配 | E 估计 |
|---|---|---|
| B_old | RootSIFT＋Ratio B | 原 reconstruct() RANSAC |
| B_new | 与 B_old 相同 | 上述固定 USAC/MAGSAC 配置 |
| C_old | RootSIFT＋LightGlue C | 原 reconstruct() RANSAC |
| C_new | 与 C_old 相同 | 与 B_new 相同 |

主对照 C_new−C_old；部署参考 C_new−B_old；B_new−B_old 检查一般后端效果；C_new−B_new 隔离同后端匹配收益。另报差上差 `(C_new−C_old)−(B_new−B_old)`，仅描述交互，不作显著性结论。B_new 有收益不能用于挽救 C_new 主门失败；若只有 B_new 改善，记录发现后停止，另审是否改目标。

复用 `outputs/stage7a_v1_after_guard_fix` 的 **46 对全量匹配**、24 份特征与已知 K；不使用留出拟合集/验证集，留出诊断的 4 个结构不足对也全部纳入。候选不足由原后端正常失败，不提前整对排除。保留全部重复方向/别名、索引顺序及原分数；不重新提取/匹配、筛点、去重、选姿态或添加回退。原始 46 个 case 和 8 份输入锁由旧协议 SHA 引用；reduced3 复用 normal5，不多算独立样本。

当前只读检查共 **235 个缓存文件**：bundle、input_hashes、implementation；24×(features.npz＋record.json)；46×2×(matches.npz＋record.json)。按根目录相对路径排序形成 SHA 映射，UTF-8 JSON 紧凑编码后 SHA256 为 `4c593e8db86bf65559624c9b65023d7219b9e3e4c47b2f87f72b104e74b729e7`，精确定义见 JSON。未来执行需重建锁、核对 record 内 SHA、像素与共享特征索引一致、K/坐标有效；不能只核对文件数。旧输入/源码/运行环境锁继续有效，不发布含个人路径的原 bundle。

只替换 E 估计。旧[教学源码](../src/georecon/teaching/two_view.py)及[参数](../configs/teaching/stage6_two_view.json)保留字节与原入口；未来候选实现须独立入口或明确扩展，不能覆盖旧函数的行为。四组 H 均用原像素 RANSAC、在 H 前重置原 RNG seed；最少匹配/E内点20、单应占优门、四姿态选解、正深度≥.6、视差≥.5°、双视图误差≤2px、接受点≥20均不变。新 E 的原生 mask 直接送原检查，不另统一掩码；H/E 比值门可能因此改变结果，完整记录而不事后调门。

## 4. 未来执行、评价与隔离（本轮未执行）

运行前先经单独授权完成至多32次合成几何契约调用：本地参数重载可调用、返回 E/mask 的形状及有限性、含 skew/不同 K 的坐标与阈值换算、参数种子隔离、已有退化与失败控制；不能用 GT 挑返回模型。发现接口/源码行为不符则 BLOCKED，先修订协议，不转用 F 拟合、标志接口或换库绕过。

随后46×4×3＝**552 个重建位置**，种子42/43/44全报；不足匹配等早退不一定实际调用 E/H。case 按原协议 scan、sharp/spaced/blur、ID 顺序，四组顺序按 JSON 的四种循环轮换，每组种子升序；串行执行、8 CPU线程、OpenCL关闭、USAC parallel=False。两个旧组均实际复跑，不能用旧数字拼四组结果。逐对种子核对历史 status/reason、全部内点/保留掩码与计数一致，接受 R/t 使用 atol=1e-9、rtol=0；不符则 BLOCKED，不先改基线来迁就新结果。

准备器只导出匹配、K、参数及无真值ID；求解进程禁止读参考姿态/GT、旧位姿/评价、模型与网络。全部预测落盘并退出后，由独立评价进程读 GT。旧输出复现检查在求解外完成，不能向求解反馈历史结果。

- 正确初始化：接受且旋转及**有向**平移误差都≤5°；错误接受：接受且联合误差 max(R角,t角)>5°，另报>20°严重错误。
- 主量是每物体 joint AUC@10：每seed在 normal5/reduced3/spaced3/blur5内平均，四条件等权，再三seed等权；AUC 定义 `mean(max(0,1-error/T))`，不是另一个积分实现。同步报告 AUC@5/20、正确率、接受数和错误接受数。
- 46对去重表保留每组每seed的计划行、拒绝/失败、救回/退化/新增错误接受和不可评价原因。拒绝、OOM、超时、未知均对全计划分母贡献0 AUC/0正确；GT未知不算错误接受，也不算安全。三个seed不是三个独立物体，不取最好seed。
- E/H 内点和三角化支持仅解释行为；GT极线相容性仍为代理，不是真实三维匹配精度。原7个受益、6个新增伤害（含2普通）、新留出4个伤害及12个未知均作固定失败索引，不能沿用留出标签给全量新位姿判对错。原约63°案例必须明确呈现。
- 成本只测本轮完整重建墙钟、RSS、准备/读写/会话成本。原匹配成本引用旧结果并标历史值，不伪装本轮测量。seed42的46个配对几何差报告中位数/p95（线性分位）；单次时延不当稳定速度Benchmark。没有新冷启动/权重加载/模型推理。

## 5. 预算与继续/停止条件

CPU线程≤8；GPU计算0、显存新增分配0，绝不因授权12GB就启动GPU任务；若有外部GPU使用仍不得突破12GB。单进程RSS≤8GiB、独立产物≤1GiB；单case单组含三seed≤180s、会话≤1800s、离线评价≤180s；未来合成契约≤32次调用/60s。数据/权重/模型下载0、训练0、付费资源0。输出新目录且拒绝覆盖；保留软件失败、超限与未执行行，不换参数重跑同一case。线程数和10000次抽样上限相同不意味着新旧实际工作量相同，内部LO/细化成本计入。

以下是有限缓存上的投入门，**全部通过才允许提议另行评审独立物体验证**；不自动进入Stage7B。浮点比较容差1e-12，整数计数无容差。

| 门 | 冻结条件 |
|---|---|
| G1 实际后端收益 | C_new对C_old：每物体宏AUC@10严格上升、宏正确率不降；46对上稳定正确数净增≥2，且每物体净增≥1。稳定正确指3/3seed正确，净增为候选稳定正确数减基线稳定正确数；部分seed变化另报，不挑seed。 |
| G2 控制伤害 | C_new对C_old及B_old分别比较：每物体/每seed错误接受总数不增加；各基线3/3正确的normal5对仍3/3正确；任何对/seed，若任一基线正确或拒绝，候选不得新增>20°错误接受。原63°例仍错误接受即未过此严重错误门。 |
| G3 部署参考 | C_new对B_old：每物体宏AUC@10与宏正确率均不下降，46对稳定正确数净增>0。仍列全部困难对退化，聚合门通过不等于逐对无伤害。 |
| G4 额外成本 | seed42完整几何C_new−C_old中位数≤1s、p95≤3s，同时公开B新旧成本。 |
| G5 完整有效 | 所有计划行、输入/环境/源码锁、合成契约、旧结果复现检查通过；无资源失败或关键GT未知。 |

若新后端只降低残差/增加内点，或有效完整结果未过质量门，记 **NO_GO_EVIDENCE并停止本候选**；不追补第二估计器、扫阈值、改接受门或加入反例规则。成本失败NO_GO_COST，资源失败NO_GO_RESOURCE；软件/输入/语义契约问题BLOCKED，外部中断/关键真值未知INCONCLUSIVE，均保留证据，未完成的质量门不能判通过。

scan1开发、scan6历史验证均已反复查看，没有盲测；本版不做参数选择，也不把“先跑scan1再跑scan6”包装成独立验证。即便所有门通过，也只记GO_FEASIBILITY；稳定性能提升仍需未来另获批准的未参与设计物体确认。既有安全替换路线不会因此自动恢复。

本轮贡献是明确一个可证伪、单一候选的后端对照及本地接口限制，没有增加算法能力或新性能证据。仅新增本协议和声明式配置，展示完整Diff后停止，等待审查；不新增阶段报告或结果表，不提交/推送。
