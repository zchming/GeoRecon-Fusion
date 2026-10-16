# 阶段 5：最小加权融合验证

已完成点过滤退化复核、稀疏对应加权融合、开发集锁定与相同 DTU 划分验证。当前只有 normal5 的少数点发生融合，结果不支持默认启用或稳定收益主张。未进行联合优化、相机平均、新稠密 MVS 或重新训练。

## 前置复核与范围

重新计算前一子步骤全部8例的 raw/filtered 点指标以及正确点被删除、unknown 正确点数量，与发布结果完全一致，见 filter_audit.json。开发集 F1@2mm 18.312%→17.082%；reduced3 误删2370个采样评价中的正确点，其中2366个属于 unknown。此前失败日志仍在原目录。此次从**未过滤 raw**开始，没有融合证据的位置保留原坐标，不能把相对旧过滤的恢复算作融合收益。

固定 scan1=dev、scan6=test，light3，normal5/reduced3/spaced3/blur5，视图、图像、预处理和上游方法设置完全沿用阶段3。前序阶段已经查看过测试结果，本次是回顾性小样本验证，不是新的盲测。

## 对应、坐标与加权规则

直接复用既有 COLMAP 稀疏三维点及其真实二维轨迹，不新跑 SfM/BA。按 worker 文件名查回图像ID，避免把 native image_id 当作 manifest 顺序。COLMAP 观测 xy 先减0.5转换像素中心，再由 common 800×600 的逆变换回原图，经 VGGT original_to_model 变换到原生392×518网格，取最近整数像素。该离散化会引入至多约0.71原生像素误差；没有最近邻点云匹配、插入稀疏点或空间扩散。

推理 Sim3 仅来自两套**预测**相机：共同注册相机至少3个且中心非共线；对 R_rawᵀ R_colmap 求SO(3)弦均值确定旋转，再对相机中心以闭式正尺度/平移拟合。中心残差/原始相机中心直径≤0.1，最大朝向差≤5°。这只统一坐标规范，不是与真值对齐，也不联合调整相机或结构。低残差仍不能证明几何正确。

对应门限固定：轨迹长度≥3、COLMAP平均重投影误差≤2 common像素；锚点在raw相机中正深度，投影到真实观测的误差≤2原生像素；锚点与raw对应点三维距离≤raw源相机Z的5%。多个锚点落入同格时取经典权重最大者，平分取较小point ID，不混合不同对应。

令 N 为视图数，s 为原raw点在其他预测深度中满足5%相对Z差的支持数，L为轨迹长度、e为COLMAP拟合误差（common像素）：

`w_raw = 1 + s/(N-1)` ；`w_col = (L/N)/(1+(e/2)^2)`

`β = α*w_col/(w_raw + α*w_col)` ；`X_fused = X_raw + β*(S(X_col)-X_raw)`

这些是显式无量纲启发式权重，未经概率校准；COLMAP拟合误差来自自身轨迹，raw深度也是相关网络输出，不是独立正确性证明。融合只修正有明确对应的网格点，没有点删除；没有观察/无合格锚点/缺失或失败的COLMAP/对齐失败都保持raw。depth/confidence仍是原始诊断产物，未与新XYZ同步生成，不可当作融合后的深度或概率。

候选仅 α∈{0.25,0.5,1}，其余门限在开发评价前固定。按开发集宏平均F1@2mm选择 **α=0.5**，同分按候选顺序；raw与旧过滤作独立基线。另设相同对应集合、β恒0.5的无权平均消融。协议提交 f389361；规则提交 **89a0884**，随后才导出scan6轨迹并运行测试。没有用测试调权重或改变门限。只是一项显式实验工具，未自动接入默认场景选择。

## 失败尝试与数值修正

初版仅按预测相机中心确定旋转：4个开发条件都无对应，normal5锚点投影偏差约6–24原生像素。此开发尝试与原协议/源码/产物保留在 outputs/stage5_fusion_v1/failures/center_only_alignment_dev_v1。随后仅在开发阶段改用预测朝向确定旋转，局部门限与权重不变，得到normal5的174个对应点；reduced3仍无对应。

锁定规则后的首次全量输出使用float64，导致即使完全回退，旧Sim3评价中的NumPy缩放舍入也与float32原始点不同。已保留全套dev/测试输出及 attempt02_comparison.json，修复为输出沿用float32、未改点字节完全相同；**α与所有门限未改**。修复后在禁止访问测试文件的进程重算开发候选，最优规则和原始分数完全复现（fit_reproduction.json）。这一修正发生在首次测试评价之后，属于数值一致性修复，不是新盲测。

## 几何评价

沿用上一子步骤的自定义预算：最多200000个固定原始预测索引与200000个等距STL索引，参考取官方Plane上方算召回，预测用官方ObsMask算精度；同一raw相机中心GT Sim3用于所有版本，无ICP。报告1/2/5mm P/R/F1及无距离截断均值。移动点后的ObsMask重新计算，并附固定raw ROI敏感性分析；dev/test normal5仅3/8个样本改变ROI归属，其余为0。固定ROI结果保存在comparison.json。

这不是官方完整DTU评测：未执行0.2mm邻域降采样、60mm距离上限或20mm离群剔除，只覆盖两个场景的局部视图，不能与官方排名比较。参考数据和协议继续沿用上一子步骤的[DTU官方SampleSet溯源](../stage5_points/report.md)。

|场景|实际修正点|raw F1@2mm|旧过滤|无权平均|加权融合|
|---|---:|---:|---:|---:|---:|
|dtu_scan1_blur5|0|20.77242%|20.66502%|20.77242%|20.77242%|
|dtu_scan1_normal5|174|14.84911%|14.54485%|14.86254%|15.01778%|
|dtu_scan1_reduced3|0|12.29416%|9.48983%|12.29416%|12.29416%|
|dtu_scan1_spaced3|0|25.33303%|23.62924%|25.33303%|25.33303%|
|dtu_scan6_blur5|0|14.21529%|14.22115%|14.21529%|14.21529%|
|dtu_scan6_normal5|328|6.83022%|6.81765%|6.87182%|6.84447%|
|dtu_scan6_reduced3|0|20.03854%|20.04971%|20.03854%|20.03854%|
|dtu_scan6_spaced3|0|21.63589%|21.75222%|21.63589%|21.63589%|

dev 宏平均 F1@2mm：raw 18.31218%；prior_filtered 17.08224%；unweighted 18.31554%；weighted_fused 18.35435%。

test 宏平均 F1@2mm：raw 15.67998%；prior_filtered 15.71018%；unweighted 15.69038%；weighted_fused 15.68355%。

## 解释与失败边界

只修正 502 / 6497792 个网格点（0.00773%），6/8例完全回退。blur5因预测相机不一致拒绝融合，spaced3无COLMAP模型，reduced3无局部合格对应。规则在开发集仅由normal5提供有效区分；这不是广泛适用性的证据。

开发集加权优于同对应的无权平均；测试normal5则无权平均优于加权，权重优势没有迁移。宏平均的微小变化只来自各一个normal5，不作统计显著性或稳定收益主张。保守门限使影响范围极小；扩大半径、放宽门限或稠密传播属于后续方法变更，本次没有为追求收益而进行。

failure_analysis.json 列出所有被修正点到固定参考样本的距离改善/恶化、2mm正确点变错误点和最差点索引。它是锁定后的GT后验诊断，不参与推理或重新拟合。即使通过对应和权重检查，也会出现点距离恶化；例如开发normal5的174个修正点中，84个距离恶化，67个从2mm以内变为2mm以外（此处为全部修正点、未套ROI的后验统计，与主表采样/ROI范围不同）。保留这些反例。

## 资源、验证与产物

8次冻结融合耗时合计 8.2957s，native轨迹导出合计 0.1620s；包含文件校验、加载、压缩和落盘，融合计时也包含同对应无权平均消融的写出，不含进程启动和旧模型推理。CPU线程8，新增GPU分配0；未加载Torch、模型或新权重。峰值融合RSS、导出RSS、评价耗时与RSS见cost_accounting.json。整卡nvidia-smi占用另记于environment.json，不能当作本步骤分配。满足本地12GB预算。

成本另完整保留此前eager选择运行的所有分支、诊断、BA OOM成本，并加本次导出/融合；它是历史阶段计时相加，不是新在线端到端延迟，不支持速度提升主张。此次实际复用raw/COLMAP，不运行旧点过滤或BA；相应最小两分支历史计时也单列。开发候选与失败尝试不计入部署增量，但全部保留。

before：outputs/stage5_v1/<scene>/selected_artifacts/raw_predictions.npz；after：outputs/stage5_fusion_v1/frozen/<scene>/fused.npz。anchors.npz包含源网格索引、COLMAP point ID、对齐锚点、两种权重与raw跨视图支持；fused.npz包含xyz、原始索引和实际β；unweighted.npz保留消融。统一prediction.json保留raw相机、坐标系和原始depth/confidence，明确新pointcloud语义。Git发布轻量逐场景JSON/规则/代码/报告，大型NPZ与模型、数据保持本地。

验证包含闭式公式、零权重强度、不允许无效权重、无对应回退、退化/缺失相机、预测相机不一致、真实观测半像素变换、碰撞、负深度和错误几何门限、float32未改字节。96项不同测试在可靠性环境（跳过1项native COLMAP）和COLMAP环境（跳过2项OpenCV、1项SciPy）交叉覆盖；跳过项在另一环境通过。validation.json验证8例全部输入hash、相机不变、公式复算、未改点字节、无GT访问日志与开发规则复现。

## 复现

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/home/zchming/.venvs/georecon-reliability/bin/python
# 原有上游产物与data/stage5_fusion_tracks需在本地；所有生成命令拒绝覆盖。
$PY scripts/stage5_fusion.py generate --scene dtu_scan6_normal5 --output-root outputs/stage5_fusion_reproduction
$PY scripts/stage5_fusion.py fit --verify-frozen
$PY scripts/stage5_fusion.py evaluate
$PY scripts/check_stage5_fusion.py
$PY -m unittest discover -s tests -v
```

本子步骤完成后停止；不进行联合优化或扩展数据/资源配置。
