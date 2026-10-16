# 阶段 5：最小可靠性点过滤验证

已完成场景选择复核、开发集规则锁定、固定 DTU 划分的点过滤与几何评价。当前结果不支持默认启用过滤；它是显式调用的实验工具，原场景选择流程保持原输出。未进行加权融合、点移动、跨点平均或联合优化。

## 前置复核与规则

8 个自然样本都选择 VGGT raw；自然回退为 0。重新校验了阶段 5 输入锁、诊断文件、相机、点文件 SHA，以及发布的决策。此前 10 个开发集故障探针和 GT 隔离探针仍保留在 docs/stage5；其中模糊场景回退到 COLMAP 后质量差，内部一致性不等于正确。旧选择规则仅用开发集重新计算并复现，未改动。

协议先提交于 `47841d4`，规则与开发集报告于 `f54255c` 锁定，之后才下载 scan6 参考几何、运行冻结过滤和测试评价。扫描划分、图像、视图数、分辨率、模糊条件和所有上游算法设置沿用阶段 3。测试姿态在前序阶段已经被查看，本次是固定样本上的回顾性点几何试验，不是新盲测。

候选为相对 Z 阈值 {0.01, 0.025, 0.05} × 至少 {1, 2} 个其他视图支持。按开发集四个条件的 F1@2mm 宏平均选取，平分按候选顺序。最优过滤候选为 **0.05 / 1 个其他视图**。不过滤基线单独比较，没有根据测试集改阈值。完整 6 个候选结果及文件访问审计见 fit_report.json。

直接将原始 `world_points` 点头 XYZ 投影至其他相机，用目标深度四邻域双线性采样比较 `|z-d|/max(z,d)`。不使用源视图给自己投票，不以深度反投影点替换点头，不使用真值、网络置信度阈值或 GT 对齐参与过滤。原生网格为 392×518，整数像素中心。保留点坐标逐元素不变，C-order 原始索引可还原 frame/y/x。没有支持的点被排除；无有效观察单独记 unknown，不能称为已知错误。遮挡未建模，相关网络输出可共同出错。

## 固定评价协议

参考来自 [DTU 官方 SampleSet](https://roboimagedata.compute.dtu.dk/?page_id=36) 内 scan1/scan6 的 STL、ObsMask*_10 和 Plane；按 HTTP Range 分别设置 96 MiB 上限，校验 archive size、ETag、ZIP CRC 和本地 SHA。GT 与官方 MATLAB 代码仅在 data/ 保存，溯源见 input_lock.json。无新权重、依赖安装或推理。

仅在离线评价时，以预测/标定相机中心拟合 Sim3 到毫米坐标；前后复用同一变换，不做 ICP 或点云真值配准。每个源点云按原始索引等距取最多 200000 点，过滤后沿用同一组样本索引，不补采样。STL 等距取最多 200000 点；精度以官方 ObsMask 筛选预测点，完整度使用 Plane 上方参考点。参考→预测查询使用全部有限预测样本；预测→参考查询使用整个参考样本。报告未经距离截断的均值及 1/2/5mm P/R/F1，空输出计 F1=0、距离不可计算则 null。

这是一项预算受限的自定义点评价：未执行官方 0.2mm 邻域降采样、60mm 距离上限或20mm离群剔除，参考亦被采样，且只使用 3/5 张图覆盖局部表面。不可与官方 DTU 全数据排名直接比较。ROI/地平面规则对照官方 PointCompareMain.m，代码哈希已记录。

## 过滤前后（2mm，百分比）

|场景|保留率|P 前→后|R 前→后|F1 前→后|
|---|---:|---:|---:|---:|
|dtu_scan1_blur5|88.58%|21.72→21.72|19.91→19.71|20.77→20.67|
|dtu_scan1_normal5|88.61%|28.52→28.20|10.04→9.80|14.85→14.54|
|dtu_scan1_reduced3|84.02%|15.69→13.29|10.11→7.38|12.29→9.49|
|dtu_scan1_spaced3|69.28%|24.44→25.40|26.29→22.09|25.33→23.63|
|dtu_scan6_blur5|87.97%|18.93→18.96|11.38→11.38|14.22→14.22|
|dtu_scan6_normal5|87.10%|6.20→6.19|7.61→7.58|6.83→6.82|
|dtu_scan6_reduced3|81.07%|33.32→33.48|14.33→14.31|20.04→20.05|
|dtu_scan6_spaced3|67.07%|19.52→20.25|24.26→23.49|21.64→21.75|

dev 宏平均 F1@2mm：18.312% → 17.082%；P：22.593% → 22.152%；R：16.586% → 14.745%。

test 宏平均 F1@2mm：15.680% → 15.710%；P：19.495% → 19.721%；R：14.394% → 14.190%。

## 失败与限制

开发集四个条件的 F1 均下降；reduced3 降幅最大。支持不足会删除实际正确点，尤其视野边缘、遮挡和少视图区域。spaced3 的精度上升伴随覆盖下降。测试 normal5 轻微退化，其余变化很小，不能视作稳定提升：四个条件来自同一 scan6，且采样误差、配准误差和点头原有几何误差共同影响结果。failure_analysis.json 列出了 GT 后验确认的正确点被删除、错误点被保留及 unknown 中正确点数量。

规则是内部一致性筛选，不是正确性概率估计；相机正确也不能保证点头正确。当前不默认启用，不扩大候选、不用测试失败重新调参。COLMAP/BA 稀疏输出缺少兼容原生稠密深度网格，当前明确不支持这条过滤器；全拒绝/无有效观察由测试覆盖，不生成假点。

实际失败已保留：一次过滤完成后元数据序列化错误（outputs/stage5_points_v1/failures/attempt01）；一次 COLMAP 环境回归测试因缺 SciPy 失败，日志保留，之后仅该离线评价测试在此环境跳过，在可靠性环境执行。详见 failures.json。

## 资源、产物与验证

8 次冻结过滤耗时合计 4.7419s（加载、分块投影、压缩和落盘，不含历史模型推理、下载、离线 GT 评价和进程启动），单进程峰值 RSS 最高 0.1330 GiB。固定 CPU 线程 8、点批次65536；此轻量步骤没有加载 Torch/CUDA，新增 GPU 分配 0，满足 12GB 预算。nvidia-smi 的整卡占用包含桌面/其他进程，见 environment.json；不是本步骤显存。每个 run.json 保留实测耗时与RSS。

原始 before 点文件仍由阶段 3/5 路径和 SHA 引用；after 为 outputs/stage5_points_v1/frozen/<scene>/filtered.npz，包含 xyz/source_point_indices；evidence.npz 保留 finite/observed/support，可重建 keep/reject/unknown。统一 prediction.json 的相机不变，pointcloud 指向新子集，depth/confidence 保留原始产物。开发集六候选文件和失败目录均保留于 outputs。大型产物、GT 和上游权重不入 Git，Git 保存规则、代码、小型逐场景 JSON 与报告。

验证覆盖：点头与深度头不一致、不准自投票、边界/unknown/非有限/全拒绝、遮挡深度冲突、分块一致性、固定样本子集覆盖率单调性、空输出评价和 GT 读取拒绝。共87项不同测试分环境交叉覆盖：可靠性环境87项/跳过1项原生COLMAP，COLMAP环境87项/跳过2项OpenCV与1项SciPy评价测试。所有跳过项均在另一环境实际通过。全量回归日志见 regression_*.txt；validation.json 记录全部8场景逐点子集、字节哈希和相机不变校验。此子步骤完成后停止，不进入融合或联合优化。

## 复现（可靠性 Python 环境）

```bash
# 已有本地阶段3/5产物；输出已存在时命令拒绝覆盖。
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/home/zchming/.venvs/georecon-reliability/bin/python
$PY scripts/check_stage5_points.py
# 开发阶段先下载 scan1 并生成4条件的 --dev-candidates；fit 只运行一次，已冻结则拒绝覆盖。
$PY scripts/stage5_points.py generate --scene dtu_scan6_reduced3 --output-root outputs/stage5_points_reproduction
# 原始冻结输出上的评价与核验：
$PY scripts/stage5_points.py evaluate
$PY scripts/check_stage5_points.py --final
$PY -m unittest discover -s tests -v
```
