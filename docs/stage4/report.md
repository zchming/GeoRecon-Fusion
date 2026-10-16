# 阶段4：不依赖真值的几何可靠性诊断

本阶段完成24个既有方法结果的复核、GT-free诊断实现、八个固定DTU实例的推理与独立GT关联验证。源代码与输入边界在计算前提交为`913863e`；配置和数据锁见`protocol.md`、`input_lock.json`。不输出正确概率或通用总分，不实施选择、回退、点过滤、相机优化或融合。

## 阶段3复核与勘误

24份预测与已发布JSON完全一致，产物摘要及重新计算GT评价一致；18条success、4条OOM、2条no_reconstruction全部保留。核对原生数据库图像ID后发现上一报告的宽间隔边名称误标：原生ID并非manifest顺序，实际保留001–010（scan1为894、scan6为857个验证匹配），020孤立。已修正阶段3报告及其生成脚本，数值与失败状态没有变化。详细ID映射见`stage3_audit.json`。

## 实现与真值边界

独立重新提取图像SIFT匹配，不使用被测相机/深度/置信度进行匹配或筛选；一半匹配拟合图像F，另一半审计，其与baseline特征位置可能重合，不能声称是严格统计独立留出观测。信号包括预测相机在这些观测上的sqrt Sampson距离（统一800×600像素单位）、条件DLT的双正深度比例，以及raw源深度投到另一视图后与目标深度的相对Z差。后者只证明联合模型内部一致，未处理遮挡；不是独立正确性证据。原始confidence只报告分布，未做概率校准。

输出层级为匹配级残差NPZ、视图对、视图和场景摘要；每个摘要附证据覆盖。没有完整三维轨迹或稠密逐像素正确率。只有少量/分布过窄的匹配、缺相机、零基线或方法失败时为unknown；不存在深度能力时not_supported。BA初始化深度没有冒充优化后深度。

八个推理进程只接收独立bundle，文件中无标定、reference或评价文件；Python访问钩子拒绝GT读取。真实文件访问拒绝测试通过。这是防误读的数据流约束，不是恶意代码沙箱。推理完成后，另一脚本`evaluate_stage4.py`才读取GT，核验固定指纹并评价。无真值阈值拟合、置信度校准或测试后重算信号。

## 逐实例观察

距离为“有效视图对的匹配中位数”再取中位数，正深度亦按有效对聚合。GT联合误差为所有可评价相机对的max(旋转、方向角)均值；对证据覆盖不足的实例，两者范围不完全相同。逐对GT及证据子集GT均值单独保存于`evaluation.json`，避免把一条好边当成全场景保证。

| 划分/实例 | 方法 | 诊断状态 | 有效/全部对 | Sampson px | 正深度比 | 跨视图相对Z差 | GT联合误差° |
|---|---|---|---:|---:|---:|---:|---:|
| dev/blur5 | colmap | partial_evidence | 9/10 | 0.498 | 0.9905 | — | 46.348 |
| dev/blur5 | vggt_raw | partial_evidence | 9/10 | 2.613 | 1.0000 | 0.00328 | 0.864 |
| dev/blur5 | vggt_ba | unknown | 0/10 | — | — | — | — |
| dev/normal5 | colmap | evidence_available | 10/10 | 0.168 | 1.0000 | — | 1.365 |
| dev/normal5 | vggt_raw | evidence_available | 10/10 | 0.653 | 1.0000 | 0.00186 | 0.451 |
| dev/normal5 | vggt_ba | unknown | 0/10 | — | — | — | — |
| dev/reduced3 | colmap | evidence_available | 3/3 | 0.145 | 1.0000 | — | 1.289 |
| dev/reduced3 | vggt_raw | evidence_available | 3/3 | 1.132 | 1.0000 | 0.00115 | 0.722 |
| dev/reduced3 | vggt_ba | evidence_available | 3/3 | 0.121 | 1.0000 | — | 1.175 |
| dev/spaced3 | colmap | unknown | 0/3 | — | — | — | — |
| dev/spaced3 | vggt_raw | partial_evidence | 1/3 | 0.308 | 1.0000 | 0.00728 | 0.511 |
| dev/spaced3 | vggt_ba | partial_evidence | 1/3 | 0.125 | 1.0000 | — | 0.868 |
| test/blur5 | colmap | partial_evidence | 9/10 | 0.365 | 0.9730 | — | 44.849 |
| test/blur5 | vggt_raw | partial_evidence | 9/10 | 1.015 | 1.0000 | 0.00256 | 0.955 |
| test/blur5 | vggt_ba | unknown | 0/10 | — | — | — | — |
| test/normal5 | colmap | evidence_available | 10/10 | 0.152 | 1.0000 | — | 1.045 |
| test/normal5 | vggt_raw | evidence_available | 10/10 | 0.662 | 1.0000 | 0.00141 | 0.453 |
| test/normal5 | vggt_ba | unknown | 0/10 | — | — | — | — |
| test/reduced3 | colmap | evidence_available | 3/3 | 0.141 | 0.9980 | — | 4.082 |
| test/reduced3 | vggt_raw | evidence_available | 3/3 | 0.551 | 0.9980 | 0.00092 | 0.330 |
| test/reduced3 | vggt_ba | evidence_available | 3/3 | 0.121 | 0.9980 | — | 0.657 |
| test/spaced3 | colmap | unknown | 0/3 | — | — | — | — |
| test/spaced3 | vggt_raw | partial_evidence | 1/3 | 0.900 | 1.0000 | 0.00867 | 0.620 |
| test/spaced3 | vggt_ba | partial_evidence | 1/3 | 0.130 | 1.0000 | — | 1.767 |

## 反例与适用边界

**低残差不能保证正确位姿。** 两个模糊COLMAP案例的独立匹配Sampson中位数约0.50/0.36px，双正深度约0.99/0.97，但平均旋转约14°、平移方向约45°。匹配约束的是图像极几何相容性；未知内参与退化观测可能容许错误的欧氏位姿。这里没有额外实验隔离原因，因此不将“内参退化”当作已证实唯一根因。正深度近1也不是高正确率。

**方法间不能用原始像素残差直接选赢家。** 同一模糊输入的raw模型GT位姿更准，却有更高Sampson残差。当前输出没有统一校准，不据此产生跨方法选择规则。

**匹配证据缺失要显式暴露。** 两个spaced3输入匹配图均有一个孤立视图，raw和BA仅1/3视图对有足够审计证据；场景诊断标partial_evidence，不能将已有一条边的低误差外推到全部相机。COLMAP无模型仍为unknown；五图BA的4次OOM也全部unknown，不用raw初始化结果补作成功。

**低深度差只是内部一致。** raw各实例跨视图相对Z差较小，仍未用独立三维真值验证深度准确率。遮挡、共同模型误差和背景匹配均可能影响这些信号；不宣称能检测所有失败或证明稠密可靠性。

## 开发集与固定测试划分的描述关系

下面仅在同方法、同划分内计算Spearman。主要误差目标为场景所有可评价相机对的联合角误差均值。higher residual通常预期对应higher error；正深度/confidence方向相反。但本轮不学习对应关系、不产生阈值或概率。BA每划分仅2个可用实例，不计算相关；恒定量也保留不可计算。

| 划分 | 方法 | 信号 | 可用实例 | rho |
|---|---|---|---:|---:|
| dev | colmap | epipolar_median_px | 3 | 1.000 |
| dev | colmap | positive_depth_ratio | 3 | -0.866 |
| dev | vggt_raw | epipolar_median_px | 4 | 0.800 |
| dev | vggt_raw | raw_cross_depth_error | 4 | 0.000 |
| dev | vggt_raw | raw_confidence_mean | 4 | -0.800 |
| test | colmap | epipolar_median_px | 3 | 0.500 |
| test | colmap | positive_depth_ratio | 3 | -1.000 |
| test | vggt_raw | epipolar_median_px | 4 | 1.000 |
| test | vggt_raw | positive_depth_ratio | 4 | 0.775 |
| test | vggt_raw | raw_cross_depth_error | 4 | 0.800 |
| test | vggt_raw | raw_confidence_mean | 4 | -0.800 |

raw的深度一致性与真实误差在dev的rho为0、test为0.8；正深度在test甚至呈与预期相反的正相关，不能建立稳定可靠性结论。Sampson或confidence的个别较强相关也仅来自3–4个条件、每划分一个对象。四种条件不是独立场景；不做显著性、泛化或校准主张。配置在计算这些相关前固定，没有据test结果调整。

沿用的scan6/test在阶段3已被观察过；本轮是同一固定划分的回顾性诊断验证，**不是新增未见测试集**。未用GT构造推理信号，但不能把这一点扩张成统计意义上的全新盲测。

## 成本与验收

复用阶段3预测，新增任务是图像匹配与小规模向量化几何，CPU更合适；未调用CUDA、未下载新模型，也未重跑baseline。GPU上限仍是12GB，本轮没有新增GPU分配。内存GPU字段null表示未做GPU测量；不冒充测得0。旧baseline推理成本仍见阶段3，不因复用而声称可在线零成本获得。审计、bundle复制和外层锁校验未统一计时，不把4.78秒当作从准备开始的部署总成本。共享图像匹配每实例只计算一次，三个方法分支另记时，没有重复加和/隐去共享成本。

| 实例 | 共享匹配秒 | 诊断内部总秒 | 完整进程秒 | OS RSS峰值GiB |
|---|---:|---:|---:|---:|
| dtu_scan1_blur5 | 0.119 | 0.365 | 0.515 | 0.219 |
| dtu_scan1_normal5 | 0.516 | 0.802 | 0.917 | 0.207 |
| dtu_scan1_reduced3 | 0.206 | 0.382 | 0.518 | 0.201 |
| dtu_scan1_spaced3 | 0.147 | 0.312 | 0.464 | 0.209 |
| dtu_scan6_blur5 | 0.125 | 0.359 | 0.515 | 0.211 |
| dtu_scan6_normal5 | 0.462 | 0.739 | 0.868 | 0.211 |
| dtu_scan6_reduced3 | 0.202 | 0.385 | 0.514 | 0.201 |
| dtu_scan6_spaced3 | 0.159 | 0.335 | 0.470 | 0.198 |

八个完整诊断进程合计4.782s，OS RSS峰值0.219GiB。GT评价及证据摘要检查另耗0.170s，不计为推理耗时。

原59项回归继续通过；新增10项在OpenCV环境全部通过，合计69项不同测试。完整COLMAP环境运行69项，其中2项OpenCV测试因依赖未装跳过，已在独立环境覆盖；见`regression_tests.txt`及`new_tests.txt`，不将跳过项伪称在该环境通过。包括对极符号歧义、正深度、错匹配、Sim(3)变换、目标深度改变、边界/空证据及真实GT访问拒绝。24份方法诊断均有状态；6个失败预测未被赋予低残差或高可靠，见`validation.json`。

## 复现与文件

新增锁为`requirements-reliability-lock.txt`。本机独立venv通过.pth只读继承已有科学计算环境，再以editable安装项目；不是重新安装全部运行库。新机器可用独立venv安装该锁和`pip install --no-deps -e .`。运行入口：

```bash
# 审计和准备分别执行；已锁定的bundle禁止覆盖
/home/zchming/.venvs/georecon-colmap/bin/python scripts/stage4_pipeline.py audit
# 首次才执行 stage4_pipeline.py prepare
/home/zchming/.venvs/georecon-reliability/bin/python scripts/stage4_pipeline.py infer
/home/zchming/.venvs/georecon-reliability/bin/python scripts/evaluate_stage4.py
/home/zchming/.venvs/georecon-reliability/bin/python scripts/report_stage4.py
# 单实例诊断也可用：
georecon analyze-reliability --bundle data/stage4_bundles/dtu_scan1_normal5 --config configs/reliability/stage4_v1.json --output outputs/new_stage4_attempt
```

现有输出目录禁止覆盖；本轮不重复推理挑最好结果。Git包含源代码、配置、锁、JSON证据与日志；图片、权重、匹配/信号NPZ保留本地data/outputs，发布JSON中的相对二进制路径以原运行目录为根。`observations.npz`保存观测坐标，`match_signals.npz`保存匹配残差，可重建诊断，不执行过滤。

本阶段到此停止。当前证据支持把几何量作为带缺失状态的诊断，而不足以直接形成可靠概率或选择策略；选择/回退与融合留待新的明确指令。
