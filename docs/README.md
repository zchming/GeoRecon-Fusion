# 文档索引

[项目首页](../README.md) · [安装、命令与复现条件](reproduction.md) · [项目侧贡献与第三方边界](contributions.md#portfolio-contributions)

首页只展示核心实现与少量结果；本索引保留完整报告、指标、冻结方案与失败入口。历史报告中的“未实现”“未提交”等表示当轮状态；当前能力以源码和最新验收为准。下列原报告、JSON/CSV、锁和失败日志均保持原路径与字节，没有归档或删除。

## 按阅读目的进入

| 阅读目的 | 入口与边界 |
|---|---|
| 理解相机与结果协议 | [统一协议](protocol.md)：像素、K、world-to-camera、尺度、成功/部分/失败状态 |
| 运行完整小例子 | [CPU协议与评价示例](reproduction.md#独立完整示例协议与位姿评价)：数学fixture，不是模型推理 |
| 运行真实教学SfM/BA | [已知K入口与所需本地输入](reproduction.md#真实教学sfm需要已有本地输入)；[几何讲解](stage6/walkthrough.md) |
| 查看成果与本人贡献边界 | [贡献、代码定位及简历表述](contributions.md#portfolio-contributions)；底层模型、估计器和优化器属于第三方 |
| 复核研究结果 | 下表按研究问题进入报告与原指标；成功、拒绝、OOM、退化和不可评估共同保留 |

## 核心实现、基线与验收

| 主题 | 完整报告 | 指标或实现入口 |
|---|---|---|
| COLMAP接入与失败记录 | [COLMAP报告](stage2_colmap/report.md) | [适配器](../src/georecon/baselines/colmap.py)；自估内参与已知K教学链不直接排名 |
| VGGT raw | [预处理、坐标与验证](stage2_vggt/report.md) | [适配器](../src/georecon/baselines/vggt.py)；相机预测可用不等于重建准确 |
| VGGT官方跟踪/BA | [验证、成本与失败](stage2_vggt_ba/report.md) · [依赖及复现](stage2_vggt_ba/reproduce.md) | [原评价](stage2_vggt_ba/evaluation.json)；三图接通，五图跟踪OOM保留 |
| DTU基线小样本对照 | [冻结结果及失败分析](stage3/report.md) | [原始汇总](stage3/results.json)；仅scan1/6，不是完整DTU榜单 |
| 双视图教学几何 | [运行与退化控制](stage6/report.md) · [数学与中间量](stage6/walkthrough.md) | [摘要](stage6/summary.json) · [two_view.py](../src/georecon/teaching/two_view.py) |
| PnP与增量注册 | [结果与失败](stage6_incremental/report.md) | [评价](stage6_incremental/evaluation.json) · [incremental.py](../src/georecon/teaching/incremental.py) |
| 固定K教学BA | [前后对照及退化](stage6_ba/report.md) · [阶段6验收](stage6_acceptance/report.md) | [评价](stage6_ba/evaluation.json) · [失败目录](stage6_ba/failure_catalog.json) · [BA实现](../src/georecon/teaching/bundle_adjustment.py) |

## 研究对照与停止判断

| 问题 | 报告与冻结范围 | 可追溯结果及结论 |
|---|---|---|
| 无GT可靠性信号是否可信 | [可靠性诊断](stage4/report.md) | [评价](stage4/evaluation.json)；内部一致与真实正确不同，保留unknown及反例 |
| 选择、过滤、融合与联合优化 | [场景选择](stage5/report.md) · [点过滤](stage5_points/report.md) · [加权融合](stage5_fusion/report.md) · [联合优化](stage5_joint/report.md) | [阶段5消融验收](stage5_acceptance/report.md) · [消融指标](stage5_acceptance/ablation_comparison.json)；没有稳定收益 |
| 原502点是否值得继续融合 | [互补性与伤害诊断](fusion_diagnostics_502/report.md) · [观测留出及源视图混杂](heldout_fusion_502/report.md) | [诊断结果](fusion_diagnostics_502/results.json) · [留出评价](heldout_fusion_502/evaluation.json)；oracle只作事后对照，不能部署为规则 |
| 学习匹配输入是否兼容 | [SIFT＋LightGlue预检](sift_lightglue_preflight/report.md) | [接口烟测](sift_lightglue_preflight/smoke.json)；历史预检不是收益证明，正式对照见Stage7A |
| Stage7A：共享特征A/B/C | [冻结协议](stage7a_protocol.md) · [受控结果与反例](stage7a_results.md) | [逐图像对](results/stage7a_pairs.csv) · [逐条件](results/stage7a_conditions.csv)；局部救回伴随新增错误，稳定门未通过 |
| Stage7A：留出安全替换 | [留出可行性](stage7a_holdout_feasibility.md) · [留出几何诊断](stage7a_heldout_geometry.md) | [可行性表](results/stage7a_holdout_feasibility.csv) · [几何诊断表](results/stage7a_heldout_geometry.csv)；负结果与不可评估保留，路线停止 |
| Stage7A：鲁棒后端与MSAC | [候选边界](stage7a_backend_candidates.md) · [冻结后端协议](stage7a_backend_control_protocol.md) · [后端结果](stage7a_backend_control_results.md) · [采样对照](stage7a_msac_sampling.md) · [后续细化](stage7a_msac_refinement.md) | [后端对照](results/stage7a_backend_control.csv) · [波动](results/stage7a_backend_stability.csv) · [采样](results/stage7a_msac_sampling.csv) · [细化](results/stage7a_msac_refinement.csv)；小幅结果不作稳定收益宣称 |
| Stage7B：种子与PnP候选 | [实现、验收、消融及五来源对照](stage7b_initialization.md#stage7b-acceptance) | [种子结果](results/stage7b_initialization.csv) · [跨来源](results/stage7b_cross_dataset.csv) · [逐场景](results/stage7b_cross_dataset_scenes.csv) · [PnP汇总](results/stage7b_final_search.csv) · [AP3P逐项变化](results/stage7b_final_ap3p_pairs.csv)；保留两排序和可选AP3P，无统一默认优胜 |
| Stage7C：AP3P留出确认 | [冻结方案、前后结果与验收](stage7c_confirmation.md#stage7c-acceptance) | [全部组合](results/stage7c_confirmation.csv) · [汇总差异](results/stage7c_confirmation_contrasts.csv) · [全部配对变化](results/stage7c_confirmation_pairs.csv)；DTU局部恢复，TUM未复制，NO_GO_CONFIRMATION |
| Stage7D：轨迹及支持不足 | [冻结诊断协议](stage7d_support_diagnostic_protocol.md) · [诊断与停止决定](stage7d_support_diagnostic.md) | [支持漏斗](results/stage7d_support_diagnostic.csv) · [诊断入口](../scripts/diagnose_stage7d_support.py)；不可评估与首轮失败保留，STOP_TRACK_REPAIR_ROUTE |

## 历史环境与原始协议

五份审计中标记为SUMMARIZE的说明集中由本表导航，**不修改其原始字节，也不将它们合并成新的实验协议**。其他当前命令优先看[复现说明](reproduction.md)，原项目范围见[项目要求](project_brief.md)。

| 历史说明 | 阅读用途 |
|---|---|
| [早期环境设置](stage0/legacy_environment_setup.md) · [阶段0环境审计](stage0/report.md) | 查看当时环境与遗留状态，不能当作当前唯一安装指南 |
| [阶段1数据与协议验证](stage1/report.md) | 数据获取、数学fixture与早期小样本边界；数据准备会联网 |
| [阶段3原冻结协议](stage3/protocol.md) | 原DTU小样本划分、预算和评价；不重新定义测试集 |
| [阶段4原诊断协议](stage4/protocol.md) | 原无GT诊断范围与独立评价边界 |
| [阶段5原选择协议](stage5/protocol.md) | 原开发集规则、回退与失败分母 |

## 复现层级与来源保护

- **数学fixture：** 基础安装后可运行完整协议/评价示例；它不检验模型性能。
- **真实重建：** 需要本地图像、已知K或基线内参协议；VGGT/LightGlue另需固定第三方源码、环境与权重。原数据、权重、大缓存不在Git。
- **缓存诊断：** Stage7D需要已保存的record、map、关键点与匹配缓存及原路径/哈希；图连通不等于有效三维轨迹。
- **历史验收：** 可能直接读取`docs/`固定路径、源码锁和`outputs/`。例如[accept_stage6.py](../scripts/accept_stage6.py)与[增量检查器](../scripts/check_stage6_incremental.py)；增量旧锁对应[abc5f85版本](https://github.com/zchming/GeoRecon-Fusion-Geometry-Reliability-Aware-Hybrid-Multi-View-3D-Reconstruction/tree/abc5f85268d9b1f4b46a25f417fb5d01cac9a76b)，不能在最新源码上修改旧锁使它通过。部分历史脚本会写回报告，不应当作无副作用的日常检查命令。

[原始匹配图来源与SHA](reproduction.md#演示资源来源) · [CPU CI与跳过解释](reproduction.md#cpu-ci与跳过解释) · [项目工作约定](../AGENTS.md)

原始报告保留全部正负结果、OOM、拒绝、部分地图与来源记录。本索引只改变阅读入口；不代表完成新的实验、归档或算法验证。
