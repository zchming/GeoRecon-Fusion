# 阶段5子步骤：最简整场景选择与回退

已完成阶段4复核、仅开发集确定规则、提交锁定后对固定8实例的整场景选择、独立评价与回退探针。运行源码及冻结规则提交为`131962e`。本次不做点过滤、加权融合、相机/点联合优化，也不将不同方法的相机补成一个结果。

## 阶段4复核与设计依据

24条诊断、已发布副本、观测/残差数组摘要、预测指纹和重新计算的GT角度/ATE一致。保留10条证据完整、8条部分可用、6条未知。模糊COLMAP有低Sampson却严重错误位姿；因此没有按跨方法最小残差、未校准confidence或任意加权总分选优。spaced3只有一条审计边有证据，不能外推全场景；无模型/OOM保持不可用。原生数据库ID→原图名称勘误仍保持正确。详见`stage4_audit.json`。

## 固定规则与开发集证据

规则是 **VGGT raw → COLMAP → VGGT官方BA**。依次选择首个身份/摘要一致、完整成功相机输出、至少一条几何证据且产物完整的候选。诊断全未知、失效/过期、部分注册、OOM、缺失或损坏产物均不能入选。partial_evidence可选，但输出警示，不赋“可靠”标签。全部不合格则abstain，no_reconstruction且不造相机。

开发集只比较三方法6种固定排列，唯一目标为4个开发实例平均joint AUC@10最大，失败按0计，定义继承阶段3。无连续阈值、残差排序、概率校准或条件标签分支。拟合进程禁止打开scan6/test和混合evaluation/results文件；仅dev真值用于确定排列，文件访问清单保留在`fit_report.json`。

| 优先级 | dev平均joint AUC@10 |
|---|---:|
| colmap → vggt_raw → vggt_ba | 0.670874 |
| colmap → vggt_ba → vggt_raw | 0.661935 |
| vggt_raw → colmap → vggt_ba | 0.936335 |
| vggt_raw → vggt_ba → colmap | 0.936335 |
| vggt_ba → colmap → vggt_raw | 0.664787 |
| vggt_ba → vggt_raw → colmap | 0.916073 |

两个raw首选排列打平，按预声明规范顺序COLMAP、BA、raw的字典序决定后续排列。所有真实dev实例均选raw，因此**不能从这些自然样本学习或验证最佳回退次序**。规范平局规则不冒充已学到的资源优化或可靠性保证。`configs/selection/stage5_v1.json`在测试选择前已提交，测试后没有重拟合。`--verify-frozen`仅从dev重算并确认同一结果，不改原policy或fit_report；验证记录见fit_reproduction_check.json。

该DTU划分仍为scan1/dev、scan6/test，每划分只有一个独立对象和4种相关条件；阶段3/4已看过scan6结果，本轮不是新增盲测。不用此小样本作泛化或显著性主张。

## 固定8实例结果

| 划分/实例 | 所选整场景方法 | 自然回退 | 警示 | joint AUC@10 | 旋转均值° | 方向均值° | ATE mm |
|---|---|---|---|---:|---:|---:|---:|
| dev/blur5 | vggt_raw | False | 部分证据 | 0.914 | 0.422 | 0.769 | 1.618 |
| dev/normal5 | vggt_raw | False | 无 | 0.955 | 0.332 | 0.360 | 0.592 |
| dev/reduced3 | vggt_raw | False | 无 | 0.928 | 0.392 | 0.722 | 0.424 |
| dev/spaced3 | vggt_raw | False | 部分证据 | 0.949 | 0.229 | 0.511 | 0.480 |
| test/blur5 | vggt_raw | False | 部分证据 | 0.905 | 0.300 | 0.945 | 1.918 |
| test/normal5 | vggt_raw | False | 无 | 0.955 | 0.240 | 0.429 | 0.887 |
| test/reduced3 | vggt_raw | False | 无 | 0.967 | 0.330 | 0.146 | 0.214 |
| test/spaced3 | vggt_raw | False | 部分证据 | 0.938 | 0.389 | 0.536 | 1.191 |

| 划分 | 方法 | 完整成功/4 | joint AUC@5/10/20 | 旋转均值° | 方向均值° | ATE均值mm |
|---|---|---:|---|---:|---:|---:|
| dev | colmap | 3/4 | 0.367/0.434/0.467 | 5.281（n=3） | 16.239 | 10.081 |
| dev | vggt_raw | 4/4 | 0.873/0.936/0.968 | 0.344（n=4） | 0.590 | 0.779 |
| dev | vggt_ba | 2/4 | 0.398/0.449/0.474 | 0.660（n=2） | 0.930 | 0.993 |
| dev | selector | 4/4 | 0.873/0.936/0.968 | 0.344（n=4） | 0.590 | 0.779 |
| test | colmap | 3/4 | 0.244/0.372/0.436 | 5.855（n=3） | 16.506 | 12.725 |
| test | vggt_raw | 4/4 | 0.882/0.941/0.971 | 0.315（n=4） | 0.514 | 1.053 |
| test | vggt_ba | 2/4 | 0.379/0.439/0.470 | 0.734（n=2） | 1.165 | 0.916 |
| test | selector | 4/4 | 0.882/0.941/0.971 | 0.315（n=4） | 0.514 | 1.053 |

**8例全部选raw，真实回退0次；精度与always-raw完全相同，没有精度收益。** 四例带部分证据警示。注册/完整输出不代表真实正确率。相机列表、世界单位及坐标系与所选源结果一致，点文件字节SHA相同；产物使用硬链接且流程仅读取，按不可变文件使用，后续编辑应另存新文件而非就地写入；相对位姿与评价专用Sim(3) ATE逐项与原raw评价一致，没有在选择中做GT对齐。所有baseline原失败继续保留于比较表，不算成选择器修复成功。

## 开发集故障注入：可执行不等于准确

以下10个探针仅改变内存副本中的候选是否可见或证据状态，未改原快照；GT只用于探针后的dev评价。不是新增真实失败或自然回退，不加入上述8例分母，也不用于重新拟合规则。

| dev实例 | 模拟条件 | 结果 | joint AUC@10 |
|---|---|---|---:|
| blur5 | raw_missing | colmap | 0.000 |
| blur5 | raw_and_colmap_missing | abstain | 0.000 |
| normal5 | raw_missing | colmap | 0.863 |
| normal5 | raw_and_colmap_missing | abstain | 0.000 |
| reduced3 | raw_missing | colmap | 0.871 |
| reduced3 | raw_and_colmap_missing | vggt_ba | 0.883 |
| reduced3 | all_missing | abstain | 0.000 |
| reduced3 | all_unknown_evidence | abstain | 0.000 |
| spaced3 | raw_missing | vggt_ba | 0.913 |
| spaced3 | raw_and_colmap_missing | vggt_ba | 0.913 |

raw不可用时，blur5回退到COLMAP但joint AUC@10=0，其旋转约14°、方向约46°；这条明确反例说明当前可用性门控不保证回退准确。reduced3屏蔽raw+COLMAP后能用原BA整场景；normal5/blur5的BA原已OOM，因而在前两方法被屏蔽后弃权。全缺失或全部证据未知也弃权，而非拼相机、复制初始化深度或设置高可靠分数。

开发集GT隔离导出探针在输入副本中加入无效reference/evaluation/calibration，选择决定、全部相机字段和产物摘要不变；见`gt_isolation_probe.json`。初次探针脚本将哨兵写入放在dev拟合拒读钩子内，被禁止打开evaluation.json，在执行选择器前终止。已保留这次脚本错误及不完整输入；改由独立进程准备哨兵后通过。策略、规则及8例真实输出未改动，也未重跑选最优。

## 成本：没有隐藏额外分支

本版本是**eager快照选择**，不是先运行raw、必要时才运行后续模型的懒执行调度器。实际新增计算只有CPU读元数据、摘要核查与原文件重定位，没有新增GPU分配或模型推理。GPU上限继续12GB；原始GPU分支使用阶段3的预算与历史记录，不能说本轮重测了显存。

以下成本为三个原baseline完整CLI墙钟（包含失败/OOM）＋共享阶段4完整诊断进程＋本轮选择完整进程。它是按历史实测段相加的回放核算，**不是重新测得的在线端到端模型墙钟**。准备、预审计和外层输入锁校验未统一计时，不计作部署总成本。单独报选择耗时不能当作混合管线速度提升。

| 实例 | 新选择进程秒 | eager核算总秒 | 原raw完整CLI秒 |
|---|---:|---:|---:|
| dtu_scan1_blur5 | 0.164 | 52.989 | 14.634 |
| dtu_scan1_normal5 | 0.164 | 50.734 | 11.678 |
| dtu_scan1_reduced3 | 0.164 | 54.246 | 11.330 |
| dtu_scan1_spaced3 | 0.167 | 53.889 | 10.879 |
| dtu_scan6_blur5 | 0.164 | 49.187 | 11.379 |
| dtu_scan6_normal5 | 0.164 | 50.694 | 11.334 |
| dtu_scan6_reduced3 | 0.164 | 53.836 | 10.628 |
| dtu_scan6_spaced3 | 0.164 | 53.296 | 10.679 |

八次新增选择进程合计 **1.314s**，OS RSS峰值0.037GiB；eager核算合计 **418.870s**，原raw合计92.541s。当前实现既未提高这8例的精度，也不支持节省端到端成本的主张。独立GT探针另耗0.108s，不并入真实8例选择耗时。

## 检查与复现

原69项回归与新增10项选择检查合计79项不同测试已交叉覆盖。完整COLMAP环境79项有2项OpenCV检查跳过；完整reliability环境79项有1项原生PyCOLMAP检查跳过，均在另一环境执行通过。选择单元检查包括优先级而非最小残差、两级回退、全未知弃权、部分证据警示、部分相机拒绝、过期诊断、GT角色拒绝、损坏产物回退、源相机/点文件保留和拟合禁止读test。独立实际8例验证了决策重算、所有摘要、输出字节、指标一致与成本加和。

复用既有reliability CPU环境，无新依赖或模型。首次准备/拟合及执行入口如下，已有锁/策略/输出禁止覆盖；若做新版本必须显式建立新输出及新验收来源，不把重跑当成本轮原始证据。

```bash
python scripts/stage5_prepare.py audit
# 验证已冻结的开发集拟合，不改policy：
python scripts/fit_stage5.py --verify-frozen
# 首次新版本才准备bundle并运行不带--verify-frozen的拟合
# 上面完成后应提交固定policy，再运行：
python scripts/run_stage5.py
python scripts/probe_stage5.py
python scripts/evaluate_stage5.py
python scripts/report_stage5.py
# 单实例入口：
georecon select-scene --bundle data/stage5_bundles/dtu_scan1_normal5 --policy configs/selection/stage5_v1.json --output outputs/new_scene_selection
```

本机解释器为`/home/zchming/.venvs/georecon-reliability/bin/python`；原生COLMAP回归使用其独立环境。`fit_report.json`保存全部6候选与dev文件访问清单；`decision.json`保存每候选拒绝原因；`cost_accounting.json`保存每条历史分支开销；GT评价仅在独立脚本中读取。Git中只发布代码、策略和JSON/日志，小型证据的二进制相对路径以原本地输出目录为根；数据、权重和点云NPZ不进入Git。

本次完成最简场景级选择/回退子步骤后停止。部分证据、raw首选失效时的回退准确性和小样本泛化仍是局限；没有开始点过滤、加权融合或联合优化。
