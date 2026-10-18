# Stage 7A 后端候选：保留 MAGSAC，新增 MSAC 待评审

**MAGSAC按用户决定保留为候选；MSAC提交用户判断取舍。** 当前46对缓存、10个固定种子上，MSAC＋LightGlue相对旧C的平均位姿精度明显提高，相对MAGSAC的平均AUC小幅提高、跨种子AUC极差约缩小42%。同时严重错误仍存在、最大误差更大，MSAC接传统B匹配时scan6退化。本轮不以单一严格门自动否定候选，也不宣称普遍稳定或可直接部署。

## 权限、设计与候选登记

用户明确要求保留既有改进、继续寻找稳定收益，出现收益/退化并存时交由用户判断。本次是Stage7A后端候选比较，不实现B/C安全替换策略，不进入Stage7B、融合、训练或BA。旧[四组协议](stage7a_backend_control_protocol.md)与[原结果](stage7a_backend_control_results.md)保持原字节；原NO_GO_EVIDENCE是当时严格门的历史结论，不改写为通过，也不再等同于“不可保留候选”。

| 候选 | 当前身份 | 已证实与待证实 |
|---|---|---|
| RootSIFT＋LightGlue＋MAGSAC | 用户已要求保留 | 固定样本平均精度收益；存在严重错误和随机敏感性 |
| RootSIFT＋LightGlue＋MSAC/LO | 本轮新增，待用户判定 | 本轮平均及较差种子指标改善；尾部和B路径退化必须一同考虑 |

运行前写定[新配置](../configs/experiments/stage7a_backend_stability_v1.json)，SHA256 `08a5b8f6a213423d08aaafa6d014d0709c7c296bf71eb52f388aa75816b941fc`。只新增一个估计器配置，不扫描阈值，不按GT挑种子或输出模型。B/C各运行old、magsac、msac三种后端；46×6×10=2760位置；种子固定42—51、全部保留。两个物体均已反复查看，新增种子只测固定匹配上的求解敏感性，不是未见场景或统计独立物体。

- MSAC采用固定上游USAC_DEFAULT风格：MSAC评分、INNER_AND_ITER_LO、均匀采样、LO样本12、LO迭代20；对应OpenCV commit `40738fb16ceddb5fb3fea747585f7ce6abb0605b` 的预设/ModelImpl默认。MAGSAC保留原配置。比较包含内部细化等整个估计器变化，不能单归因评分函数。
- 其余控制不变：confidence=.999、10000次上限、1.5公共像素名义阈值、双原始K含skew、原像素顺序及重复方向；原H检查/四姿态/DLT/接受门不变。无新提取、匹配、模型推理、数据/权重下载或点过滤。
- 同样复用235文件聚合锁、原输入/源码与cv2二进制锁。先完成合成契约，预测进程只接收复制的全量观测；禁止读取原缓存、历史结果、GT或网络。预测全部落盘退出后独立评价。
- 组顺序按case索引循环六组，CPU8线程、串行、OpenCL关闭；单case单组≤180s，会话≤1800s，RSS≤8GiB，产物≤1GiB。GPU计算0；保留12GB授权上限，不引入不受支持的GPU实现。
- 事前定义“值得报告”口径：两物体AUC10对旧C至少+.05，或对MAGSAC至少+.02，或严重事件至少减半且各物体AUC损失≤.02。仅用于提示，不自动接受/拒绝；所有维度和负结果仍报告。本轮两种C候选均触发对旧C收益提示，MSAC未达到对MAGSAC+.02或严重错误减半的双物体提示。

## 平均精度与随机敏感性

AUC与正确初始化均沿用旧定义：联合角误差max(旋转、有向平移方向)；正确要求接受且≤5°。四条件等权后对10seed等权；reduced3复用normal5，不增加独立样本。AUC不是准确率，不评价尺度或稠密点云。

| 物体 | 组 | 平均AUC@10 | seed最小—最大AUC@10 | 宏正确初始化率 | 错误接受事件 | >20°事件 |
|---|---|---:|---|---:|---:|---:|
| scan1 | B_old | 0.5429 | 0.5429—0.5429 | 68.33% | 50 | 30 |
| scan1 | B_magsac | 0.6485 | 0.5973—0.7027 | 69.83% | 42 | 25 |
| scan1 | B_msac | 0.6464 | 0.6169—0.6685 | 70.08% | 46 | 24 |
| scan1 | C_old | 0.6027 | 0.6027—0.6027 | 73.33% | 40 | 0 |
| scan1 | C_magsac | 0.7292 | 0.6846—0.7758 | 77.83% | 21 | 6 |
| scan1 | C_msac | 0.7481 | 0.7173—0.7697 | 79.83% | 12 | 2 |
| scan6 | B_old | 0.4760 | 0.4760—0.4760 | 60.83% | 80 | 30 |
| scan6 | B_magsac | 0.6000 | 0.4558—0.6752 | 64.67% | 51 | 24 |
| scan6 | B_msac | 0.5690 | 0.3883—0.6440 | 60.00% | 62 | 28 |
| scan6 | C_old | 0.5722 | 0.5722—0.5722 | 70.83% | 60 | 10 |
| scan6 | C_magsac | 0.7285 | 0.6952—0.7632 | 77.08% | 31 | 16 |
| scan6 | C_msac | 0.7354 | 0.7194—0.7587 | 77.58% | 31 | 15 |

事件分母是每物体23对×10seed=230位置；跨物体每组460位置。230/460都不是独立样本数。C_msac相对C_old：scan1平均AUC10 +.1455（相对+24.14%），scan6 +.1632（+28.52%）；宏正确率分别+6.50、+6.75个百分点。10个seed中两个物体的宏AUC及宏正确率均高于旧C。

对同样10seed的C_magsac，C_msac平均AUC仅+.0190/+.0069；较差seed的宏AUC分别由.6846→.7173、.6952→.7194。AUC极差从.0913→.0523、.0680→.0393，分别缩小42.65%/42.14%。这是观测到的随机敏感性改善，不能外推为新物体稳定性、显著性或任意seed保证。

| 全46对、10seed的C路径 | 旧C | MAGSAC | MSAC |
|---|---:|---:|---:|
| 正确 / 460计划位置 | 330 | 373 | 383 |
| 接受 / 460 | 430 | 425 | 426 |
| 错误接受 | 100 | 52 | 43 |
| >20°错误接受 | 10 | 22 | 17 |
| 10seed全部正确的图像对 | 33 | 30 | 31 |

MSAC相对MAGSAC：正确位置+10，错误接受−9（−17.31%），严重错误−5（−22.73%），严重受影响对7→5。相对旧C，正确位置+53，但严重错误10→17。所有10seed的MSAC正确对数为36—40，MAGSAC为34—41，旧C恒为33。不可将“平均/较差seed提高”写成逐对无伤害。

## 必须交给用户判断的退化

**第一，scan6尾部仍明显差于旧C。** 接受位姿的联合误差p95：旧C17.17°、MAGSAC61.49°、MSAC62.68°；最大误差63.37°→145.50°→147.57°。p95为接受结果条件分布，各组接受数量也公开，不用它替代全计划分母指标。scan1 p95则从旧C9.21°、MAGSAC10.89°降至MSAC5.06°，两个物体行为不同。

**第二，普通对有改善也有伤害转移。** 下表正确数均为/10seed；最大值仅对被接受姿态计算。

| 图像对 | 旧C正确数 | MAGSAC正确数 | MSAC正确数 | MAGSAC→MSAC最大误差 |
|---|---:|---:|---:|---|
| scan1_sharp_001_004 | 10 | 4 | 7 | 107.66°→5.94° |
| scan1_sharp_001_005 | 10 | 8 | 6 | 109.21°→113.79° |
| scan6_sharp_001_004 | 0 | 6 | 7 | 105.41°→12.76° |
| scan6_spaced_010_020 | 0 | 0 | 0 | 145.50°→147.57° |

scan1 sharp001–004消除了本次10seed里的>100°错误，但仍有3个seed略超5°；sharp001–005相对MAGSAC反而正确数减少、最大错误更大。scan6 spaced010–020两新后端均9次错误接受、1次拒绝，不能写成已解决。

MSAC全部5个严重受影响对为：scan1 blur001–005、sharp001–005；scan6 blur001–005、sharp001–005、spaced010–020。原6个伤害、7个受益、留出未知案例的所有case均保留在[276行逐对逐组表](results/stage7a_backend_stability.csv)，每行含10个seed的状态、原因、正确性、R/t/联合误差、几何耗时与E/H/接受点数；原始2760行也保留。没有只发布变好的案例。

**第三，传统B前端出现退化。** scan6 B_msac相对B_magsac：AUC10 .6000→.5690，宏正确率64.67%→60.00%，错误接受51→62、严重错误24→28，最差seed AUC .4558→.3883。说明本后端并非对所有匹配输入都更好；当前较好候选是C＋MSAC这一组合。不得把“替换第三方后端”包装为普适原创优化。

## 验证、成本与复现

- 2760位置完整，无软件/资源失败或GT未知；552个重叠的旧/MAGSAC位置与前次原结果复现，未复用旧数字冒充复跑。
- 独立只读核对所有落盘SHA、实际拟合像素、同匹配同seed的H矩阵/掩码完全相同，另写公式复算位姿角误差、AUC宏聚合和严重计数，不新增几何求解。worker文件访问记录核对无GT/模型/历史结果读取。
- 新增6项测试全通过；完整回归205项：204通过、1跳过、0失败/错误。跳过是缺可选PyCOLMAP的原生空场景测试，未算通过。前置9次profile入口调用含old/MAGSAC及早退，MSAC实际5次；全回归再次执行MSAC5次，总10次未超过16次新profile契约预算。
- 会话36.923s，离线评价3.681s，前置契约.697s，全回归11.253s分别报告。worker峰值RSS134.84MiB，运行产物约139MiB；CPU8线程、无CUDA/Torch/模型调用，未使用GPU。几何累计耗时C旧/MAGSAC/MSAC约10.04/6.82/6.50s，单次串行剖面、包含不同接受/拒绝路径，不宣称稳定部署加速。
- 1022个既有项目文件（含此前未提交候选）与2274个既有证据文件逐文件哈希未变。保留旧代码、协议、负结果、未跟踪用户实验和原复现入口，不删除/归档历史。没有提交/推送、合并main或进入Stage7B。

[运行入口](../scripts/run_stage7a_backend_stability.py)、[后端profile与计分](../src/georecon/matching/backend_stability.py)、[测试](../tests/test_stage7a_backend_stability.py)。完整原始产物与源码副本在Git忽略目录 `outputs/stage7a_backend_stability_v1/`，合成契约在 `outputs/stage7a_backend_stability_contracts_v1/`。实施代码未提交，运行基提交42458b6与实际source SHA分别记录。只clone仓库不能复现：还需要原缓存、前次四组产物与冻结本地环境；无需新权重或vendor下载。新输出目录均拒绝覆盖。

```bash
PYTHONPATH=src python scripts/run_stage7a_backend_stability.py contracts \
  --config configs/experiments/stage7a_backend_stability_v1.json \
  --output outputs/backend_contracts_new
PYTHONPATH=src python scripts/run_stage7a_backend_stability.py run \
  --source-run outputs/stage7a_backend_control_v1 --cache outputs/stage7a_v1_after_guard_fix \
  --config configs/experiments/stage7a_backend_stability_v1.json \
  --contract-report outputs/backend_contracts_new/contracts.json --output outputs/backend_stability_new
PYTHONPATH=src python scripts/run_stage7a_backend_stability.py evaluate \
  --source-run outputs/stage7a_backend_control_v1 --cache outputs/stage7a_v1_after_guard_fix \
  --output outputs/backend_stability_new
```

| 可审计证据 | SHA256 |
|---|---|
| provenance.json | `beb12181f19f5574e2acf90e63e6361178d3c72e194f3c3ab4f6b473ce5ced35` |
| session.json | `2dd267a408e42207d644a7570876ddaa0d66dffdea05b70bede73adf642861ba` |
| evaluation.json | `6cbb5dd65f41c62c16aea494ea49961aa029e28c12e0dde6a5e66cb38fd20c4b` |
| pairs.csv | `063e10c9803e4876e9c4e155590a8528830af549c6312812ee52541f455778a7` |
| independent_checks.json | `68e8b57b42d01c02f5eb54961162b15b47ed93acc4371f3fd3d31bb480311f62` |

累计贡献增加了第三种后端的可复用适配、可复核的10seed比较与收益/退化分离呈现。当前证明固定小样本的平均与随机敏感性改进，尚未证明未见物体泛化、稠密点云质量或原创算法优势。保留MAGSAC；MSAC由用户据上述取舍决定是否保留，继续研究需独立预声明下一项，不因结果好看继续扩展本次范围。
