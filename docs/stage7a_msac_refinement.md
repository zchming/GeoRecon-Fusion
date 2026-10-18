# Stage 7A MSAC继续探索：减少错误接受，保留严重尾部退化

**新增待评审候选：C＋MSAC，E名义噪声容差1px。** 先按运行前规则仅用scan1选择，再评价scan6；两个物体平均AUC小幅上升，错误接受43→35（−18.60%），接受总数426不变。严重错误17→18、scan6最差seed下降，不能宣称全面稳定。原MSAC1.5px仍保留为优先参考，不自动部署或删掉旧候选。

用户授权保留MSAC并持续尝试，直到平均精度、波动或错误接受出现改善后再报告；本轮没有在前两批未达收益通知门时结束。总共预声明三批、九个变体，5520个重建位置均保留；下面同时公开无收益、微小混合收益与退化，不能把选择后的结果当独立测试或原创算法结论。

## 冻结设计与边界

每批原46对全量C缓存、24张公共图内容、已知K/800×600、种子42—51；四组每批1840位置。基线始终为之前优先保留的均匀抽样MSAC12/20/10000、E名义容差1.5px。只改变E估计内部一个控制项；H的1.5px、原四姿态/DLT、深度/视差、2px重投影门与最低支持点数均不变。没有匹配筛选、分数排序、B/C安全替换、BA、融合、训练、数据/模型下载或Stage7B。

| 批次 | 冻结候选（各自只变一项） | 写定配置 |
|---|---|---|
| v1 | LO样本12→50；LO次数上限20→50；RANSAC上限10000→50000 | [v1](../configs/experiments/stage7a_msac_refinement_v1.json) |
| v2 | LO方式改为INNER_LO、NULL、GC；其余参数不变 | [v2](../configs/experiments/stage7a_msac_refinement_v2.json) |
| v3 | **新E噪声容差实验**：1.5→.75/1/2公共像素；不改H或后续接受阈值 | [v3](../configs/experiments/stage7a_msac_refinement_v3.json) |

这是三个新的显式受控试验，不是修改原冻结Stage7A或回写旧结论。参数均在相应预测运行前写定；没有运行后改参数、挑seed或删case。v3明确记录effective E阈值、除平均焦距后的名义阈值及不变H阈值，不能把继承的legacy/H统计误写成候选实际E容差。平均焦距归一化仍是原近似，不声称严格各向同性像素距离。

每批CPU8线程、串行、OpenCL关闭；四组按case循环顺序，seed升序；会话≤1800s、单case单组≤180s、RSS≤8GiB、产物≤1GiB。OpenCV路径不调用GPU，12GB GPU授权未被用来改变算法实现。提高上限不代表实际执行更多内部迭代，本地API未提供完整内部工作计数。

所有输入/已知K/原始像素及原235文件聚合锁验证；worker只得到准备观测与配置，禁止读取GT、旧缓存/结果、权重和网络。预测落盘退出后单独评价。scan1与scan6均已反复查看，全部实验属于开发证据，不能包装成新物体盲测。

## 全部尝试，不只公布胜者

AUC沿用联合角误差max(旋转、有向平移方向)的线性定义，四条件等权后十seed等权；正确为接受且≤5°。每组事件分母46对×10seed=460，不是460个独立样本。reduced3只复用normal5，拒绝与未知不移出计划分母。

| 批次 | 配置 | scan1平均AUC10 | scan6平均AUC10 | 错误接受事件 | >20°事件 |
|---|---|---:|---:|---:|---:|
| v1 | base | 0.7481 | 0.7354 | 43 | 17 |
| v1 | sample50 | 0.7481 | 0.7354 | 43 | 17 |
| v1 | lo50 | 0.7481 | 0.7354 | 43 | 17 |
| v1 | iter50k | 0.7481 | 0.7354 | 43 | 17 |
| v2 | base | 0.7481 | 0.7354 | 43 | 17 |
| v2 | inner | 0.7512 | 0.7346 | 42 | 17 |
| v2 | none | 0.7481 | 0.7354 | 43 | 17 |
| v2 | graphcut | 0.7481 | 0.7354 | 43 | 17 |
| v3 | base | 0.7481 | 0.7354 | 43 | 17 |
| v3 | noise075 | 0.7528 | 0.7397 | 40 | 21 |
| v3 | noise1 | 0.7545 | 0.7381 | 35 | 18 |
| v3 | noise2 | 0.7357 | 0.7205 | 49 | 24 |

v1三种配置的全部460位置位姿/状态与基线相同，没有观测收益；不能据参数对象可读写就声称增加了有效计算。v2 INNER_LO只有很小混合变化：scan1 AUC+.0031、scan6−.0008、错误接受少1；NULL/GC位姿/状态与基线相同。没有将这些包装成稳定增益，按用户授权继续独立预声明下一批。

v3 .75px也有小幅平均收益，但严重错误17→21；2px两物体AUC均降、严重错误17→24。没有在看到scan6后改选另一容差。完整[552行逐对逐组结果](results/stage7a_msac_refinement.csv)保留三批全部5520位置的十seed状态、原因、正确性、R/t/联合误差、E/H支持与耗时。

## 为什么主候选是1px：开发集先选，再评价历史验证

v3运行前确定规则：开发集宏AUC不得比基线低超过.005、正确率不得低超过2百分点；符合者先最少严重错误，再最少错误接受，再最高宏AUC，精确并列优先base。只根据scan1计算，得到noise1；在首次解析scan6参考相机用于位姿评价之前将选择JSON落盘。输入完整性SHA校验会预读参考文件字节，但未用其数值选择配置；这不等于“从未读取该文件”。

选择文件记录eligible=base/noise075/noise1，noise2不符合。开发集noise1严重事件1（base2）、错误接受8（base12）。独立复算选择，并将scan6标签全部改成最差后仍得到完全相同选择，未实现逐对姿态选择。所有scan6控制组也公开，但主候选没有在看到其评价后重排。由于scan6历史上已被查看，这仍不是独立泛化证据。

| 指标 | 现有MSAC1.5px | 开发集选定MSAC1px |
|---|---:|---:|
| scan1平均AUC10 | .7481 | .7545 |
| scan6平均AUC10 | .7354 | .7381 |
| scan1宏正确率 | 79.83% | 81.08% |
| scan6宏正确率 | 77.58% | 78.33% |
| 正确位置/460 | 383 | 391 |
| 接受位置/460 | 426 | 426 |
| 错误接受 | 43 | 35 |
| 严重错误>20° | 17 | 18 |
| 十seed全部正确的不同图像对 | 31 | 33 |

错误接受相对减少18.60%，不是大规模拒绝换来的下降：接受数相同、正确数净增8。逐位置则是16个纠正、8个退化（相关重复，不作独立统计显著性解释）。平均AUC仅+.0063/+.0028，属于小幅提升，未达到预设“较明显平均增益”提示门；本次触发的是错误接受减少门。不能将这一结果称为大幅平均精度提升。

## 必须一同呈现的退化

scan1 AUC跨seed极差.0523→.0446，但scan6极差.0393→.0532，**最差seed AUC .7194→.7040**。严重事件17→18，最大联合误差147.57°→151.82°。scan1接受误差p95从5.06°降至2.89°、scan6从62.68°降至61.13°，与最坏风险的变化方向不同，不混为全面稳健。

| 图像对 | base正确数/10 | noise1正确数/10 | 具体变化 |
|---|---:|---:|---|
| scan1_sharp_001_004 | 7 | 10 | 最大误差5.94°→1.73° |
| scan6_sharp_001_004 | 7 | 10 | 最大误差12.76°→4.96° |
| scan1_sharp_001_005 | 6 | 7 | 正确数改善，但seed48严重错误113.79°→115.53° |
| scan6_blur_002_005 | 9 | 9 | seed51从原1.11°正确结果恶化为103.32°，不能被相同正确数掩盖 |
| scan6_spaced_010_020 | 0 | 0 | 错误接受9→10，最大147.57°→151.82° |

原scan1 sharp003–005两配置均10/10正确，原scan6普通受害对sharp001–004变为10/10；但并非所有普通对都得到保护。全部新旧伤害、原约63°宽间隔失败和历史留出未知case都保留在固定计划中；不能声称安全替换路线已恢复。

## 验证、成本、复现与贡献

- 三批5520位置全部完成，无软件/OOM/超时或资源失败；基线每批460位置复现，合计1380位置。独立只读核对原像素、H矩阵/掩码一致、位姿角误差/AUC/错误接受和严重计数，保存输出与源码存档SHA，无新拟合或DLT。
- v1前置5项、v2前置6项、v3前置8项契约测试通过；最终完整回归219项：218通过、1跳过、0失败/错误。跳过为缺可选PyCOLMAP的原生空场景测试，不计通过。新增8项覆盖单因素参数、失败分母、收益提示、不同K/skew实际求解、H固定、独立开发集选择与验证标签不影响选择。
- 三批预测会话合计97.78s（32.09/32.75/32.94），离线评价合计8.82s；前置契约合计2.17s，完整回归另12.56s。峰值worker RSS135.39MiB；三批产物约365MiB，GPU计算0。未测成端到端匹配/模型速度，不宣称部署加速。
- 1035个既有项目文件与7535个既有证据文件逐文件哈希不变。旧源码/配置/缓存、全部负结果、上次评价器失败与用户未跟踪实验均保留；没有改main、提交、推送或进入Stage7B。

[运行入口](../scripts/run_stage7a_msac_refinement.py)、[通用计分/开发集选择](../src/georecon/matching/msac_refinement.py)、[测试](../tests/test_stage7a_msac_refinement.py)。原始候选/掩码/读取记录/完整评价与每批实际源码副本位于Git忽略目录 `outputs/stage7a_msac_refinement_v1/`、`v2/`、`v3/`（后两者同完整前缀），前置契约各在对应 `outputs/stage7a_msac_refinement_contracts_vN/`。实现逐批演进，各批预测存档SHA分别校验，不能把当前v3源码冒充v1/v2实施版本。未提交实现的基提交42458b6与实际源码SHA分别记录。

复现需冻结环境、原匹配缓存和上轮十seed来源产物，只clone仓库不足以复现；新运行/评价均拒绝覆盖。下面使用当前代码与v3配置复跑，其他版本更换相应配置且使用新目录；精确历史实现另保留在各run/sources中。

```bash
PYTHONPATH=src python scripts/run_stage7a_msac_refinement.py contracts \
  --config configs/experiments/stage7a_msac_refinement_v3.json --output outputs/refinement_contracts_new
PYTHONPATH=src python scripts/run_stage7a_msac_refinement.py run \
  --source-run outputs/stage7a_backend_stability_v1 --cache outputs/stage7a_v1_after_guard_fix \
  --config configs/experiments/stage7a_msac_refinement_v3.json \
  --contract-report outputs/refinement_contracts_new/contracts.json --output outputs/refinement_new
PYTHONPATH=src python scripts/run_stage7a_msac_refinement.py evaluate \
  --source-run outputs/stage7a_backend_stability_v1 --cache outputs/stage7a_v1_after_guard_fix \
  --output outputs/refinement_new
```

| 证据 | SHA256 |
|---|---|
| v1/evaluation.json | `24c8c42ce7c88a6d05fffc750832a3d65c4e72d4617e448b5937f594eb47d50c` |
| v2/evaluation.json | `502a47ac055819754bef3a85ed2488e31ce66652ddb1c6ab62727d1b3b6be2e9` |
| v3/evaluation.json | `0c746e35a7adff90cc2b780efcafffe51949e4cb7ebfef7f04e05aa4067c284b` |
| v3/development_selection.json | `ff0a461a977410598e5f518eb5903b0fd885bd904bc4158476cbb2a9ec7c4113` |

累计贡献在原重建/评价、教学SfM与失败审计之上，增加预声明的MSAC消融、开发集噪声校准和错误接受下降证据。它是现成后端的工程校准，不是原创估计器，也未证明新物体泛化、稠密点云精度或全面稳定提升。现有MSAC继续保留，1px配置作为收益/尾部风险并存的新候选交由用户判断，不自动替换。
