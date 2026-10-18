# Stage 7A：MSAC优先候选与PROSAC排序尝试

**当前优先研究候选：RootSIFT＋LightGlue＋均匀抽样MSAC（原索引顺序）。** 用户明确决定：MSAC相对MAGSAC有收益则优先保留，并继续尝试。MAGSAC及旧RANSAC保留作历史对照，不删除、不覆盖、不自动部署。本轮完成一次预先固定的PROSAC/分数排序尝试，结果未改善，不升级为优先候选。

## 优先级依据与范围

此前[同10seed比较](stage7a_backend_candidates.md)中，MSAC相对MAGSAC：两物体平均AUC10分别.7292→.7481、.7285→.7354；跨seed AUC极差缩小约42%；错误接受52→43、严重错误22→17。平均精度优势小于相对旧C的提升，个别普通对及最坏误差仍退化。当前优先级是用户依据取舍作出的研究选择，不代表所有指标占优、默认产品配置已更改或已证明泛化。

本轮只在Stage7A后端内尝试一项：用缓存LightGlue分数给MSAC的E抽样排序，采用PROSAC。没有B/C位姿安全替换策略、融合、点过滤、BA、训练、VGGT或Stage7B；没有下载新数据、权重或重新匹配。GPU授权上限12GB，当前OpenCV路径使用CPU，无CUDA/Torch调用。

运行前固定[协议配置](../configs/experiments/stage7a_msac_sampling_v1.json)，SHA256 `a862552d9350069f8f4c97239ee0e8ee29a1e136d7b25546bd0406f06132750a`。只用原46对全量C匹配、同一24图、800×600与已知双K、42—51十个种子；两个物体都已反复查看，属于开发/历史验证，新增实验不构成未见场景证据。1840位置全部执行，不扫阈值或挑种子。

| 组 | E抽样 | E输入顺序 | 用途 |
|---|---|---|---|
| U_lex | 均匀 | 原索引 | 优先MSAC基线，要求复现之前全部460位置 |
| U_score | 均匀 | LightGlue分数降序 | 单独检查排序影响 |
| P_lex | PROSAC | 原索引 | 无有效质量排名的机制对照，不提议用于部署 |
| P_score | PROSAC | LightGlue分数降序 | 唯一新候选，与U_lex主比较 |

分数与匹配索引30261项逐项对齐，有限且在[0,1]，等于缓存matching_scores0对应项；它们不是校准的正确概率。分数并列时按原(i,j)排序。所有匹配都保留，只重排E调用输入，输出mask映射回原索引；H/位姿/三角化继续使用原像素与恢复mask。四组MSAC评分、LO样本12/迭代20、10000次上限、confidence=.999、1.5像素名义阈值等均相同，仅sampler与E顺序按2×2矩阵变化，不用会改变其他LO设置的USAC_PROSAC整包标志。

单case单组≤180s、总会话≤1800s、CPU8线程、OpenCL关闭、RSS≤8GiB、产物≤1GiB；四组按case循环轮换、seed升序、串行执行。预测进程只接收准备观测/分数/K/参数，禁止读取GT、历史运行与网络；落盘后独立评价。通知门预先设为：两物体AUC各+.01、或正确率各+2百分点、或严重事件减少至少1/3且各物体AUC损失≤.01；仅作报告提示，不自动裁决候选。

## 结果：未获得新增收益

AUC为联合角误差的原线性AUC，拒绝对全计划分母贡献0；四条件等权、再十seed等权。正确要求接受且旋转/有向平移都≤5°。表内错误事件分母为每物体23对×10seed=230，相关对和seed不是独立物体重复。

| 物体 | 组 | 平均AUC10 | seed最小—最大 | 宏正确率 | 错误接受事件 | >20°事件 |
|---|---|---:|---|---:|---:|---:|
| scan1 | U_lex | 0.7481 | 0.7173—0.7697 | 79.83% | 12 | 2 |
| scan1 | U_score | 0.7541 | 0.7300—0.7721 | 80.58% | 11 | 2 |
| scan1 | P_lex | 0.6101 | 0.4934—0.6634 | 65.67% | 60 | 55 |
| scan1 | P_score | 0.4823 | 0.4104—0.6087 | 51.00% | 60 | 29 |
| scan6 | U_lex | 0.7354 | 0.7194—0.7587 | 77.58% | 31 | 15 |
| scan6 | U_score | 0.7240 | 0.6924—0.7482 | 75.58% | 38 | 19 |
| scan6 | P_lex | 0.6113 | 0.4835—0.7094 | 65.08% | 58 | 51 |
| scan6 | P_score | 0.4175 | 0.3325—0.5275 | 43.75% | 92 | 68 |

| 全46对×10seed=460位置 | U_lex | U_score | P_lex | P_score |
|---|---:|---:|---:|---:|
| 正确 | 383 | 378 | 299 | 252 |
| 接受 | 426 | 427 | 417 | 404 |
| 错误接受 | 43 | 49 | 118 | 152 |
| 严重错误>20° | 17 | 21 | 106 | 97 |

P_score相对优先MSAC的AUC10变化为−.2659/−.3179，错误接受43→152、严重错误17→97。没有可保留的跨物体平均收益或尾部改善。U_score在scan1小幅+.0060，但scan6−.0114、严重错误17→21，不因单物体变好更换基线。P_lex也明显退化，不能把它当作公平的“质量引导有效”证据。

完整[184行逐对逐组表](results/stage7a_msac_sampling.csv)每行包含十seed状态、原因、R/t/联合误差、正确性、E/H内点、保留点和耗时；全部1840位置与失败候选均保留。严重错误、普通退化、原约63°案例和留出不可评估case仍在固定分母中。结论仅限制当前缓存、阈值与排序/采样组合，不证明PROSAC普遍无效，也未将具体成因归为平面、重复纹理或模型分数错误。

## 软件失败、验证与成本

求解进程正常完成、没有OOM或资源触限；原几何拒绝全部保留。**首次离线评价发生软件失败**：复用的评价器错误地向较新来源目录读取不存在的cache_lock.json。完整stderr保留，未当作算法负结果或覆盖成功预测。

修复仅在评价侧按同一235文件定义重建缓存锁、核对原聚合SHA；没有重复1840次求解或修改参数。原预测脚本和其他源码副本保留，评价结果明确分别记录预测脚本SHA `29ab36d1b3ce44b1619d70c3990cf0d9a22c54aefa61f4b5086702edad4543dd` 与修复后评价器SHA `314e8d068115b9c93c7707099bf1167955c4f4537fe2d13889615f067d0756c1`，仅允许这个已知脚本版本的离线修复例外，其他依赖漂移仍拒绝。worker/run/guard函数的AST与预测存档完全相同；不把修复后的代码冒充原预测版本。

- 前置5项契约通过；新增1项缓存锁回归，验证不依赖来源目录cache_lock.json、锁覆盖235文件且能检测改动/缺项。全回归211项：210通过、1跳过、0失败/错误；跳过为缺可选PyCOLMAP的原生空场景测试，不算通过。修复前后软件记录均保留。
- 基线U_lex全部460位置复现；独立只读复核1840位置的原像素、Python独立排序、native mask反置换、H矩阵/掩码一致性、位姿角误差、AUC宏聚合及严重计数，无新求解。worker访问记录无GT/历史运行/权重读取。
- 会话32.711s、修复后评价2.933s，前置契约.522s，全回归11.936s分别报告；首次评价失败没有独立过程计时，不编造耗时。worker峰值RSS135.55MiB、产物约121MiB，无GPU计算；不是完整匹配/模型端到端速度测试。
- 1028个既有项目文件和5435个既有证据文件哈希不变；已有MSAC/MAGSAC代码、配置、缓存及旧负结果均未覆盖。不修改main、不提交或推送，不进入Stage7B。

## 复现与当前贡献

[入口](../scripts/run_stage7a_msac_sampling.py)、[E排序/掩码适配与计分](../src/georecon/matching/msac_sampling.py)、[契约测试](../tests/test_stage7a_msac_sampling.py)、[评价修复回归](../tests/test_stage7a_msac_cache_lock.py)。需冻结本地环境、原匹配缓存和上轮十seed产物；只clone仓库不足以复现。新目录拒绝覆盖，原始所有掩码/候选、读取记录、源码副本和软件失败保存在Git忽略目录 `outputs/stage7a_msac_sampling_v1/`；前置契约在 `outputs/stage7a_msac_sampling_contracts_v1/`。未来从修复后的源码新跑需重生成对应契约报告，旧报告不能冒充新源码验证。

```bash
PYTHONPATH=src python scripts/run_stage7a_msac_sampling.py contracts \
  --config configs/experiments/stage7a_msac_sampling_v1.json --output outputs/sampling_contracts_new
PYTHONPATH=src python scripts/run_stage7a_msac_sampling.py run \
  --source-run outputs/stage7a_backend_stability_v1 --cache outputs/stage7a_v1_after_guard_fix \
  --config configs/experiments/stage7a_msac_sampling_v1.json \
  --contract-report outputs/sampling_contracts_new/contracts.json --output outputs/sampling_new
PYTHONPATH=src python scripts/run_stage7a_msac_sampling.py evaluate \
  --source-run outputs/stage7a_backend_stability_v1 --cache outputs/stage7a_v1_after_guard_fix \
  --output outputs/sampling_new
```

| 审计证据 | SHA256 |
|---|---|
| provenance.json | `886f39e3011cf89bce0cb7e22821568428638d5326cab4ac1f7aa980909862ec` |
| session.json | `0a33da5185b1c39580e3d8b13e52774aad841603b14ff57e5e458ee5a6b36d8b` |
| evaluation.json | `bd521817e3d99cc80fc10aa1098dd639023938c6ce59faad52f535a916898434` |
| pairs.csv | `70109f5506a3b490031968c84d6e894f7d024c8066ec8bc900c047f857d38435` |
| independent_checks.json | `02465eb89b4d7dbe7edf141c08095eefabfb835a3e90ca4e59727d10f714265b` |
| evaluate.stderr.txt | `b0e615b7171dadebe231d5158d83af342a71072e16b8a470d35accd4a8969c95` |

本轮贡献是形成C＋MSAC的明确候选优先级，并用不改下游几何的2×2对照排除一次无效改进，保留所有失败及独立评价修复证据。没有新增准确率/鲁棒性收益；既有MSAC收益继续保留，尚未证明未知物体泛化、稠密质量或原创算法优势。下一项尝试需在观察其结果前独立固定，不能从本批结果临时挑顺序、种子或阈值。
