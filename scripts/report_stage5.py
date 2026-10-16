"""Publish the simple policy, its no-gain result, and explicitly simulated fallback cases."""
from pathlib import Path
from georecon.data.protocol import read_json
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5'
e=read_json(DOC/'evaluation.json');fit=read_json(DOC/'fit_report.json');faults=read_json(DOC/'fault_probes.json');probe=read_json(DOC/'gt_isolation_probe.json')
def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'
lines=['# 阶段5子步骤：最简整场景选择与回退','',
'已完成阶段4复核、仅开发集确定规则、提交锁定后对固定8实例的整场景选择、独立评价与回退探针。运行源码及冻结规则提交为`131962e`。本次不做点过滤、加权融合、相机/点联合优化，也不将不同方法的相机补成一个结果。','',
'## 阶段4复核与设计依据','',
'24条诊断、已发布副本、观测/残差数组摘要、预测指纹和重新计算的GT角度/ATE一致。保留10条证据完整、8条部分可用、6条未知。模糊COLMAP有低Sampson却严重错误位姿；因此没有按跨方法最小残差、未校准confidence或任意加权总分选优。spaced3只有一条审计边有证据，不能外推全场景；无模型/OOM保持不可用。原生数据库ID→原图名称勘误仍保持正确。详见`stage4_audit.json`。','',
'## 固定规则与开发集证据','',
'规则是 **VGGT raw → COLMAP → VGGT官方BA**。依次选择首个身份/摘要一致、完整成功相机输出、至少一条几何证据且产物完整的候选。诊断全未知、失效/过期、部分注册、OOM、缺失或损坏产物均不能入选。partial_evidence可选，但输出警示，不赋“可靠”标签。全部不合格则abstain，no_reconstruction且不造相机。','',
'开发集只比较三方法6种固定排列，唯一目标为4个开发实例平均joint AUC@10最大，失败按0计，定义继承阶段3。无连续阈值、残差排序、概率校准或条件标签分支。拟合进程禁止打开scan6/test和混合evaluation/results文件；仅dev真值用于确定排列，文件访问清单保留在`fit_report.json`。','',
'| 优先级 | dev平均joint AUC@10 |','|---|---:|']
for t in fit['trials']:lines.append(f"| {' → '.join(t['policy']['priority'])} | {fmt(t['mean_dev_joint_auc10'],6)} |")
lines+=['',
'两个raw首选排列打平，按预声明规范顺序COLMAP、BA、raw的字典序决定后续排列。所有真实dev实例均选raw，因此**不能从这些自然样本学习或验证最佳回退次序**。规范平局规则不冒充已学到的资源优化或可靠性保证。`configs/selection/stage5_v1.json`在测试选择前已提交，测试后没有重拟合。`--verify-frozen`仅从dev重算并确认同一结果，不改原policy或fit_report；验证记录见fit_reproduction_check.json。','',
'该DTU划分仍为scan1/dev、scan6/test，每划分只有一个独立对象和4种相关条件；阶段3/4已看过scan6结果，本轮不是新增盲测。不用此小样本作泛化或显著性主张。','',
'## 固定8实例结果','',
'| 划分/实例 | 所选整场景方法 | 自然回退 | 警示 | joint AUC@10 | 旋转均值° | 方向均值° | ATE mm |','|---|---|---|---|---:|---:|---:|---:|']
for r in e['rows']:
 lines.append(f"| {r['split']}/{r['scene_id'].split('_',2)[2]} | {r['selected_method']} | {r['fallback_used']} | {'部分证据' if r['warning'] else '无'} | {fmt(r['joint_auc']['10'])} | {fmt(r['rotation_deg'])} | {fmt(r['translation_deg'])} | {fmt(r['ate_mm'])} |")
lines+=['','| 划分 | 方法 | 完整成功/4 | joint AUC@5/10/20 | 旋转均值° | 方向均值° | ATE均值mm |','|---|---|---:|---|---:|---:|---:|']
for g in e['groups']:
 lines.append(f"| {g['split']} | {g['method']} | {g['successes']}/4 | {'/'.join(fmt(g['mean_joint_auc'][str(t)]) for t in [5,10,20])} | {fmt(g['rotation_available_subset']['mean'])}（n={g['rotation_available_subset']['count']}） | {fmt(g['translation_available_subset']['mean'])} | {fmt(g['ate_available_subset']['mean'])} |")
lines+=['',
'**8例全部选raw，真实回退0次；精度与always-raw完全相同，没有精度收益。** 四例带部分证据警示。注册/完整输出不代表真实正确率。相机列表、世界单位及坐标系与所选源结果一致，点文件字节SHA相同；产物使用硬链接且流程仅读取，按不可变文件使用，后续编辑应另存新文件而非就地写入；相对位姿与评价专用Sim(3) ATE逐项与原raw评价一致，没有在选择中做GT对齐。所有baseline原失败继续保留于比较表，不算成选择器修复成功。','',
'## 开发集故障注入：可执行不等于准确','',
'以下10个探针仅改变内存副本中的候选是否可见或证据状态，未改原快照；GT只用于探针后的dev评价。不是新增真实失败或自然回退，不加入上述8例分母，也不用于重新拟合规则。','',
'| dev实例 | 模拟条件 | 结果 | joint AUC@10 |','|---|---|---|---:|']
for p in faults['probes']:lines.append(f"| {p['scene_id'].split('_',2)[2]} | {p['scenario']} | {p['decision']['selected_method'] or 'abstain'} | {fmt(p['gt_joint_auc10'])} |")
lines+=['',
'raw不可用时，blur5回退到COLMAP但joint AUC@10=0，其旋转约14°、方向约46°；这条明确反例说明当前可用性门控不保证回退准确。reduced3屏蔽raw+COLMAP后能用原BA整场景；normal5/blur5的BA原已OOM，因而在前两方法被屏蔽后弃权。全缺失或全部证据未知也弃权，而非拼相机、复制初始化深度或设置高可靠分数。','',
'开发集GT隔离导出探针在输入副本中加入无效reference/evaluation/calibration，选择决定、全部相机字段和产物摘要不变；见`gt_isolation_probe.json`。初次探针脚本将哨兵写入放在dev拟合拒读钩子内，被禁止打开evaluation.json，在执行选择器前终止。已保留这次脚本错误及不完整输入；改由独立进程准备哨兵后通过。策略、规则及8例真实输出未改动，也未重跑选最优。','',
'## 成本：没有隐藏额外分支','',
'本版本是**eager快照选择**，不是先运行raw、必要时才运行后续模型的懒执行调度器。实际新增计算只有CPU读元数据、摘要核查与原文件重定位，没有新增GPU分配或模型推理。GPU上限继续12GB；原始GPU分支使用阶段3的预算与历史记录，不能说本轮重测了显存。','',
'以下成本为三个原baseline完整CLI墙钟（包含失败/OOM）＋共享阶段4完整诊断进程＋本轮选择完整进程。它是按历史实测段相加的回放核算，**不是重新测得的在线端到端模型墙钟**。准备、预审计和外层输入锁校验未统一计时，不计作部署总成本。单独报选择耗时不能当作混合管线速度提升。','',
'| 实例 | 新选择进程秒 | eager核算总秒 | 原raw完整CLI秒 |','|---|---:|---:|---:|']
for r in e['rows']:
 b=next(b for b in e['baseline_rows'] if b['scene_id']==r['scene_id'] and b['method']=='vggt_raw')
 lines.append(f"| {r['scene_id']} | {fmt(r['new_process_seconds'])} | {fmt(r['accounted_eager_seconds'])} | {fmt(b['historical_cli_seconds'])} |")
new=sum(r['new_process_seconds'] for r in e['rows']);total=sum(r['accounted_eager_seconds'] for r in e['rows']);raw=sum(b['historical_cli_seconds'] for b in e['baseline_rows'] if b['method']=='vggt_raw');rss=max(r['memory_bytes']['peak_rss_os'] for r in e['runs'])/1024**3
lines+=['',f'八次新增选择进程合计 **{new:.3f}s**，OS RSS峰值{rss:.3f}GiB；eager核算合计 **{total:.3f}s**，原raw合计{raw:.3f}s。当前实现既未提高这8例的精度，也不支持节省端到端成本的主张。独立GT探针另耗{probe["wall_seconds"]:.3f}s，不并入真实8例选择耗时。', '',
'## 检查与复现','',
'原69项回归与新增10项选择检查合计79项不同测试已交叉覆盖。完整COLMAP环境79项有2项OpenCV检查跳过；完整reliability环境79项有1项原生PyCOLMAP检查跳过，均在另一环境执行通过。选择单元检查包括优先级而非最小残差、两级回退、全未知弃权、部分证据警示、部分相机拒绝、过期诊断、GT角色拒绝、损坏产物回退、源相机/点文件保留和拟合禁止读test。独立实际8例验证了决策重算、所有摘要、输出字节、指标一致与成本加和。','',
'复用既有reliability CPU环境，无新依赖或模型。首次准备/拟合及执行入口如下，已有锁/策略/输出禁止覆盖；若做新版本必须显式建立新输出及新验收来源，不把重跑当成本轮原始证据。','',
'```bash',
'python scripts/stage5_prepare.py audit',
'# 验证已冻结的开发集拟合，不改policy：',
'python scripts/fit_stage5.py --verify-frozen',
'# 首次新版本才准备bundle并运行不带--verify-frozen的拟合',
'# 上面完成后应提交固定policy，再运行：',
'python scripts/run_stage5.py',
'python scripts/probe_stage5.py',
'python scripts/evaluate_stage5.py',
'python scripts/report_stage5.py',
'# 单实例入口：',
'georecon select-scene --bundle data/stage5_bundles/dtu_scan1_normal5 --policy configs/selection/stage5_v1.json --output outputs/new_scene_selection',
'```','',
'本机解释器为`/home/zchming/.venvs/georecon-reliability/bin/python`；原生COLMAP回归使用其独立环境。`fit_report.json`保存全部6候选与dev文件访问清单；`decision.json`保存每候选拒绝原因；`cost_accounting.json`保存每条历史分支开销；GT评价仅在独立脚本中读取。Git中只发布代码、策略和JSON/日志，小型证据的二进制相对路径以原本地输出目录为根；数据、权重和点云NPZ不进入Git。','',
'本次完成最简场景级选择/回退子步骤后停止。部分证据、raw首选失效时的回退准确性和小样本泛化仍是局限；没有开始点过滤、加权融合或联合优化。']
(DOC/'report.md').write_text('\n'.join(lines)+'\n')
print('Stage5 report generated')
