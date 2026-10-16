"""Publish joint refinement tradeoffs, costs and cumulative evidence-backed contributions."""
import sys,platform,subprocess
from pathlib import Path
import numpy as np
import scipy
from georecon.data.protocol import read_json,write_json,sha256
from georecon.geometry.alignment import Similarity
from stage5_points import Evaluator
from stage5_joint import ROOT,DOC,OUT,PROTOCOL,RULE,CONDITIONS


def report():
    comparison=read_json(DOC/'comparison.json');rows=comparison['rows'];failures=[]
    for scan in [1,6]:
        ev=Evaluator(scan)
        for cond in CONDITIONS:
            scene=f'dtu_scan{scan}_{cond}';ev.prepare(scene);base=OUT/'frozen'/scene
            row=next(r for r in rows if r['scene_id']==scene)
            with np.load(ROOT/'outputs/stage5_fusion_v1/frozen'/scene/'anchors.npz') as z:ids=z['source_point_indices'];pids=z['point_id']
            with np.load(ROOT/'outputs/stage5_fusion_v1/frozen'/scene/'fused.npz') as z:before=z['xyz'][ids]
            with np.load(base/'optimized.npz') as z:after=z['xyz'][ids]
            a=ev.alignment;sim=Similarity(a['scale'],np.array(a['R']),np.array(a['t']))
            db=ev.tree.query(sim.apply(before),workers=8)[0] if len(ids) else np.array([])
            da=ev.tree.query(sim.apply(after),workers=8)[0] if len(ids) else np.array([])
            order=np.argsort(da-db)[::-1][:10]
            failures.append({'scene_id':scene,'accepted':row['run']['rows'][0]['solver']['accepted'],'solver_status':row['run']['rows'][0]['solver']['status'],
                'scope':'all eligible points against fixed sampled STL, mm, no ROI; posthoc only','eligible_points':len(ids),'improved':int((da<db-1e-9).sum()),'worsened':int((da>db+1e-9).sum()),
                'good_to_bad_2mm':int(((db<=2)&(da>2)).sum()),'bad_to_good_2mm':int(((db>2)&(da<=2)).sum()),'mean_before_mm':float(db.mean()) if len(ids) else None,'mean_after_mm':float(da.mean()) if len(ids) else None,
                'worst_cases':[{'source_point_index':int(ids[i]),'colmap_point_id':int(pids[i]),'before_mm':float(db[i]),'after_mm':float(da[i]),'change_mm':float(da[i]-db[i])} for i in order]})
    write_json(DOC/'failure_analysis.json',{'rows':failures,'GT_not_used_by_solver_or_acceptance':True,'weights_or_bounds_not_retuned':True,'natural_solver_failures':sum(not r['run']['rows'][0]['solver']['accepted'] and r['run']['problem_sha256'] is not None for r in rows)})
    methods=['raw','before_fused','grouped_initial','optimized'];summary={}
    for split in ['dev','test']:
        subset=[r for r in rows if r['split']==split]
        summary[split]={method:{str(t):{metric:float(np.mean([r[method]['thresholds_mm'][str(t)][metric] for r in subset])) for metric in ['precision','recall','f1']} for t in [1,2,5]} for method in methods}
    summary.update(accepted_cases=sum(r['run']['rows'][0]['solver']['accepted'] for r in rows),fallback_cases=sum(not r['run']['rows'][0]['solver']['accepted'] for r in rows),
        changed_points=sum(r['run']['rows'][0]['moved_points_vs_fusion'] for r in rows),default_enabled=False,interpretation='objective and fitted residual improve, but point geometry declines vs fusion in dev/test; pose tradeoffs depend on fixed frame vs refitted evaluation alignment')
    write_json(DOC/'summary.json',summary)
    costs=[]
    for r in rows:
        previous=read_json(ROOT/'docs/stage5_fusion/cost_accounting.json')
        previous_case=next(p for p in previous['rows'] if p['scene_id']==r['scene_id'])
        costs.append({'scene_id':r['scene_id'],'previous_historical_eager_plus_fusion_seconds':previous_case['sum_historical_and_incremental_seconds'],
            'new_joint_process_body_seconds':r['run']['wall_seconds'],'new_solver_seconds':r['run']['rows'][0]['solver'].get('wall_seconds',0),
            'accounted_historical_sum_seconds':previous_case['sum_historical_and_incremental_seconds']+r['run']['wall_seconds']})
    joint_sum=sum(r['new_joint_process_body_seconds'] for r in costs);solver_sum=sum(r['new_solver_seconds'] for r in costs)
    peak=max(r['run']['peak_rss_gib'] for r in rows)
    write_json(DOC/'cost_accounting.json',{'rows':costs,'new_joint_seconds_sum':joint_sum,'solver_only_seconds_sum':solver_sum,'peak_joint_process_rss_gib':peak,
        'offline_evaluation_wall_seconds':comparison['offline_evaluation_wall_seconds'],'peak_evaluator_rss_gib':comparison['peak_evaluator_rss_gib'],'new_GPU_allocated_bytes':0,
        'fresh_online_end_to_end_wall_time':None,'scope':'historical eager baseline including BA failures/diagnostics/selection, fusion and new joint body; not fresh online latency; new joint timing includes grouped-initial and attempt/accepted artifact writes; excludes process startup'})
    source_paths=[ROOT/'src/georecon/optimization/joint.py',ROOT/'scripts/stage5_joint.py',ROOT/'scripts/check_stage5_joint.py',ROOT/'scripts/report_stage5_joint.py',ROOT/'scripts/stage5_fusion.py',ROOT/'scripts/stage5_points.py']
    write_json(DOC/'input_lock.json',{'protocol_sha256':sha256(PROTOCOL),'rule_sha256':sha256(RULE),'protocol_commit':'0489431','configuration_frozen_before_test_commit':'c3510ce',
        'previous_fusion_input_lock_sha256':sha256(ROOT/'docs/stage5_fusion/input_lock.json'),'stage3_freeze_sha256':sha256(ROOT/'docs/stage3/freeze.json'),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in source_paths},
        'cases':[{'scene_id':r['scene_id'],'input_hashes':r['run']['input_hashes'],'optimized_sha256':r['run']['rows'][0]['artifact_sha256'],'problem_sha256':r['run']['problem_sha256']} for r in rows]})
    write_json(DOC/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'device':'CPU sparse TRF/LSMR','threads':8,'new_packages_or_weights':[],
        'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv'],capture_output=True,text=True).stdout,'new_GPU_allocated_bytes':0,'budget_gib':12})
    lines=['# 阶段 5：最小联合位姿与稀疏结构优化','',
        '已复核加权融合结果和局部退化，实现并验证最小联合优化。配置仅由scan1开发集确定，之后锁定并运行同一scan6测试条件。联合优化降低拟合残差，但相对融合结果的固定采样F1略退化；保持显式实验工具，不默认启用。此子步骤结束后停止，不扩展数据或方法。','',
        '## 前置证据与范围','',
        'fusion_audit.json重新计算全部8例的raw/加权/无权几何指标和局部点变差计数，均与上一步发布结果相同。加权仅影响502/6497792点，6/8例回退；开发normal5中84/174个修正点距离参考样本更远，67个从2mm内变到2mm外。开发集权重优势未迁移到测试，无权平均在测试normal5更好。','',
        '保持原来的scan1开发/scan6测试、light3、normal5/reduced3/spaced3/blur5、视图数、图像与分辨率。仅使用上一子步骤已有的合格轨迹，不放宽融合门限、不新提取匹配、不传播到其余稠密点，也不重新加载模型。前序测试结果已被查看，这是固定小样本的回顾性实验，不是新的盲测。','',
        '## 联合变量、目标与预算','',
        '变量包括第2至第N台相机的左乘旋转增量及归一化平移增量，以及每条合格COLMAP轨迹的一个共享三维点。内参固定；第一相机的序列化位姿完全固定，原raw/对齐COLMAP的三维点先验约束尺度与世界坐标规范。只有既有合格网格位置会写入共享点，其他float32点字节保持不变。该步骤是带先验的稀疏BA，不是新的基础模型或新求解算法。','',
        '同轨迹的融合点均值用作共享结构初值，并保存grouped_initial对照。当前两个有效场景的合格point ID恰好互不重复，故该对照与融合前完全相同；不会把初始化变化冒充优化收益。每个共享点使用其全部可用的真实COLMAP二维轨迹，而非自身生成的像素。native坐标继续按COLMAP xy减0.5、common逆变换、VGGT原图到模型变换转换。','',
        '归一化尺度为合格原raw点的源相机正Z中位数，不使用真值。残差包括：','',
        '- 实测轨迹的原生像素重投影误差/2，乘sqrt(w_col)；',
        '- 每个合格网格raw XYZ先验，sigma=归一化尺度的1%，乘sqrt(w_raw)；',
        '- 每个共享点一次对齐COLMAP XYZ先验，sigma=尺度的1%，乘sqrt(0.5*w_col)，不按轨迹观测数重复添加；',
        '- 非固定相机的旋转/平移先验，sigma分别0.5°、尺度的0.5%，乘sqrt(lambda)；',
        '- 非正深度软屏障，优化后另严格检查观测正深度。','',
        '所有归一化标量残差采用soft_l1，权重沿用冻结融合规则，不是校准概率。使用已安装SciPy的[有界稀疏least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)，TRF/LSMR与显式稀疏Jacobian结构；非线性参数同时求解，属于联合优化，不是轮流独立平均。','',
        '每个候选max_nfev=60（另记录包括有限差分在内的实际残差调用数）、求解墙钟上限20s；旋转每分量≤1°、相机平移每分量≤尺度1%、点增量每分量≤尺度2%；最多2000个共享点、12000个观测。只比较lambda∈{1,10}，其余配置预先固定，按开发集宏平均F1@2mm选择lambda=1。协议提交0489431，配置提交**c3510ce**，然后才运行测试联合优化。','',
        '接受条件仅含求解器成功状态、有限值、目标不增和正深度，**不含GT**。目标下降不是正确性认证。达到预算或失败时保留attempt参数/点/相机，最终输出逐字节回退到融合前；没有合格对应也直接回退。','',
        '## 点几何与位姿结果','',
        '点几何继续使用同一最多200000个原始预测索引和200000个固定STL样本，官方ObsMask/Plane、1/2/5mm P/R/F1与无距离剔除均值。每个版本使用**同一原raw相机→GT Sim3**，不因优化相机变化重拟合点云GT变换；同时保留固定raw ROI敏感性。该预算评测不同于官方完整DTU排名。','',
        '|场景|原raw F1@2mm|融合前|共享初值|联合优化后|接受优化|','|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append('|'+r['scene_id']+'|'+'|'.join(f"{100*r[m]['thresholds_mm']['2']['f1']:.5f}%" for m in methods)+'|'+('是' if r['run']['rows'][0]['solver']['accepted'] else '回退')+'|')
    lines+=['']
    for split in ['dev','test']:
        lines.append(f"{split} 宏平均F1@2mm："+'；'.join(f"{m} {100*summary[split][m]['2']['f1']:.5f}%" for m in methods)+'。');lines.append('')
    lines+=['|有效场景|拟合重投影RMS px 前→后|旋转误差° 前→后|平移方向误差° 前→后|独立评价Sim3 ATE mm 前→后|固定raw评价框架相机RMSE mm 前→后|','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        s=r['run']['rows'][0]['solver']
        if not s['accepted']:continue
        a,b=r['poses_before'],r['poses_after'];fixed=r['fixed_frame_camera_rmse_mm']
        lines.append(f"|{r['scene_id']}|{s['initial_reprojection_rms_native_px']:.4f}→{s['final_reprojection_rms_native_px']:.4f}|{a['rotation_deg_registered_subset']['mean']:.4f}→{b['rotation_deg_registered_subset']['mean']:.4f}|{a['translation_direction_deg_registered_subset']['mean']:.4f}→{b['translation_direction_deg_registered_subset']['mean']:.4f}|{a['camera_center_ate']['rmse']:.4f}→{b['camera_center_ate']['rmse']:.4f}|{fixed['before']:.4f}→{fixed['after']:.4f}|")
    lines+=['',
        '普通ATE会为每组预测相机分别拟合评价Sim3，固定框架指标则使用原raw的变换；必须一起看，不能用重新对齐后的改善掩盖原始世界框架的漂移。重投影误差来自优化自身拟合轨迹，不是独立验证信号。','',
        '## 失败分析与结论边界','',
        '主指标使用固定200000点采样，只覆盖极少量修正点。对所有受影响点另作未套ROI的后验检查：dev normal5平均参考距离2.2251→1.5635mm，同时70/174点变差、34点从2mm内变为2mm外。局部均距改善与采样F1下降可以同时出现，不能把这一微小主指标变化扩展成整体几何必然变差的结论；完整各点统计见failure_analysis.json。','',
        '2例normal5求解成功，6例因既有融合无对应保持原输出；没有自然OOM或求解器预算失败。dev/test normal5的固定采样点F1均低于融合前；开发集位姿误差也退化。测试相对位姿和重新对齐ATE改善，但固定raw框架相机RMSE变大。不能宣称联合优化统一提升重建质量。','',
        'failure_analysis.json记录所有合格点优化前后到固定参考样本距离的改善/变差、2mm正确点变错误点及最差索引；为锁定后的GT诊断，不参与接受规则或改配置。仅两个物体、每个四种相关条件，且只有一个条件产生有效优化，泛化和显著性均未建立。','',
        '本次没有异常实现失败；保留了实际几何退化，另以合成max_nfev=1探针验证不收敛时拒绝更新且attempt可恢复。前序融合的中心对齐失败与浮点精度修正仍在原目录，未覆盖或删除。','',
        '## 成本、验证和产物','',
        f"8次冻结运行主体合计{joint_sum:.4f}s，其中真正求解合计{solver_sum:.4f}s，峰值进程RSS {peak:.4f}GiB。CPU线程8；这里是小型稀疏求解，未调用Torch/CUDA或新模型，新增GPU分配0，满足12GB约束。nvidia-smi整卡占用另存environment.json，不当成本步骤分配。计时包含加载校验、初值/attempt/最终压缩写出，不含进程启动；不是新在线端到端延迟。cost_accounting.json同时保留上游所有历史分支、诊断、OOM、融合和本次增量，不作速度收益主张。",
        '',
        'before：outputs/stage5_fusion_v1/frozen/<scene>/fused.npz与prediction.json；after：outputs/stage5_joint_v1/frozen/<scene>/optimized.npz与prediction.json。problem.npz保存实际稀疏问题，optimized_attempt.npz保存求解参数、共享点和候选相机，run.json保存目标、调用数、迭代轨迹与接受状态。depth/confidence仍是未优化原始诊断，不是与优化结果一致的新深度图。大产物留本地，Git发布配置、源码、轻量逐场景JSON和报告。','',
        '104项不同测试分环境交叉覆盖：可靠性环境104项跳过1个原生COLMAP测试；COLMAP环境104项跳过2个OpenCV、1个SciPy评价和8个联合优化测试，这些在可靠性环境实际通过。新增8项覆盖同时更新位姿/点、稀疏依赖正确性、固定首相机、边界、预算失败、约束上限、保留未改float32字节和目标定义。validation.json进一步复算8例公式/目标、参数边界、相机、GT隔离访问记录与原始索引/字节。','',
        '## 截至当前的贡献','',
        '1. 建立COLMAP、VGGT raw、VGGT官方BA及后处理之间统一的相机/像素/尺度/结果协议与可审计的分阶段实验管线，保留OOM、失败、预算和真实成本。',
        '2. 实现本地推理不读取GT的几何诊断与unknown状态处理，以及开发集拟合、规则锁定、测试配置隔离和回退验证。',
        '3. 完成场景选择、点过滤、稀疏对应加权融合、带先验的最小联合优化及必要对照，产物与输入可追溯。',
        '4. 提供受限样本中的失败证据：低内部残差不保证正确，unknown不等于错误点，权重优势可能不迁移，联合拟合残差下降仍可能伴随局部几何退化、固定采样主指标下降或坐标框架漂移。','',
        '这些是已实现的工程能力与当前证据支持的经验发现；COLMAP、VGGT、官方BA、Sim3和SciPy求解器不是本项目原创。目前尚未证明稳定超过VGGT raw，也未完成足以支撑算法新颖性或SOTA主张的证据。[累计贡献清单](../contributions.md)区分了这两类贡献与未证实主张。','',
        '## 复现','',
        '```bash','export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8','PY=/home/zchming/.venvs/georecon-reliability/bin/python',
        '# 依赖已有本地上游产物，所有生成命令拒绝覆盖。',
        '$PY scripts/stage5_joint.py generate --scene dtu_scan6_normal5 --output-root outputs/stage5_joint_reproduction',
        '$PY scripts/stage5_joint.py fit --verify-frozen',
        '$PY scripts/stage5_joint.py evaluate',
        '$PY scripts/check_stage5_joint.py',
        '$PY -m unittest discover -s tests -v','```','',
        '本授权子步骤完成后停止，不自动进入新阶段、扩展场景或更复杂优化。','']
    (DOC/'report.md').write_text('\n'.join(lines))
    contribution='''# 累计贡献与证据边界

截至阶段5最小联合优化，系统已经完成“基线—诊断—选择—过滤—融合—联合优化”的最小实验链。当前贡献以工程协议和可审计的经验结果为主，不能直接等同于新算法或性能突破。

|类别|已经完成的内容|证据|边界|
|---|---|---|---|
|统一协议|处理像素中心、预处理变换、内外参、world-to-camera、任意尺度、统一结果与失败状态|[阶段2](stage2_vggt/report.md)、[协议](protocol.md)|集成现有方法，不是原创骨干模型|
|实验工程|固定DTU开发/测试划分与8个条件、输入/规则/源码哈希、资源预算、实际OOM和成本归档|[阶段3](stage3/report.md)|两场景、回顾性试验；点指标是采样预算下的自定义评价|
|推理诊断|匹配图、独立观测残差、正深度/角度、跨视图深度一致性、unknown状态与GT读取拒绝|[阶段4](stage4/report.md)|诊断量不是校准正确性概率，观测也未证明统计独立|
|策略与回退|开发集确定场景优先级、不可用回退、全失败弃权和故障探针|[场景选择](stage5/report.md)|自然样本都选择raw，未带来精度或速度提升|
|融合/优化实验|最小点过滤、显式对应加权融合、带原始先验的相机和稀疏结构联合优化|[过滤](stage5_points/report.md)、[融合](stage5_fusion/report.md)、[联合优化](stage5_joint/report.md)|已有算法组件上的最小实现；影响范围小，不能宣称整体原创或稳定收益|
|负结果与反例|内部一致但GT位姿错误；unknown中正确点被误删；权重优势不迁移；优化拟合更好但仍有点退化、固定采样指标下降|上述逐场景比较、failure_analysis与失败目录|是当前条件下的经验发现，泛化与算法新颖性均未建立|

当前可支持的表述：**我们建立了一个可审计、运行时不读取真值的多视图重建比较与可靠性策略实验系统，并在固定小样本中验证了多种简单策略的失败边界。**

当前不能支持的表述：已经提出普遍优于VGGT的融合算法、得到校准可靠性概率、达到SOTA、充分证明泛化，或将官方VGGT/COLMAP/BA与现成优化器记为原创。

后续每次项目结果汇报应附简短累计贡献总结，并更新可支持与尚未证实的结论。扩展实验与新阶段仍须用户明确授权。
'''
    (ROOT/'docs/contributions.md').write_text(contribution)

if __name__=='__main__':report()
