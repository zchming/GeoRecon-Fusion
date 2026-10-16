"""Report weighted fusion results and posthoc failures; never change locked rules."""
import sys,platform,subprocess
from pathlib import Path
import numpy as np
import scipy
from georecon.data.protocol import read_json,write_json,sha256
from georecon.geometry.alignment import Similarity
from stage5_points import Evaluator
from stage5_fusion import ROOT,DOC,OUT,PROTOCOL,RULE,CONDITIONS


def report():
    comparison=read_json(DOC/'comparison.json');rows=comparison['rows'];analysis=[]
    for scan in [1,6]:
        ev=Evaluator(scan)
        for condition in CONDITIONS:
            scene=f'dtu_scan{scan}_{condition}';ev.prepare(scene);dest=OUT/'frozen'/scene
            row=next(r for r in rows if r['scene_id']==scene);a=ev.alignment
            sim=Similarity(a['scale'],np.array(a['R']),np.array(a['t']))
            with np.load(dest/'anchors.npz') as z:ids=z['source_point_indices'];point_ids=z['point_id']
            with np.load(ROOT/'outputs/stage5_v1'/scene/'selected_artifacts/raw_predictions.npz') as z:raw=z['world_points'].reshape(-1,3)[ids]
            with np.load(dest/'fused.npz') as z:fused=z['xyz'][ids]
            before=ev.tree.query(sim.apply(raw),workers=8)[0] if len(ids) else np.array([])
            after=ev.tree.query(sim.apply(fused),workers=8)[0] if len(ids) else np.array([])
            order=np.argsort(after-before)[::-1][:10]
            item={'scene_id':scene,'scope':'posthoc all affected points to fixed sampled STL, mm; not masked ROI and not a new fitting signal',
                'fallback':row['run']['fallback'],'reason':row['run']['fallback_reason'],'affected_points':len(ids),
                'distance_improved':int((after<before-1e-9).sum()),'distance_worsened':int((after>before+1e-9).sum()),
                'good_to_bad_2mm':int(((before<=2)&(after>2)).sum()),'bad_to_good_2mm':int(((before>2)&(after<=2)).sum()),
                'mean_before_mm':float(before.mean()) if len(ids) else None,'mean_after_mm':float(after.mean()) if len(ids) else None,
                'worst_displacement_examples':[{'source_point_index':int(ids[i]),'colmap_point_id':int(point_ids[i]),'distance_before_mm':float(before[i]),'distance_after_mm':float(after[i]),'distance_change_mm':float(after[i]-before[i])} for i in order]}
            analysis.append(item)
    write_json(DOC/'failure_analysis.json',{'rows':analysis,'gt_only_after_rule_freeze':True,'weights_not_changed':True})
    summary={};methods=['raw','prior_filtered','unweighted','weighted_fused']
    for split in ['dev','test']:
        subset=[r for r in rows if r['split']==split]
        summary[split]={m:{str(t):{metric:float(np.mean([r[m]['thresholds_mm'][str(t)][metric] for r in subset])) for metric in ['precision','recall','f1']} for t in [1,2,5]} for m in methods}
    incremental=sum(r['run']['wall_seconds'] for r in rows)
    exports=sum(r['run']['track_export']['wall_seconds'] for r in rows)
    costs=[]
    for r in rows:
        cost=read_json(ROOT/'outputs/stage5_v1'/r['scene_id']/'cost_accounting.json')
        costs.append({'scene_id':r['scene_id'],'previous_eager_accounted_seconds':cost['accounted_eager_total_seconds'],'new_native_track_export_seconds':r['run']['track_export']['wall_seconds'],
            'new_fusion_seconds':r['run']['wall_seconds'],'sum_historical_and_incremental_seconds':cost['accounted_eager_total_seconds']+r['run']['wall_seconds']+r['run']['track_export']['wall_seconds'],
            'minimal_raw_colmap_reuse_branch_historical_seconds':sum(cost['baseline_branches'][m]['invocation']['full_cli_wall_seconds'] for m in ['colmap','vggt_raw']),
            'scope':'historical eager selection/diagnostics/all branches (including BA OOM) plus new export/fusion; not new online wall time; prior optional filtering is not executed by fusion'})
    write_json(DOC/'cost_accounting.json',{'rows':costs,'historical_eager_total_seconds':sum(r['previous_eager_accounted_seconds'] for r in costs),'new_track_export_seconds_sum':exports,'new_frozen_fusion_seconds_sum':incremental,
        'offline_evaluation_wall_seconds':comparison['offline_evaluation_wall_seconds'],'peak_fusion_rss_gib':max(r['run']['peak_rss_gib'] for r in rows),'peak_export_rss_gib':max(r['run']['track_export']['peak_rss_gib'] for r in rows),
        'peak_evaluator_rss_gib':comparison['peak_evaluator_rss_gib'],'additional_gpu_allocated_bytes':0,'fresh_end_to_end_wall_time':None})
    summary.update(affected_points=sum(r['run']['rows'][0]['moved_points'] for r in rows),total_points=sum(r['run']['rows'][0]['point_count'] for r in rows),fallback_cases=sum(r['run']['fallback'] for r in rows),
        default_enabled=False,new_frozen_fusion_seconds=incremental,new_track_export_seconds=exports,new_gpu_allocated_bytes=0,
        interpretation='one normal5 scene per split has eligible sparse anchors; tiny macro gains do not establish robust hybrid improvement; test unweighted exceeds weighted; no default enable')
    write_json(DOC/'summary.json',summary)
    lock={'protocol_sha256':sha256(PROTOCOL),'rule_sha256':sha256(RULE),'rule_frozen_before_test_commit':'89a0884','protocol_commit':'f389361',
        'stage3_freeze_sha256':sha256(ROOT/'docs/stage3/freeze.json'),'prior_filter_input_lock_sha256':sha256(ROOT/'docs/stage5_points/input_lock.json'),
        'source_files_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/georecon/fusion/anchors.py',ROOT/'scripts/stage5_fusion.py',ROOT/'scripts/export_stage5_fusion_tracks.py',ROOT/'scripts/stage5_points.py',ROOT/'scripts/check_stage5_fusion.py',ROOT/'scripts/report_stage5_fusion.py']},
        'cases':[]}
    for r in rows:
        scene=r['scene_id'];base=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/sparse'
        lock['cases'].append({'scene_id':scene,'input_hashes':r['run']['input_hashes'],'native_model_files_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(base.rglob('*.bin'))},
            'anchors_sha256':r['run']['anchors_sha256'],'fused_sha256':r['run']['rows'][0]['sha256'],'unweighted_sha256':r['run']['unweighted_sha256']})
    write_json(DOC/'input_lock.json',lock)
    write_json(DOC/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'pycolmap_track_export':rows[0]['run']['track_export']['pycolmap'],
        'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv'],capture_output=True,text=True).stdout,'new_model_weights_or_packages':[],'device':'CPU','threads':8,'gpu_budget_gib':12,'additional_gpu_allocated_bytes':0})
    lines=['# 阶段 5：最小加权融合验证','',
        '已完成点过滤退化复核、稀疏对应加权融合、开发集锁定与相同 DTU 划分验证。当前只有 normal5 的少数点发生融合，结果不支持默认启用或稳定收益主张。未进行联合优化、相机平均、新稠密 MVS 或重新训练。','',
        '## 前置复核与范围','',
        '重新计算前一子步骤全部8例的 raw/filtered 点指标以及正确点被删除、unknown 正确点数量，与发布结果完全一致，见 filter_audit.json。开发集 F1@2mm 18.312%→17.082%；reduced3 误删2370个采样评价中的正确点，其中2366个属于 unknown。此前失败日志仍在原目录。此次从**未过滤 raw**开始，没有融合证据的位置保留原坐标，不能把相对旧过滤的恢复算作融合收益。','',
        '固定 scan1=dev、scan6=test，light3，normal5/reduced3/spaced3/blur5，视图、图像、预处理和上游方法设置完全沿用阶段3。前序阶段已经查看过测试结果，本次是回顾性小样本验证，不是新的盲测。','',
        '## 对应、坐标与加权规则','',
        '直接复用既有 COLMAP 稀疏三维点及其真实二维轨迹，不新跑 SfM/BA。按 worker 文件名查回图像ID，避免把 native image_id 当作 manifest 顺序。COLMAP 观测 xy 先减0.5转换像素中心，再由 common 800×600 的逆变换回原图，经 VGGT original_to_model 变换到原生392×518网格，取最近整数像素。该离散化会引入至多约0.71原生像素误差；没有最近邻点云匹配、插入稀疏点或空间扩散。','',
        '推理 Sim3 仅来自两套**预测**相机：共同注册相机至少3个且中心非共线；对 R_rawᵀ R_colmap 求SO(3)弦均值确定旋转，再对相机中心以闭式正尺度/平移拟合。中心残差/原始相机中心直径≤0.1，最大朝向差≤5°。这只统一坐标规范，不是与真值对齐，也不联合调整相机或结构。低残差仍不能证明几何正确。','',
        '对应门限固定：轨迹长度≥3、COLMAP平均重投影误差≤2 common像素；锚点在raw相机中正深度，投影到真实观测的误差≤2原生像素；锚点与raw对应点三维距离≤raw源相机Z的5%。多个锚点落入同格时取经典权重最大者，平分取较小point ID，不混合不同对应。','',
        '令 N 为视图数，s 为原raw点在其他预测深度中满足5%相对Z差的支持数，L为轨迹长度、e为COLMAP拟合误差（common像素）：','',
        '`w_raw = 1 + s/(N-1)` ；`w_col = (L/N)/(1+(e/2)^2)`','',
        '`β = α*w_col/(w_raw + α*w_col)` ；`X_fused = X_raw + β*(S(X_col)-X_raw)`','',
        '这些是显式无量纲启发式权重，未经概率校准；COLMAP拟合误差来自自身轨迹，raw深度也是相关网络输出，不是独立正确性证明。融合只修正有明确对应的网格点，没有点删除；没有观察/无合格锚点/缺失或失败的COLMAP/对齐失败都保持raw。depth/confidence仍是原始诊断产物，未与新XYZ同步生成，不可当作融合后的深度或概率。','',
        '候选仅 α∈{0.25,0.5,1}，其余门限在开发评价前固定。按开发集宏平均F1@2mm选择 **α=0.5**，同分按候选顺序；raw与旧过滤作独立基线。另设相同对应集合、β恒0.5的无权平均消融。协议提交 f389361；规则提交 **89a0884**，随后才导出scan6轨迹并运行测试。没有用测试调权重或改变门限。只是一项显式实验工具，未自动接入默认场景选择。','',
        '## 失败尝试与数值修正','',
        '初版仅按预测相机中心确定旋转：4个开发条件都无对应，normal5锚点投影偏差约6–24原生像素。此开发尝试与原协议/源码/产物保留在 outputs/stage5_fusion_v1/failures/center_only_alignment_dev_v1。随后仅在开发阶段改用预测朝向确定旋转，局部门限与权重不变，得到normal5的174个对应点；reduced3仍无对应。','',
        '锁定规则后的首次全量输出使用float64，导致即使完全回退，旧Sim3评价中的NumPy缩放舍入也与float32原始点不同。已保留全套dev/测试输出及 attempt02_comparison.json，修复为输出沿用float32、未改点字节完全相同；**α与所有门限未改**。修复后在禁止访问测试文件的进程重算开发候选，最优规则和原始分数完全复现（fit_reproduction.json）。这一修正发生在首次测试评价之后，属于数值一致性修复，不是新盲测。','',
        '## 几何评价','',
        '沿用上一子步骤的自定义预算：最多200000个固定原始预测索引与200000个等距STL索引，参考取官方Plane上方算召回，预测用官方ObsMask算精度；同一raw相机中心GT Sim3用于所有版本，无ICP。报告1/2/5mm P/R/F1及无距离截断均值。移动点后的ObsMask重新计算，并附固定raw ROI敏感性分析；dev/test normal5仅3/8个样本改变ROI归属，其余为0。固定ROI结果保存在comparison.json。','',
        '这不是官方完整DTU评测：未执行0.2mm邻域降采样、60mm距离上限或20mm离群剔除，只覆盖两个场景的局部视图，不能与官方排名比较。参考数据和协议继续沿用上一子步骤的[DTU官方SampleSet溯源](../stage5_points/report.md)。','',
        '|场景|实际修正点|raw F1@2mm|旧过滤|无权平均|加权融合|','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|'+r['scene_id']+'|'+str(r['run']['rows'][0]['moved_points'])+'|'+'|'.join(f"{100*r[m]['thresholds_mm']['2']['f1']:.5f}%" for m in methods)+'|')
    lines+=['']
    for split in ['dev','test']:
        lines.append(f"{split} 宏平均 F1@2mm："+'；'.join(f"{m} {100*summary[split][m]['2']['f1']:.5f}%" for m in methods)+'。');lines.append('')
    lines+=['## 解释与失败边界','',
        f"只修正 {summary['affected_points']} / {summary['total_points']} 个网格点（{100*summary['affected_points']/summary['total_points']:.5f}%），6/8例完全回退。blur5因预测相机不一致拒绝融合，spaced3无COLMAP模型，reduced3无局部合格对应。规则在开发集仅由normal5提供有效区分；这不是广泛适用性的证据。",
        '',
        '开发集加权优于同对应的无权平均；测试normal5则无权平均优于加权，权重优势没有迁移。宏平均的微小变化只来自各一个normal5，不作统计显著性或稳定收益主张。保守门限使影响范围极小；扩大半径、放宽门限或稠密传播属于后续方法变更，本次没有为追求收益而进行。','',
        'failure_analysis.json 列出所有被修正点到固定参考样本的距离改善/恶化、2mm正确点变错误点和最差点索引。它是锁定后的GT后验诊断，不参与推理或重新拟合。即使通过对应和权重检查，也会出现点距离恶化；例如开发normal5的174个修正点中，84个距离恶化，67个从2mm以内变为2mm以外（此处为全部修正点、未套ROI的后验统计，与主表采样/ROI范围不同）。保留这些反例。','',
        '## 资源、验证与产物','',
        f"8次冻结融合耗时合计 {incremental:.4f}s，native轨迹导出合计 {exports:.4f}s；包含文件校验、加载、压缩和落盘，融合计时也包含同对应无权平均消融的写出，不含进程启动和旧模型推理。CPU线程8，新增GPU分配0；未加载Torch、模型或新权重。峰值融合RSS、导出RSS、评价耗时与RSS见cost_accounting.json。整卡nvidia-smi占用另记于environment.json，不能当作本步骤分配。满足本地12GB预算。",
        '',
        '成本另完整保留此前eager选择运行的所有分支、诊断、BA OOM成本，并加本次导出/融合；它是历史阶段计时相加，不是新在线端到端延迟，不支持速度提升主张。此次实际复用raw/COLMAP，不运行旧点过滤或BA；相应最小两分支历史计时也单列。开发候选与失败尝试不计入部署增量，但全部保留。','',
        'before：outputs/stage5_v1/<scene>/selected_artifacts/raw_predictions.npz；after：outputs/stage5_fusion_v1/frozen/<scene>/fused.npz。anchors.npz包含源网格索引、COLMAP point ID、对齐锚点、两种权重与raw跨视图支持；fused.npz包含xyz、原始索引和实际β；unweighted.npz保留消融。统一prediction.json保留raw相机、坐标系和原始depth/confidence，明确新pointcloud语义。Git发布轻量逐场景JSON/规则/代码/报告，大型NPZ与模型、数据保持本地。','',
        '验证包含闭式公式、零权重强度、不允许无效权重、无对应回退、退化/缺失相机、预测相机不一致、真实观测半像素变换、碰撞、负深度和错误几何门限、float32未改字节。96项不同测试在可靠性环境（跳过1项native COLMAP）和COLMAP环境（跳过2项OpenCV、1项SciPy）交叉覆盖；跳过项在另一环境通过。validation.json验证8例全部输入hash、相机不变、公式复算、未改点字节、无GT访问日志与开发规则复现。','',
        '## 复现','',
        '```bash','export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8','PY=/home/zchming/.venvs/georecon-reliability/bin/python',
        '# 原有上游产物与data/stage5_fusion_tracks需在本地；所有生成命令拒绝覆盖。',
        '$PY scripts/stage5_fusion.py generate --scene dtu_scan6_normal5 --output-root outputs/stage5_fusion_reproduction',
        '$PY scripts/stage5_fusion.py fit --verify-frozen',
        '$PY scripts/stage5_fusion.py evaluate',
        '$PY scripts/check_stage5_fusion.py',
        '$PY -m unittest discover -s tests -v','```','',
        '本子步骤完成后停止；不进行联合优化或扩展数据/资源配置。','']
    (DOC/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':report()
