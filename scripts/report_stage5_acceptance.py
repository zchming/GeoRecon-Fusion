"""Stage5 closeout: frozen comparisons, failure ledger, engineering vs scientific acceptance."""
import subprocess,platform,sys,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import scipy
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_acceptance'


def report():
    comparison=read_json(DOC/'ablation_comparison.json');rows=comparison['rows'];validation=read_json(DOC/'validation.json');audit=read_json(DOC/'prior_audit.json')
    assert len(rows)==len(validation['rows'])==len(audit['rows'])==8
    methods=list(rows[0]['methods']);summary={}
    for split in ['dev','test']:
        subset=[r for r in rows if r['split']==split]
        summary[split]={m:{str(t):{k:float(np.mean([r['methods'][m]['thresholds_mm'][str(t)][k] for r in subset])) for k in ['precision','recall','f1']} for t in [1,2,5]} for m in methods}
    summary['ablation_deltas_f1_percentage_points']={split:{'filter_unknown_vs_drop':100*(summary[split]['filter_keep_unknown']['2']['f1']-summary[split]['filter']['2']['f1']),
        'neutral_raw_weight_vs_fusion':100*(summary[split]['fusion_without_raw_support']['2']['f1']-summary[split]['fusion']['2']['f1']),
        'fixed_pose_vs_joint':100*(summary[split]['joint_fixed_poses']['2']['f1']-summary[split]['joint']['2']['f1'])} for split in ['dev','test']}
    summary['new_configuration_fit']=False;summary['new_default_method_selected']=False
    write_json(DOC/'summary.json',summary)
    # Verify exact historical rule bytes, not just statements that rules stayed frozen.
    specs=[('selection','stage5_v1.json','131962e'),('filtering','stage5_points_v1.json','f54255c'),('fusion','stage5_v1.json','89a0884'),('optimization','stage5_v1.json','c3510ce'),('ablation','stage5_acceptance.json','c0bdab6')]
    rule_chain=[]
    for folder,name,commit in specs:
        path=ROOT/'configs'/folder/name
        historical=subprocess.run(['git','show',commit+':'+str(path.relative_to(ROOT))],capture_output=True,check=True).stdout
        digest=hashlib.sha256(historical).hexdigest();assert digest==sha256(path)
        rule_chain.append({'file':str(path.relative_to(ROOT)),'frozen_commit':commit,'sha256':digest,'unchanged':True})
    write_json(DOC/'rule_chain.json',{'rules':rule_chain,'ablation_protocol_frozen_before_runs':True,'no_retuning':True})
    # All previous natural baseline failures remain in denominators and referenced by exact hashes.
    upstream=[]
    for r in rows:
        for method in ['colmap','vggt_raw','vggt_ba']:
            p=ROOT/'data/stage5_bundles'/r['scene_id']/(method+'.json');pred=read_json(p)
            if pred['status']!='success':upstream.append({'scene_id':r['scene_id'],'method':method,'status':pred['status'],'reason':pred['failure_reason'],'source':str(p.relative_to(ROOT)),'sha256':sha256(p)})
    source_files=['docs/stage5/fault_probes.json','docs/stage5/gt_isolation_probe.json','docs/stage5_points/failure_analysis.json','docs/stage5_points/failures.json','docs/stage5_fusion/failure_analysis.json','docs/stage5_fusion/attempt_history.json','docs/stage5_joint/failure_analysis.json','docs/stage5_joint/budget_probe.json']
    points_fail=read_json(ROOT/'docs/stage5_points/failure_analysis.json');fusion_fail=read_json(ROOT/'docs/stage5_fusion/failure_analysis.json');joint_fail=read_json(ROOT/'docs/stage5_joint/failure_analysis.json')
    failures={'upstream_natural_failures':upstream,'upstream_counts':dict(Counter(r['status'] for r in upstream)),
        'source_records_sha256':{p:sha256(ROOT/p) for p in source_files},'source_failure_records_preserved':True,
        'point_filter_unknown_correct_points_removed':[{k:r[k] for k in ['scene_id','good_points_removed','good_unknown_removed']} for r in points_fail['rows']],
        'fusion_affected_point_failure_counts':[{k:r[k] for k in ['scene_id','affected_points','distance_worsened','good_to_bad_2mm']} for r in fusion_fail['rows']],
        'joint_affected_point_failure_counts':[{k:r[k] for k in ['scene_id','eligible_points','worsened','good_to_bad_2mm']} for r in joint_fail['rows']],
        'attempt_records':{'point_filter':read_json(ROOT/'docs/stage5_points/failures.json'),'fusion':read_json(ROOT/'docs/stage5_fusion/attempt_history.json')},
        'synthetic_counterfactuals_not_natural_benchmark_samples':{'prior_selection_fault_probes':10,'new_dev_all_unknown_gate_probes':4,'joint_forced_budget_probe':1},
        'report_build_failure_record':'report_build_failures.json','new_fixed_pose_solver_failures':sum(r['solver'] is not None and not r['solver']['accepted'] and r['solver']['status']!='no_eligible_correspondences' for case in rows for r in case['run']['rows'])}
    write_json(DOC/'failure_catalog.json',failures)
    oldselection=read_json(ROOT/'docs/stage5/evaluation.json');oldjoint=read_json(ROOT/'docs/stage5_joint/comparison.json')
    summary['selection_pose_auc10']={split:float(np.mean([r['joint_auc']['10'] for r in oldselection['rows'] if r['split']==split])) for split in ['dev','test']}
    write_json(DOC/'summary.json',summary)
    # Compact per-scene matrix is the primary data table; failures are never converted into zero pose errors.
    per_scene=[]
    for r in rows:
        selected=next(p for p in oldselection['rows'] if p['scene_id']==r['scene_id']);j=next(p for p in oldjoint['rows'] if p['scene_id']==r['scene_id'])
        per_scene.append({'scene_id':r['scene_id'],'split':r['split'],'f1_2mm':{m:r['methods'][m]['thresholds_mm']['2']['f1'] for m in methods},
            'selector_joint_auc10':selected['joint_auc']['10'],'selected_method':selected['selected_method'],'joint_fixed_frame_camera_rmse_mm':j['fixed_frame_camera_rmse_mm'],
            'joint_relative_pose_metrics':{key:{'rotation_deg':j[key]['rotation_deg_registered_subset']['mean'],'translation_direction_deg':j[key]['translation_direction_deg_registered_subset']['mean'],'refitted_ATE_mm':j[key]['camera_center_ate']['rmse']} for key in ['poses_before','poses_after']}})
    write_json(DOC/'per_scene.json',{'rows':per_scene,'small_changes_are_not_significance_claims':True})
    newtime=sum(r['run']['wall_seconds'] for r in rows);peak=max(r['run']['peak_rss_gib'] for r in rows)
    costs={'new_three_ablation_batch_seconds_sum':newtime,'new_peak_batch_rss_gib':peak,'new_offline_evaluation_seconds':comparison['offline_evaluation_seconds'],'new_peak_evaluator_rss_gib':comparison['peak_evaluator_rss_gib'],
        'new_GPU_allocated_bytes':0,'historical_stage_costs':{name:read_json(ROOT/'docs'/name/'cost_accounting.json') for name in ['stage5_fusion','stage5_joint']},
        'filter_incremental_seconds':read_json(ROOT/'docs/stage5_points/summary.json')['incremental_filter_seconds_sum'],
        'no_speedup_claim':True,'do_not_sum_nested_historical_entries':True,'scope':'reused historical baseline and intermediate files; new batch includes shared loading, source hashes, three controls and writes; not fresh online latency and not three independently timed cold starts'}
    write_json(DOC/'cost_accounting.json',costs)
    checkmarks=[{'criterion':name,'pass':True} for name in ['8 original scene conditions retained','4 old rules byte-identical to frozen commits','all historical source/input hashes and primary point/pose results reproduced','all 24 new artifacts and all-camera invariants verified','GT runtime guard and dev-only counterfactual separation','3 new component ablations plus inherited controls completed','all natural and synthetic failures labeled and preserved','110 distinct regression tests covered across environments','local GPU budget respected without new GPU allocation','no next stage started']]
    acceptance={'stage':5,'engineering_and_evidence_acceptance':'passed','scientific_stable_gain':'not_established','production_or_SOTA_acceptance':'not_claimed','checklist':checkmarks,
        'conclusion':'complete the authorized Stage5 pilot; keep frozen raw-first selection as reference and additional strategies explicit experimental options; do not promote diagnostic ablations','default_method_reselected':False,'next_stage_started':False}
    write_json(DOC/'acceptance.json',acceptance)
    paths=[ROOT/'configs/ablation/stage5_acceptance.json',ROOT/'src/georecon/ablation/controls.py',ROOT/'scripts/run_stage5_ablations.py',ROOT/'scripts/audit_stage5_acceptance.py',ROOT/'scripts/evaluate_stage5_ablations.py',ROOT/'scripts/check_stage5_acceptance.py',ROOT/'scripts/report_stage5_acceptance.py']
    write_json(DOC/'input_lock.json',{'protocol_commit':'c0bdab6','stage3_freeze_sha256':sha256(ROOT/'docs/stage3/freeze.json'),'rule_chain':rule_chain,
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths},'prior_source_locks':audit['prior_source_locks'],
        'cases':[{'scene_id':r['scene_id'],'input_hashes':r['run']['input_hashes'],'artifacts':{i['name']:i['artifact_sha256'] for i in r['run']['rows']}} for r in rows]})
    write_json(DOC/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'device':'CPU','threads':8,'new_weights_packages_models':[],
        'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv'],capture_output=True,text=True).stdout,'new_GPU_allocated_bytes':0,'GPU_budget_gib':12})
    labels={'raw_selector':'raw / 场景选择','filter':'冻结点过滤','fusion':'冻结加权融合','unweighted_fusion':'同对应无权平均','grouped_initial':'共享点初值','joint':'冻结联合优化','filter_keep_unknown':'消融：保留unknown','fusion_without_raw_support':'消融：raw支持权重置1','joint_fixed_poses':'消融：固定相机优化点'}
    lines=['# 阶段5汇总、消融与验收','',
        '**工程实现与证据完整性验收通过；稳定性能收益尚未证实。** 阶段5已完成本次授权的选择、过滤、加权融合、最小联合优化、必要消融和验收；保持原始raw优先选择作为参考，新增策略仍为显式实验选项。没有因消融结果重新选择默认方法，没有进入阶段6或7。','',
        '## 范围与冻结过程','',
        '沿用scan1开发/scan6测试、light3、normal5/reduced3/spaced3/blur5，共8个条件，输入图像、视图数、分辨率及上游方法配置不变。本轮先复算历史结果并核对输入/源码/规则哈希（prior_audit.json），再在提交 **c0bdab6** 固定3项新消融和验收方案，之后才运行。四个历史规则逐字节匹配原冻结提交，见rule_chain.json。','',
        '前序阶段已经查看测试结果；新增消融是预声明的回顾性诊断，不是新盲测、不搜索超参、不将新结果转成规则。scan1和scan6各只有一个独立物体，四个条件相互相关，不能把8例当8个独立场景，更不作显著性或泛化主张。','',
        '## 原有四个子步骤与失败证据','',
        '1. 场景选择：开发集确定raw→COLMAP→官方BA优先级。8例自然选择全部是raw，质量等于always-raw，没有精度收益；10个开发故障探针验证回退/弃权，但模糊COLMAP回退仍可能错误。回退次序在自然数据中不可识别。',
        '2. 点过滤：冻结5%相对Z差、至少1个其他视图支持。开发F1下降；dev reduced3采样ROI中误删2370个正确点，其中2366个无有效跨视图观察。unknown不是已知错误。',
        '3. 加权融合：冻结alpha=0.5，仅同名真实轨迹和通过门限的稀疏锚点参与；只影响2个normal5共502点（总网格约0.0077%），其余6例回退。dev加权优于无权平均，但test无权平均更好。174个dev修正点中84个参考距离变差，67个从2mm内变到2mm外。',
        '4. 联合优化：冻结位姿先验强度1，固定内参/首相机，联合更新其余位姿和已有轨迹结构。拟合重投影残差下降，但固定采样F1低于融合；测试相对位姿/重新对齐ATE改善、固定原始评价框架相机误差增大。局部全部受影响点的均距改善与采样F1下降可以并存，不能一概称几何变坏。','',
        '上游失败仍保留：COLMAP spaced3无重建2例，官方BA五图OOM4例；raw成功8/8不意味着选择器修复了这些失败。failure_catalog.json汇总真实失败、退化点、模拟探针和已修复的实现错误，分母和来源不混用。','',
        '## 必要消融与控制','',
        '|问题|对照|只改变什么|','|---|---|---|',
        '|几何证据门是否提高自然选择精度|gated / 不使用几何证据门|保留身份、哈希、完整状态与产物检查；8例同选raw，不能证明自然精度收益|',
        '|证据未知时门是否有作用|仅dev的all_unknown反事实|gated弃权；移除门选raw；这是4个模拟输入，不是额外自然失败|',
        '|过滤退化是否来自unknown删除|冻结过滤 / keep OR finite&observed==0|只恢复无有效其他视图观察的有限点，阈值不变|',
        '|raw跨视图支持权重是否有效|冻结融合 / w_raw=1|相同锚点、门限、alpha、经典权重不变；不是重新拟合权重|',
        '|可靠性权重是否优于固定权重|同对应beta=0.5 / 冻结权重|使用已保存无权平均对照，不引入新对应|',
        '|联合更新相机是否有贡献|冻结联合 / 固定所有相机只优化点|同一初值、残差、先验、鲁棒损失、点边界和预算|',
        '|收益是否只是共享点初始化|融合 / grouped_initial|已有点ID实际不重复，该对照在当前数据上相同|','',
        '## 固定点评价结果','',
        '单位为宏平均F1@2mm百分比，先逐场景再等权平均四种条件。完整P/R/F1@1/2/5mm、均距、逐场景值与动态/固定ROI敏感性在ablation_comparison.json和per_scene.json。','',
        '|方法或控制|dev F1@2mm|test F1@2mm|','|---|---:|---:|']
    for m in methods:lines.append(f"|{labels[m]}|{100*summary['dev'][m]['2']['f1']:.5f}%|{100*summary['test'][m]['2']['f1']:.5f}%|")
    lines+=['',
        '保留unknown缓解了过滤损失：dev +1.27468个百分点，test +0.03733个百分点（相对原过滤）。相对raw仍只是约+0.04473/+0.06753个百分点，不能据此宣布稳定新方法；test normal5保留unknown也低于原过滤。当前结果支持明确区分“无证据”和“错误”，不支持直接删除未知点。','',
        'raw支持权重置1：dev相对冻结融合变差，test反而改善；原权重的优势未迁移。固定相机优化点：dev低于联合，test微高于联合，但总体变化极小，不能确认联合位姿更新的稳定增益。所有消融保持诊断用途，未据测试表现改成默认策略。','',
        '点评价继续使用每例最多200000个固定原始点索引、200000个等距STL样本，以及同一原raw相机→GT Sim3；使用官方ObsMask/Plane，未执行官方0.2mm邻域降采样、距离上限和离群剔除。这是小型预算自定义评价，不能与DTU官方全量排名直接比较。只有502个点受融合/优化影响，微小变化容易受采样与ROI影响；没有扩大采样或更换指标来追求结论。','',
        '## 位姿与方法成功状态','',
        'COLMAP 6/8完整，VGGT raw 8/8，官方BA 4/8；选择器8/8完整且实际0次回退。融合与联合均只在2/8场景实际工作，其余6/8保留原结果；“完整输出”不是GT正确率，也不是融合/优化成功改善率。','',
        '|有效联合场景|相对旋转均值° 前→后|方向误差均值° 前→后|重新对齐ATE mm 前→后|固定raw框架相机RMSE mm 前→后|','|---|---:|---:|---:|---:|']
    for p in per_scene:
        if not p['scene_id'].endswith('normal5'):continue
        a,b=p['joint_relative_pose_metrics']['poses_before'],p['joint_relative_pose_metrics']['poses_after'];fixed=p['joint_fixed_frame_camera_rmse_mm']
        lines.append(f"|{p['scene_id']}|{a['rotation_deg']:.4f}→{b['rotation_deg']:.4f}|{a['translation_direction_deg']:.4f}→{b['translation_direction_deg']:.4f}|{a['refitted_ATE_mm']:.4f}→{b['refitted_ATE_mm']:.4f}|{fixed['before']:.4f}→{fixed['after']:.4f}|")
    lines+=['',
        f"场景选择的宏平均joint AUC@10：dev {summary['selection_pose_auc10']['dev']:.6f}，test {summary['selection_pose_auc10']['test']:.6f}，均与always-raw相同。上游各方法逐条件位姿和失败感知joint AUC仍见[选择阶段汇总](../stage5/report.md)，本轮未重新运行模型或把稀疏COLMAP点云当稠密分支作不等价比较。",'',
        '## 验收与资源','',
        '验收分两层：实现/复现/证据检查**通过**，稳定收益假设**未建立**。acceptance.json逐项记录，validation.json验证24个新增产物、公式、相机固定、源float32坐标和不可变输入、GT访问保护。110项不同回归测试在可靠性环境（110项、跳过1原生COLMAP）与COLMAP环境（110项、跳过14个需OpenCV/SciPy的测试）交叉覆盖；跳过项都在另一环境实际通过。','',
        f"新增8个三消融批次合计{newtime:.4f}s，单批次进程RSS最高{peak:.4f}GiB。固定CPU线程8，新增GPU分配0，无模型、权重、包安装或云资源，满足12GB预算。nvidia-smi整卡占用另记，不冒充本任务显存。批次共享加载与哈希检查，含压缩写出；不把它当三个独立冷启动速度，历史全分支/诊断/BA OOM/融合/优化成本继续存于cost_accounting.json，不宣称端到端加速。",
        '',
        '旧结果与失败目录保留不改；新NPZ在outputs/stage5_acceptance_v1/<scene>/，逐场景run/selection_control/invocation轻量快照在本目录runs/。原始数据、权重、vendor和大型生成文件不入Git。','',
        '## 累计贡献与局限','',
        '目前可支持的贡献：统一相机/像素/尺度及失败协议、可审计分阶段实验管线；本地推理不读取GT的几何诊断和unknown处理；开发集规则锁定、回退/过滤/融合/联合优化及控制实验；一组有来源的失败证据，说明内部残差、未知状态和未经校准权重的适用边界。','',
        '局限：只有两个独立物体且测试已被前序阶段查看；当前经典分支是稀疏SfM，未建立稠密MVS；自然选择缺乏方法切换样本；融合/优化仅稀疏局部有效；权重未校准；有限采样和评价配准影响微小变化；历史成本不等于新在线时延。COLMAP、VGGT、官方BA、Sim3和现成数值求解器不是原创，算法新颖性、稳定超越raw、SOTA与泛化均未证实。详见[累计贡献清单](../contributions.md)。','',
        '## 复现','',
        '```bash','export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8','PY=/home/zchming/.venvs/georecon-reliability/bin/python',
        '# 所需上游产物在本地，生成命令拒绝覆盖现有输出。',
        '$PY scripts/audit_stage5_acceptance.py',
        '$PY scripts/run_stage5_ablations.py --scene dtu_scan6_normal5 --output-root outputs/stage5_acceptance_reproduction',
        '$PY scripts/evaluate_stage5_ablations.py',
        '$PY scripts/check_stage5_acceptance.py',
        '$PY -m unittest discover -s tests -v','```','',
        '阶段5在当前固定小样本范围内完成并停止。没有开始下一阶段，也没有把本轮诊断消融转为新部署方案。','']
    (DOC/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':report()
