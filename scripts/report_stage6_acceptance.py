"""Stage6 acceptance report and factual contribution boundary, no algorithm tuning."""
from pathlib import Path
import platform,re,subprocess
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_acceptance'


def test_coverage():
    logs={};coverage={};skips={}
    for env in ['reliability','colmap']:
        content=(DOC/('regression_'+env+'.txt')).read_text();assert re.search(r'Ran 134 tests',content) and '\nOK (skipped=' in content
        rows={}
        for line in content.splitlines():
            m=re.match(r'(test_\w+ \([^)]+\)) \.\.\. (.+)$',line)
            if m:rows[m[1]]=m[2]
        assert len(rows)==134;assert all(value=='ok' or value.startswith('skipped ') for value in rows.values());coverage[env]={key for key,value in rows.items() if value=='ok'};skips[env]={key for key,value in rows.items() if value.startswith('skipped ')}
        logs[env]={'discovered':len(rows),'passed':len(coverage[env]),'skipped':len(skips[env]),'log_sha256':sha256(DOC/('regression_'+env+'.txt'))}
    assert coverage['reliability']|coverage['colmap']==coverage['reliability']|skips['reliability'];assert not skips['reliability']&skips['colmap']
    return {'environments':logs,'distinct_tests_passed_in_at_least_one_environment':len(coverage['reliability']|coverage['colmap']),'skipped_in_both_environments':[],'teaching_test_counts':{'two_view':10,'PnP_incremental':6,'BA':8},'regression_count_includes_prior_stages':True}


def report():
    audit=read_json(DOC/'audit.json');coverage=test_coverage();assert audit['all_historical_files_unchanged'];write_json(DOC/'test_coverage.json',coverage)
    pair=read_json(ROOT/'docs/stage6/summary.json');inc=read_json(ROOT/'docs/stage6_incremental/evaluation.json');ba=read_json(ROOT/'docs/stage6_ba/evaluation.json')
    matrices=[]
    for r in inc:
        b=next(v for v in ba if v['case_id']==r['case_id']);matrices.append({'case_id':r['case_id'],'kind':r['kind'],'incremental_status':r['status'],'registered':r['registered'],'input_images':r['input_images'],'points_before_and_after_BA':r['points'],'PnP_inliers_in_fixed_registration_order':[step['inliers'] for step in r['pnp_steps']],'BA_reason':b['BA_reason'],'BA_converged':b['converged'],'before_BA':b['before'],'after_BA':b['after']})
    write_json(DOC/'results.json',{'two_view':pair,'incremental_and_BA_same_registered_set_points_and_observations':matrices,'independent_real_objects':1,'DTU_scan':1,'conditions':['reduced3','normal5','blur5','spaced3'],'no_new_data_or_configuration_search':True,'not_a_new_benchmark_or_blind_test':True})
    failures=[]
    for r in audit['all_21_run_records_and_exits_verified']:
        stage=r['stage'];base=ROOT/'outputs'/dict((s,o) for s,o,c in __import__('accept_stage6').STEPS)[stage]/r['case_id']
        if stage=='stage6_ba':
            stats=read_json(base/'ba.json');reason=stats['reason'];category='solver_evaluation_budget' if reason=='accepted_evaluation_budget_limited' else ('not_applicable_or_failed_input_map' if not stats['accepted'] else None)
        else:
            p=read_json(base/'prediction.json');reason=p['failure_reason'];category=('synthetic_expected_rejection' if r['kind']=='synthetic' else 'real_failure') if reason else None
        if category:failures.append({'stage':stage,'case_id':r['case_id'],'kind':r['kind'],'category':category,'reason':reason,'original_status':r['status']})
    for row in read_json(ROOT/'docs/stage6_ba/failure_catalog.json'):
        if row['reason'].startswith('accepted_BA_has_'):failures.append({'stage':'stage6_ba','kind':'real','category':'accepted_solution_tradeoff',**row})
    for row in read_json(ROOT/'docs/stage6_ba/synthetic_controls.json'):failures.append({'stage':'stage6_ba','case_id':row['case_id'],'kind':'synthetic','category':'synthetic_expected_rejection','reason':row['stats']['reason'],'explicit_resource_profile_change':row['explicit_profile_change']})
    write_json(DOC/'failure_catalog.json',{'cases':failures,'counting_note':'same seed failure recurs downstream; records are not independent scenes; expected controls are not natural failures','point_accuracy_or_completeness':'not evaluated against GT for teaching sparse maps'})
    configs={stage:{'path':'configs/teaching/'+config,'sha256':sha256(ROOT/'configs/teaching'/config)} for stage,out,config in __import__('accept_stage6').STEPS}
    write_json(DOC/'protocol.json',{'scope':'stage6 teaching SfM acceptance only','intrinsics':'known per-image distortion-free K supplied explicitly, K_processed=A K_original, zero-based pixel centers','images':'existing DTU scan1 only, Pillow bilinear 1600x1200 to 800x600, no crop or view substitution','inference_truth_policy':'K declared input; no pose, scale or point reference file reads in recorded inference; GT evaluation only','gauge_before_BA':'seed camera1 I/0 and unit seed translation norm','gauge_after_BA':'first seed pose fixed; second seed center largest initial coordinate component relative to first is fixed; baseline norm can change, scale still arbitrary','BA_unchanged_entities':['registered image set','input order','K','point IDs/count','RGB','keypoint observations'],'configurations':configs,'boundary':'no stage7, no benchmark extension, no new reliability/fusion/training/UI','source_provenance':'historical Git package plus locked uncommitted source bytes reconstructed; added modules legitimately change the current full-package hash'})
    snapshot=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip();apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    resources={stage:read_json(ROOT/'docs'/stage/'cost_accounting.json') for stage,_,_ in __import__('accept_stage6').STEPS}
    write_json(DOC/'resources.json',{'historical_profiles':resources,'acceptance_check_GPU_snapshot':snapshot,'acceptance_check_compute_apps_snapshot':apps,'acceptance_no_CUDA_or_model_calls':True,'GPU_budget_GiB':12,'current_snapshot_is_not_a_measured_peak':True,'audit_wall_seconds':audit['seconds'],'not_end_to_end_speed_benchmark':'different substeps include repeated images/controls and timing scopes; no aggregate pipeline speed claim','environment':{'python':platform.python_version(),'numpy':__import__('numpy').__version__,'opencv':__import__('cv2').__version__,'scipy':__import__('scipy').__version__}})
    requirements=[{'requirement':r,'status':'passed','evidence':e} for r,e in [
        ('feature extraction and matching','11 pair and 5 sequence replays; saved stable keypoint and match indices'),('ratio/mutual filtering and RANSAC','two_view.py, frozen configs and all masks/hypotheses replayed'),('known-K essential matrix and pose recovery','four R/t hypotheses, I/0 first camera and unit baseline checked'),('triangulation and cheirality','replayed DLT, positive depths, >=0.5 degree parallax and <=2px initial observation gates'),('PnP and incremental registration','fixed-order registered-set outputs, unique track IDs; blank-view rejection preserves map'),('reprojection error','independent before/after observation reprojection, complete residual and rejection arrays'),('small bundle adjustment','sparse fixed-K joint poses/XYZ, gauge checks, unchanged points/tracks/K, convergence labels'),('scope, degeneracies and reproducibility','failure_catalog, historical source/input/output locks, synthetic controls and test coverage')]]
    acceptance={'stage':6,'scope':'minimal explainable calibrated teaching SfM on existing local DTU small samples','stage_acceptance':'passed','engineering_and_evidence_acceptance':'passed','requirements':requirements,'historical_cases_verified':{'two_view':11,'incremental':5,'BA':5,'additional_BA_synthetic_failure_controls':4},'distinct_tests_cross_environment':coverage['distinct_tests_passed_in_at_least_one_environment'],'known_K_preserved':True,'old_records_and_all_comparable_failures_preserved':True,'no_inference_pose_or_scale_truth_reads_recorded':True,'all_BA_runs_converged':False,'novel_algorithm_or_general_performance_gain_established':False,'research_project_complete':False,'stage7_started':False,'GPU_budget_GiB':12,'stop_at_stage_boundary':True}
    write_json(DOC/'acceptance.json',acceptance)
    lines=['# 阶段6：教学版 SfM 阶段验收','', '**阶段6最小教学范围的工程与证据验收通过。** 已覆盖项目要求的匹配、RANSAC、已知K的E/相对位姿、三角化、PnP、增量注册、重投影计算及小规模BA。并不代表完整COLMAP工程能力、整个研究项目完成或性能创新。停止于阶段6，不进入阶段7。','', '## 已验收的范围与协议','', '本轮仅汇总和检查已有实现，没有修改算法、配置、图像数量、分辨率或预算，没有新增数据、模型、实验条件或规则搜索。沿用scan1的现有DTU小样本：normal5、reduced3、blur5、spaced3；它们是同一物体的重叠条件，不是四个独立场景。','', 'K为明确提供的已知标定；800×600、零起点像素中心、双线性resize和K_processed=A K_original保持不变。DTU真实R/t、尺度和参考点仅用于独立离线评价，不进入记录中的推理进程；审计钩子是防误用护栏，不是OS沙箱。不能把此known-K教学结果与此前自估内参COLMAP作同条件排名。','', '双视图/增量模型以首相机I/0和单位种子基线定规；BA固定首相机以及第二种子中心的一个非零坐标分量，仍为任意尺度，BA后不要求种子基线范数为1。BA保留全部注册状态、K、点ID/数量、颜色与观测，不重新注册、过滤或合并点。','', '## 双视图结果','', '|实际DTU图像对|互检匹配|接受点|相对旋转误差°|平移方向误差°|结果|','|---|---:|---:|---:|---:|---|']
    for r in pair['real_matches_and_points']:
        fmt=lambda v:'—' if v is None else f'{v:.3f}'
        lines.append(f"|{r['case_id']}|{r['mutual_matches']}|{r['kept_points']}|{fmt(r['rotation_error_deg'])}|{fmt(r['translation_direction_error_deg'])}|{r['status']}|")
    lines += ['', '8组控制均符合预期：非平面精确/噪声离群点两组成功；空白、重复图像、平面、纯旋转、极小基线、匹配不足六组拒绝。单应性占优是保守歧义门限，不能据此证明场景一定平面或纯旋转；模糊近邻对这次位姿误差较小，也不代表模糊有益。','', '## PnP、增量注册与BA前后','', '|序列|注册|点数（BA前后相同）|PnP内点（注册顺序）|重投影RMSE px：前→后|BA状态|','|---|---:|---:|---|---|---|']
    for r in matrices:
        a,b=r['before_BA']['reprojection_rmse_px'],r['after_BA']['reprojection_rmse_px'];change='—' if a is None else f'{a:.3f} → {b:.3f}'
        lines.append(f"|{r['case_id']}|{r['registered']}/{r['input_images']}|{r['points_before_and_after_BA']}|{', '.join(str(i) for i in r['PnP_inliers_in_fixed_registration_order']) or '—'}|{change}|{r['BA_reason']}|")
    lines+=['', '普通/模糊近邻使用固定种子001/003，按清单顺序注册剩余图像，每图一次；宽间隔固定种子001/020失败后不搜索替代种子。空白002为人工控制，不是额外真实样本；PnP无对应后维持partial和原404点。BA本实现要求至少3台注册相机，因此跳过此双视图残余模型；这不是上游求解器不能处理两视图BA。','', '|三组成功地图|平均相对旋转误差°：前→后|平均平移方向误差°：前→后|重新拟合Sim3的ATE mm：前→后|','|---|---|---|---|']
    for r in matrices:
        if not r['before_BA']['registered_images']>=3:continue
        def change(key):return f"{r['before_BA'][key]:.3f} → {r['after_BA'][key]:.3f}"
        lines.append(f"|{r['case_id']}|{change('mean_pair_rotation_deg')}|{change('mean_pair_translation_direction_deg')}|{change('camera_center_ATE_refit_Sim3_mm')}|")
    lines += ['', '这些位姿指标是相同已注册集合中所有图像对的平均，未注册对未填零。三图与五图中的同一图像/点会重复出现，不能把这些数字当独立重复实验。稀疏点数量增长未证明GT点云精度或完整度改善，这两项尚未评价。','', '## 失败、退化与预算限制','', '- 宽间隔001/020仅4个互检匹配，无法初始化；下游不能靠PnP或BA恢复不存在的地图。它在多个步骤重复出现，不能累计成多个独立失败场景。','- 空白新增视图无2D–3D对应，PnP拒绝，保留原模型；BA按已注册相机数跳过。','- 普通五图BA达到锁定的80次主函数评估预算、未收敛；候选满足有限/正深度/鲁棒目标不增后接受。没有延长预算或重试，不能称全部BA收敛。','- 普通五图最大观测误差1.991→2.516px，模糊五图1.923→2.860px，各有3个观测超过2px。平均误差下降不保证尾部改善；BA预定接受条件不包括逐观测2px硬门限，也未事后删去这些观测。','- 三组在固定优化前Sim3下的相机ATE分别1.290→2.334、2.288→6.594、1.762→5.195mm，与重新拟合Sim3后的改善方向相反。该固定框架指标依赖初始对齐和尺度约定，不能直接等同于全部几何变差；重新对齐改善也不能证明固定框架改善。','- BA共线、零基线、观测图断开、强制超时四组控制均显式拒绝。超时控制单独把墙钟预算设为0秒，其余真实运行仍为60秒；这不是自然DTU超时。','', '完整逐案例目录见[failure_catalog.json](failure_catalog.json)。未发现新的运行失败，也没有用重新求解替换原来的预算限制或退化结果。','', '## 必要检查与验收证据','', f'本轮核对{audit["historical_file_count"]}个历史文件字节哈希，验证21条原始运行与退出码、输入与配置/源码锁、全部BA控制产物。11组双视图特征与几何数组、5组增量流程重新复算；BA采用独立逐观测重投影、鲁棒目标、正深度与固定规范检查，不重复优化或改预算。','', '全包源码哈希会因后续新增模块而改变。本轮按原运行Git提交与当时锁定的新增源码字节重建历史包哈希，逐运行验证，未把当前哈希强行替换进旧记录。旧增量/BA检查器的写入改为只比较既有JSON，所有原记录、轻量报告和本地产物均保持不变。','', '134项不同测试在两个环境重新运行并交叉覆盖：OpenCV/SciPy环境133通过、1跳过；COLMAP环境100通过、34跳过；没有两边都跳过的测试。阶段6相关24项（双视图10、PnP/增量6、BA8），其余为此前模块回归检查；没有把134都算作新SfM测试。','', '|要求|证据|验收|','|---|---|---|']
    for r in requirements:lines.append(f"|{r['requirement']}|{r['evidence']}|通过|")
    lines += ['', '机器可读结论见[acceptance.json](acceptance.json)，全检查见[audit.json](audit.json)与[test_coverage.json](test_coverage.json)。验收通过指获授权的小型教学范围具备实现、解释、复现和失败处理证据；不要求所有输入成功、所有BA收敛或每项误差改善。','', '## 资源与复现','', '|历史子步骤|记录规模|主体总秒|控制器总秒|任务新增显存|','|---|---|---|---|','|双视图|3真实对+8控制|1.302|4.023|0|','|增量注册|4真实条件+1空白控制|1.363|2.532|0|','|BA|同5组已有地图|6.096（求解3.344）|7.967|0|','', '不同步骤包含重复视图/控制且计时范围不同，不能相加宣称完整流水线速度；控制器包含进程启动和显卡采样开销。历史峰值RSS最高约0.461GiB。均使用8线程CPU OpenCV/NumPy/SciPy，无CUDA模型；历史整卡采样约503–509MiB/12282MiB，包含桌面，不是任务分配量。','', f'本次验收显卡快照：`{snapshot}`。这是检查时快照，不是峰值测量；本轮只做CPU复算和测试，无新增GPU算法或模型。实际资源记录见[resources.json](resources.json)，WSL进程枚举有限，预算保持12GB以内。','', '```bash','PYTHONPATH=src OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python scripts/accept_stage6.py','PYTHONPATH=src python -m unittest discover -s tests -v','PYTHONPATH=src python scripts/report_stage6_acceptance.py','```','', '需本地原DTU输入与历史outputs，Git不含图像、权重或大型NPZ。需要具有OpenCV/SciPy的teaching、optimization依赖环境；完整原始求解命令与单次运行记录保留在各子步骤报告。本脚本只复算/核对旧结果，不重跑特征配置搜索、注册求解预算或BA实验；报告脚本还要求本轮两份回归日志已存在。','', '## 累计贡献与局限','', '累计建立COLMAP/VGGT/官方BA统一接口、坐标与预处理协议、小样本固定对比、无GT几何诊断，以及选择/过滤/融合/联合优化和必要消融的证据链。本阶段补齐经典已知K教学SfM全链路，用中间产物、固定规范、失败处理和跨环境测试验证基础几何实现。','', '贡献目前以可审计工程、可解释教学和条件有限的经验发现为主。阶段5稳定融合收益仍未建立；阶段6仅一个物体、重叠条件、已知K、固定种子、单次有限预算，没有点云真值质量验证、回环、重复地标合并或生产级鲁棒性证据。OpenCV求解器、SciPy优化器和现成模型不计作原创；不能宣称新算法、SOTA或泛化性能。','', '阶段6按当前约定验收完成后停止。未启动阶段7成果整理、扩展Benchmark、额外可靠性/融合、训练或UI工作。']
    (DOC/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(DOC/'delivery_source_hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'scripts/accept_stage6.py',ROOT/'scripts/report_stage6_acceptance.py']})
    print('Stage6 acceptance passed; 134 distinct cross-environment tests; stage7 not started.')

if __name__=='__main__':report()
