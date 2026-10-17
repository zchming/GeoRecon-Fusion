"""Publish teaching validation, limitations and cumulative contribution update."""
from pathlib import Path
from collections import Counter
import subprocess,sys,platform
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6';OUT=ROOT/'outputs/stage6_two_view_v1'


def report():
    import cv2
    evaluation=read_json(DOC/'evaluation.json');rows=evaluation['rows'];lock=read_json(DOC/'input_lock.json');validation=read_json(DOC/'validation.json');assert len(validation['rows'])==11
    execution=read_json(DOC/'execution.json');steps=[read_json(OUT/r['case_id']/'steps.json') for r in rows]
    body=sum(r['wall_seconds'] for r in steps);full=sum(r['full_process_wall_seconds_including_sampling'] for r in execution['rows']);peak=max(r['peak_rss_gib'] for r in steps)
    card_used=[];child_listed=False
    for rec in execution['rows']:
        for snap in [rec['gpu_before'],*rec['gpu_samples'],rec['gpu_after']]:
            try:card_used.append(float(snap['card']['stdout'].rsplit(',',1)[-1].strip()))
            except (KeyError,ValueError):pass
            for line in snap.get('compute',{}).get('stdout','').splitlines():
                if line.split(',')[0].strip()==str(rec['pid']):child_listed=True
    costs={'algorithm_process_body_seconds_sum':body,'controller_measurement_windows_seconds_sum':full,'controller_includes_nvidia_sampling_overhead':True,'peak_worker_rss_gib':peak,
        'algorithm_GPU_allocated_bytes':0,'sampled_whole_card_used_mib_min':min(card_used) if card_used else None,'sampled_whole_card_used_mib_max':max(card_used) if card_used else None,
        'teaching_child_seen_in_exposed_compute_listing':child_listed,'WSL_listing_limit':'whole-card desktop usage is not task allocation; per-process listing may be limited; no CUDA calls/model and OpenCL explicitly disabled'}
    write_json(DOC/'cost_accounting.json',costs)
    summary={'scope':'Stage6 calibrated two-view teaching substep','real_pairs':3,'real_successes':sum(r['status']=='success' for r in rows if r['kind']=='real'),
        'controls':8,'control_expected_statuses_met':evaluation['all_expected_statuses_met'],'all_status_counts':dict(Counter(r['status'] for r in rows)),
        'real_matches_and_points':[{k:r[k] for k in ['case_id','status','failure_reason','rotation_error_deg','translation_direction_error_deg']}|{'mutual_matches':r['geometry']['matches'],'kept_points':r['geometry']['kept_points']} for r in rows if r['kind']=='real'],
        'known_intrinsics_are_input':True,'new_downloads_or_models':0,'new_Benchmark_results':False,'PnP_incremental_BA_in_scope':False,'next_stage_started':False}
    write_json(DOC/'summary.json',summary)
    write_json(DOC/'acceptance.json',{'scope':'authorized Stage6 minimal calibrated two-view teaching substep','substep_acceptance':'passed','whole_incremental_SfM_implemented':False,'PnP_incremental_BA':False,'stage7_started':False,'stage5_evidence_reviewed':True,'real_pair_successes':2,'real_pair_failures_preserved':1,'all_8_control_expectations_met':True,'known_K_declared_input':True,'no_pose_or_scale_reference_reads':True,'feature_and_geometry_replay_identical':True,'cross_environment_tests':120,'performance_ranking_or_novelty_claim':False,'GPU_allocated_bytes':0})
    write_json(DOC/'failure_catalog.json',{'rows':[{'case_id':r['case_id'],'kind':r['kind'],'mode':r['input_mode'],'reason':r['failure_reason'],'matches':r['geometry']['matches'],
        'source':str((OUT/r['case_id']).relative_to(ROOT)),'steps_sha256':sha256(OUT/r['case_id']/'steps.json'),'process_returncode':read_json(OUT/r['case_id']/'process.json')['returncode']} for r in rows if r['status']!='success'],
        'controls_not_natural_failures':True,'actual_real_pair_failures':1,'unexpected_process_errors':0,'threshold_or_input_changes_after_results':False})
    paths=[ROOT/'src/georecon/teaching/two_view.py',ROOT/'src/georecon/teaching/run.py',ROOT/'src/georecon/cli.py',ROOT/'scripts/prepare_stage6.py',ROOT/'scripts/run_stage6.py',ROOT/'scripts/evaluate_stage6.py',ROOT/'scripts/check_stage6.py',ROOT/'scripts/report_stage6.py']
    write_json(DOC/'source_lock.json',{'configuration_commit':'9e8e030','config_sha256':sha256(ROOT/'configs/teaching/stage6_two_view.json'),'input_lock_sha256':sha256(DOC/'input_lock.json'),'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in paths},
        'stage5_acceptance_sha256':sha256(ROOT/'docs/stage5_acceptance/acceptance.json'),'algorithm_not_retuned_after_real_run':True})
    write_json(DOC/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'opencv':cv2.__version__,'opencv_OpenCL_used':False,
        'new_packages_weights_models':[],'GPU_budget_gib':12,'GPU_snapshot':execution['rows'][0]['gpu_before'],'opencv_documentation':'official 4.x web API guidance; actual tested runtime version recorded separately'})
    lines=['# 阶段6：最小教学版双视图SfM验证','',
        '**本次授权的双视图教学子步骤已完成。** 实现SIFT匹配、Essential Matrix鲁棒估计、四组相对位姿选解、显式DLT及正深度/视差/重投影验证。没有PnP、增量注册、BA、自标定、稠密重建、新Benchmark或阶段7工作；不把此最小子步骤称作完整增量SfM。','',
        '## 阶段5复核与协议边界','',
        'stage5_audit.json核对了阶段5验收、冻结规则、所有记录的源码/输入/24个消融主产物哈希：工程与证据验收通过，稳定收益未证实。原方法和失败目录保持原状，没有把教学输出接入选择/融合/优化或重新排序模型。','',
        '配置、输入和实现提交 **9e8e030** 后才运行全部真实图像对与控制。使用本地既有DTU scan1图像，复用1600×1200→800×600的Pillow双线性像素中心变换；没有下载新数据。DTU原图来自既有Rectified目录，本教学模块采用已知无畸变K。','',
        '与阶段2/3 COLMAP自估内参协议不同：这里明确允许标定K作为算法输入，转换为Knew=A*K；真实R/t、点、尺度只由独立评价脚本读取。运行输入有严格字段白名单，只含两张图/观测及各自K；进程拒绝读取reference/calibration目录或stage6_truth。它不能被宣传为无标定或绝对无真值信息的方法，也不能与先前不同内参协议的模型直接作性能排名。','',
        '## 真实图像对','',
        '|图像对|SIFT关键点|互检匹配|E内点|保留三角化点|旋转误差°|平移方向误差°|状态/失败|','|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['kind']!='real':continue
        g=r['geometry'];feat=r['features']['keypoints'];rot='—' if r['rotation_error_deg'] is None else f"{r['rotation_error_deg']:.4f}";direction='—' if r['translation_direction_error_deg'] is None else f"{r['translation_direction_error_deg']:.4f}"
        lines.append(f"|{r['case_id']}|{feat[0]}/{feat[1]}|{g['matches']}|{g['essential_inliers']}|{g['kept_points']}|{rot}|{direction}|{r['status']} / {r['failure_reason'] or '—'}|")
    lines+=['',
        '001/020仅4对互检匹配，按冻结20对门限停止，未换图、降低阈值或编造位姿；索引间隔本身不是已测量的重叠率。模糊对在本例方向误差更小，不代表模糊有益或稳健性更好：匹配子集和RANSAC都会变化，只有一个物体的少量教学图像对。','',
        '成功表示内部检查通过，不是GT正确率。已知K条件下的旋转/平移方向误差仍非零；低重投影误差不保证真实几何。两视图的尺度不可恢复，norm(t)=1、首相机I/0；不计算需要≥3非共线相机的ATE，不把误差缺失填零。','',
        '## 合成与退化验证','',
        '|控制|输入类型|结果|含义|','|---|---|---|---|']
    notes={'blank_pair':'关键点/匹配为0','identical_dtu_image':'同一真实图像复制，不构成新真实场景','nonplanar_exact':'无噪声位姿/尺度约定与DLT数学检查','nonplanar_noise_outliers':'0.2像素噪声+32/160随机外点','planar':'H主导，保守拒绝歧义','pure_rotation':'没有可恢复的平移基线','tiny_baseline':'本例H主导；单独数学测试也验证小视差风险','too_few_matches':'7对少于工程门限20，不是五点法的数学最小值'}
    for r in rows:
        if r['kind']=='real':continue
        lines.append(f"|{r['case_id']}|{r['input_mode']}|{r['status']} / {r['failure_reason'] or '—'}|{notes[r['case_id']]}|")
    lines+=['',
        '8个控制的预期状态全部满足。无噪声非平面160点均接受，旋转/方向误差数值上为0，单位基线XYZ均距约8.8e-13；这是人为生成fixture的数学验证，不是真实重建性能。含噪声/外点控制接受128个真实匹配、保留假匹配0，旋转误差0.2735°、方向误差2.3793°，oracle尺度后的中位XYZ误差约0.3354个单位基线，不能省略噪声导致的结构误差。','',
        '平面/纯旋转/极小基线由H竞争门保守拒绝，不意味着H主导必然证明平面或旋转，也不表示所有此类情况只能失败。本模块未做单应分解或更复杂退化处理。','',
        '## 中间过程与可复算性','',
        '核心代码[teaching/two_view.py](../../src/georecon/teaching/two_view.py)展示互检、归一化E、四解计数、DLT和逐点掩码；[运行封装](../../src/georecon/teaching/run.py)负责严格输入、真值隔离和统一输出。[数学与产物 walkthrough](walkthrough.md)解释像素变换、E四解、C2=-Rᵀt、单位尺度及数组索引映射。','',
        '每例outputs/stage6_two_view_v1/<case>/保存prediction/steps/invocation/process、全部几何候选及稀疏点；图像模式另有features_matches.npz与matches.png（青=接受点、绿=E内点未接受、红=其余）。失败时不导出接受点云或已注册相机，但保留调试候选和完整原因。NPZ、图像与本地数据不入Git；逐例小JSON、配置、源码、测试、报告已归档。','',
        'check_stage6.py重新提取图像特征、匹配并重跑几何，与归档数组/步骤一致，逐点验证单位基线、旋转矩阵、双正深度、视差和两幅图误差；核验无位姿参考读取、输入和产物SHA、预期退出码。120项不同测试在可靠性环境（跳过1原生COLMAP）和COLMAP环境（跳过20个依赖OpenCV/SciPy的测试）交叉覆盖，所有跳过项在另一环境实际通过。','',
        '## 资源与限制','',
        f"11例算法进程主体耗时合计{body:.4f}s，最高进程RSS {peak:.4f}GiB；控制器测量窗口合计{full:.4f}s，含nvidia-smi采样开销，不当作纯算法速度。CPU线程8，OpenCL关闭，无CUDA/模型调用，新增GPU分配0。整卡采样使用{costs['sampled_whole_card_used_mib_min']}–{costs['sampled_whole_card_used_mib_max']}MiB（含桌面），未在暴露的compute列表发现教学子进程；WSL列表可能受限，不将整卡占用当作任务显存。详见cost_accounting.json和process.json。满足12GB约束，无新模型、权重或包安装。",
        '',
        '已知局限：K必须给定且畸变已处理；只有两视图，尺度任意；RANSAC/互检并不消除所有歧义；H门可能过度拒绝；没有PnP、增量轨迹管理、BA、回环、稠密结构或大场景验证；真实示例仅同一DTU物体的少量图像对，合成对应测试没有验证图像特征检测性能。','',
        '## 截至目前的贡献','',
        '累计已有统一坐标/失败/资源协议和可审计实验管线，推理几何诊断、开发集冻结的选择/过滤/融合/联合优化及必要消融，并保留负结果。本步增加可解释的已知K双视图教学链、四解/DLT中间产物与退化验证，支持理解和展示几何基础。','',
        'SIFT、E分解、RANSAC、DLT、COLMAP/VGGT及现成数值库不是本项目原创；工程集成、验证工具与有界经验发现不能直接等同于算法新颖性或SOTA。阶段5稳定收益仍未证实，教学成功也不改变这一结论。[累计贡献与边界](../contributions.md)。','',
        '## 运行与停止边界','',
        '```bash','export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8','PY=/home/zchming/.venvs/georecon-reliability/bin/python',
        '# 依赖本地已有prepared input；输出必须是新目录。',
        '$PY -m georecon.cli run-teaching-sfm --input data/stage6_pairs/dtu_scan1_001_003/input.json --config configs/teaching/stage6_two_view.json --output outputs/stage6_reproduction',
        '# success退出0；普通无法重建退出1；输入/协议错误退出2。',
        '$PY scripts/evaluate_stage6.py',
        '$PY scripts/check_stage6.py',
        '$PY -m unittest discover -s tests -p test_teaching_sfm.py -v','```','',
        '初始输入由prepare_stage6.py在本地既有DTU数据上一次性构建，输入锁存在时拒绝隐式重建；每个准备输入仅含K和图像/观测，真实位姿参考在单独data/stage6_truth。准备/评价访问参考不等于运行进程访问参考，输入假设均已明确。','',
        '本次仅完成阶段6最小双视图子步骤，停止于此；不自动实现完整增量SfM，不进入阶段7成果整理或UI。','']
    (DOC/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':report()
