"""Publish bounded offline point evaluation with failure cases, no rule updates."""
import sys,subprocess,platform
from pathlib import Path
import numpy as np
import scipy
from georecon.data.protocol import read_json,write_json,sha256
from stage5_points import Evaluator,ROOT,DOC,OUT,CONDITIONS,PROTOCOL,RULE


def report():
    comparison=read_json(DOC/'comparison.json');failures=[]
    for scan in [1,6]:
        ev=Evaluator(scan)
        for cond in CONDITIONS:
            scene=f'dtu_scan{scan}_{cond}';ev.prepare(scene)
            dest=OUT/'frozen'/scene
            with np.load(dest/'filtered.npz') as a: keep=np.isin(ev.ids,a['source_point_indices'])
            with np.load(dest/'evidence.npz') as a: unknown=(a['finite']&(a['observed']==0))[ev.ids]
            valid=ev.in_mask;distance=np.full(len(valid),np.inf)
            distance[valid]=ev.tree.query(ev.xyz[valid],workers=8)[0]
            good=distance<=2;bad=~good
            failures.append({'scene_id':scene,'scope':'same fixed source sample, official prediction ROI, 2mm distance to sampled GT',
                'in_mask':int(valid.sum()),'good_points_before':int((valid&good).sum()),'good_points_removed':int((valid&good&~keep).sum()),
                'bad_points_before':int((valid&bad).sum()),'bad_points_kept':int((valid&bad&keep).sum()),
                'unknown_in_mask':int((valid&unknown).sum()),'good_unknown_removed':int((valid&unknown&good).sum())})
    write_json(DOC/'failure_analysis.json',{'rows':failures,'classification_is_posthoc_GT_only':True,'rule_not_changed':True})
    lock={'protocol_sha256':sha256(PROTOCOL),'rule_sha256':sha256(RULE),
          'frozen_before_test_commit':'f54255c','stage3_freeze_sha256':sha256(ROOT/'docs/stage3/freeze.json'),
          'source_files':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'scripts/stage5_points.py',ROOT/'scripts/stage5_point_data.py',ROOT/'src/georecon/filtering/points.py']},
          'official_evaluation_source':{'url':'https://roboimagedata.compute.dtu.dk/?page_id=36','archive':'https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip','files_sha256':{p.name:sha256(p) for p in sorted((ROOT/'data/stage5_point_gt/code').glob('*.m'))}},'gt':{}}
    for scan in [1,6]:
        lock['gt'][str(scan)]=read_json(ROOT/f'data/stage5_point_gt/scan{scan}/provenance.json')
    write_json(DOC/'input_lock.json',lock)
    rows=comparison['rows'];summary={}
    for split in ['dev','test']:
        subset=[r for r in rows if r['split']==split]
        summary[split]={side:{metric:float(np.mean([r[side]['thresholds_mm']['2'][metric] for r in subset])) for metric in ['precision','recall','f1']} for side in ['before','after']}
    elapsed=sum(r['run']['wall_seconds'] for r in rows)
    summary['incremental_filter_seconds_sum']=elapsed
    summary['max_filter_process_rss_gib']=max(r['run']['peak_rss_gib'] for r in rows)
    summary['gpu_allocated_bytes']=0
    summary['default_enable_recommended']=False
    summary['reason']='dev-selected best filtering candidate is worse than the no-filter baseline; test is one scene with four dependent conditions'
    write_json(DOC/'summary.json',summary)
    write_json(DOC/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
        'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv'],capture_output=True,text=True).stdout,
        'filter_device':'CPU','gpu_budget_gib':12,'torch_model_loaded':False,'new_packages_or_weights':[]})
    lines=['# 阶段 5：最小可靠性点过滤验证','',
        '已完成场景选择复核、开发集规则锁定、固定 DTU 划分的点过滤与几何评价。当前结果不支持默认启用过滤；它是显式调用的实验工具，原场景选择流程保持原输出。未进行加权融合、点移动、跨点平均或联合优化。','',
        '## 前置复核与规则','',
        '8 个自然样本都选择 VGGT raw；自然回退为 0。重新校验了阶段 5 输入锁、诊断文件、相机、点文件 SHA，以及发布的决策。此前 10 个开发集故障探针和 GT 隔离探针仍保留在 docs/stage5；其中模糊场景回退到 COLMAP 后质量差，内部一致性不等于正确。旧选择规则仅用开发集重新计算并复现，未改动。','',
        '协议先提交于 `47841d4`，规则与开发集报告于 `f54255c` 锁定，之后才下载 scan6 参考几何、运行冻结过滤和测试评价。扫描划分、图像、视图数、分辨率、模糊条件和所有上游算法设置沿用阶段 3。测试姿态在前序阶段已经被查看，本次是固定样本上的回顾性点几何试验，不是新盲测。','',
        '候选为相对 Z 阈值 {0.01, 0.025, 0.05} × 至少 {1, 2} 个其他视图支持。按开发集四个条件的 F1@2mm 宏平均选取，平分按候选顺序。最优过滤候选为 **0.05 / 1 个其他视图**。不过滤基线单独比较，没有根据测试集改阈值。完整 6 个候选结果及文件访问审计见 fit_report.json。','',
        '直接将原始 `world_points` 点头 XYZ 投影至其他相机，用目标深度四邻域双线性采样比较 `|z-d|/max(z,d)`。不使用源视图给自己投票，不以深度反投影点替换点头，不使用真值、网络置信度阈值或 GT 对齐参与过滤。原生网格为 392×518，整数像素中心。保留点坐标逐元素不变，C-order 原始索引可还原 frame/y/x。没有支持的点被排除；无有效观察单独记 unknown，不能称为已知错误。遮挡未建模，相关网络输出可共同出错。','',
        '## 固定评价协议','',
        '参考来自 [DTU 官方 SampleSet](https://roboimagedata.compute.dtu.dk/?page_id=36) 内 scan1/scan6 的 STL、ObsMask*_10 和 Plane；按 HTTP Range 分别设置 96 MiB 上限，校验 archive size、ETag、ZIP CRC 和本地 SHA。GT 与官方 MATLAB 代码仅在 data/ 保存，溯源见 input_lock.json。无新权重、依赖安装或推理。','',
        '仅在离线评价时，以预测/标定相机中心拟合 Sim3 到毫米坐标；前后复用同一变换，不做 ICP 或点云真值配准。每个源点云按原始索引等距取最多 200000 点，过滤后沿用同一组样本索引，不补采样。STL 等距取最多 200000 点；精度以官方 ObsMask 筛选预测点，完整度使用 Plane 上方参考点。参考→预测查询使用全部有限预测样本；预测→参考查询使用整个参考样本。报告未经距离截断的均值及 1/2/5mm P/R/F1，空输出计 F1=0、距离不可计算则 null。','',
        '这是一项预算受限的自定义点评价：未执行官方 0.2mm 邻域降采样、60mm 距离上限或20mm离群剔除，参考亦被采样，且只使用 3/5 张图覆盖局部表面。不可与官方 DTU 全数据排名直接比较。ROI/地平面规则对照官方 PointCompareMain.m，代码哈希已记录。','',
        '## 过滤前后（2mm，百分比）','',
        '|场景|保留率|P 前→后|R 前→后|F1 前→后|','|---|---:|---:|---:|---:|']
    for r in rows:
        a,b=r['before']['thresholds_mm']['2'],r['after']['thresholds_mm']['2']
        lines.append(f"|{r['scene_id']}|{100*r['run']['rows'][0]['retention']:.2f}%|{100*a['precision']:.2f}→{100*b['precision']:.2f}|{100*a['recall']:.2f}→{100*b['recall']:.2f}|{100*a['f1']:.2f}→{100*b['f1']:.2f}|")
    lines+=['']
    for split in ['dev','test']:
        a,b=summary[split]['before'],summary[split]['after']
        lines.append(f"{split} 宏平均 F1@2mm：{100*a['f1']:.3f}% → {100*b['f1']:.3f}%；P：{100*a['precision']:.3f}% → {100*b['precision']:.3f}%；R：{100*a['recall']:.3f}% → {100*b['recall']:.3f}%。")
        lines.append('')
    lines+=['## 失败与限制','',
        '开发集四个条件的 F1 均下降；reduced3 降幅最大。支持不足会删除实际正确点，尤其视野边缘、遮挡和少视图区域。spaced3 的精度上升伴随覆盖下降。测试 normal5 轻微退化，其余变化很小，不能视作稳定提升：四个条件来自同一 scan6，且采样误差、配准误差和点头原有几何误差共同影响结果。failure_analysis.json 列出了 GT 后验确认的正确点被删除、错误点被保留及 unknown 中正确点数量。','',
        '规则是内部一致性筛选，不是正确性概率估计；相机正确也不能保证点头正确。当前不默认启用，不扩大候选、不用测试失败重新调参。COLMAP/BA 稀疏输出缺少兼容原生稠密深度网格，当前明确不支持这条过滤器；全拒绝/无有效观察由测试覆盖，不生成假点。','',
        '实际失败已保留：一次过滤完成后元数据序列化错误（outputs/stage5_points_v1/failures/attempt01）；一次 COLMAP 环境回归测试因缺 SciPy 失败，日志保留，之后仅该离线评价测试在此环境跳过，在可靠性环境执行。详见 failures.json。','',
        '## 资源、产物与验证','',
        f"8 次冻结过滤耗时合计 {elapsed:.4f}s（加载、分块投影、压缩和落盘，不含历史模型推理、下载、离线 GT 评价和进程启动），单进程峰值 RSS 最高 {summary['max_filter_process_rss_gib']:.4f} GiB。固定 CPU 线程 8、点批次65536；此轻量步骤没有加载 Torch/CUDA，新增 GPU 分配 0，满足 12GB 预算。nvidia-smi 的整卡占用包含桌面/其他进程，见 environment.json；不是本步骤显存。每个 run.json 保留实测耗时与RSS。",
        '',
        '原始 before 点文件仍由阶段 3/5 路径和 SHA 引用；after 为 outputs/stage5_points_v1/frozen/<scene>/filtered.npz，包含 xyz/source_point_indices；evidence.npz 保留 finite/observed/support，可重建 keep/reject/unknown。统一 prediction.json 的相机不变，pointcloud 指向新子集，depth/confidence 保留原始产物。开发集六候选文件和失败目录均保留于 outputs。大型产物、GT 和上游权重不入 Git，Git 保存规则、代码、小型逐场景 JSON 与报告。','',
        '验证覆盖：点头与深度头不一致、不准自投票、边界/unknown/非有限/全拒绝、遮挡深度冲突、分块一致性、固定样本子集覆盖率单调性、空输出评价和 GT 读取拒绝。共87项不同测试分环境交叉覆盖：可靠性环境87项/跳过1项原生COLMAP，COLMAP环境87项/跳过2项OpenCV与1项SciPy评价测试。所有跳过项均在另一环境实际通过。全量回归日志见 regression_*.txt；validation.json 记录全部8场景逐点子集、字节哈希和相机不变校验。此子步骤完成后停止，不进入融合或联合优化。','',
        '## 复现（可靠性 Python 环境）','',
        '```bash',
        '# 已有本地阶段3/5产物；输出已存在时命令拒绝覆盖。',
        'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8',
        'PY=/home/zchming/.venvs/georecon-reliability/bin/python',
        '$PY scripts/check_stage5_points.py',
        '# 开发阶段先下载 scan1 并生成4条件的 --dev-candidates；fit 只运行一次，已冻结则拒绝覆盖。',
        '$PY scripts/stage5_points.py generate --scene dtu_scan6_reduced3 --output-root outputs/stage5_points_reproduction',
        '# 原始冻结输出上的评价与核验：',
        '$PY scripts/stage5_points.py evaluate',
        '$PY scripts/check_stage5_points.py --final',
        '$PY -m unittest discover -s tests -v',
        '```','']
    (DOC/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':report()
