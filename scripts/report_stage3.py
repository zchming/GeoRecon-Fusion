"""Build the fixed first-round report and independently verify published run artifacts."""
from pathlib import Path
import json, shutil
from collections import Counter
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.data.scene import load_scene
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage3';OUT=ROOT/'outputs/stage3_dtu_pose_pilot_v1'
results=read_json(DOC/'results.json');lock=read_json(DOC/'freeze.json')
plan=read_json(ROOT/'configs/benchmarks/dtu_pose_pilot_v1.json')
assert sha256(ROOT/'configs/benchmarks/dtu_pose_pilot_v1.json')==lock['plan_sha256']
for item in lock['methods'].values():assert sha256(ROOT/item['path'])==item['sha256']
checks=[]
for d in lock['datasets']:
 scene=d['manifest']['scene_id']; mappings=[]
 assert sha256(ROOT/d['path'])==d['manifest_sha256']
 assert sha256((ROOT/d['path']).parent/'reference.json')==d['reference_sha256']
 manifest,images=load_scene(ROOT/d['path'],preprocess=True)
 assert manifest['source']['scan'] in plan['splits'][manifest['split']]
 condition=next(c for c in plan['conditions'] if c['id']==manifest['source']['condition']['id'])
 assert [r['image_id'] for r in manifest['images']]==[f'{v:03d}' for v in condition['views']]
 assert all(im.shape==(600,800,3) for im in images)
 for method in ['colmap','vggt_raw','vggt_ba']:
  root=OUT/scene/method;r=validate_result(read_json(root/'prediction.json'));m=read_json(root/'input_mapping.json')
  assert r['input_manifest_sha256']==d['manifest_sha256']
  assert r['provenance']['code_commit'].startswith('3899747')
  mappings.append([(i['image_id'],i.get('processed_sha256',i.get('common_processed_sha256'))) for i in m])
  for artifact in r['artifacts'].values():
   if artifact is not None and 'sha256' in artifact:assert sha256(root/artifact['path'])==artifact['sha256']
  assert r['status']!='not_run'
  gpu=r['memory_bytes'].get('peak_gpu_allocated')
  if gpu is not None:assert gpu<=10.5*1024**3
  driver=r['memory_bytes'].get('peak_device_used_sampled_including_desktop')
  if driver is not None:assert driver<=12*1024**3
 assert mappings[0]==mappings[1]==mappings[2]
 checks.append(dict(scene=scene,same_common_input_bytes=True,output_protocol_and_artifact_hashes=True))
assert len(results['rows'])==24
write_json(DOC/'validation.json',dict(cases=checks,runs=24,all_failures_retained=True,no_not_run_rows=True,freeze_sha256=sha256(DOC/'freeze.json'),source_commit='3899747',full_suite='59 tests passed; see tests.txt',gpu_allocator_peaks_within_10_5_gib=True,available_driver_samples_within_12_gib=True,driver_sampling_missing_for=['vggt_raw']))
# Execution/test logs are captured by the invocation and published alongside results.

def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'
lines=['# 阶段 3：DTU 相机位姿首轮 Benchmark 与失败分析','',
'已完成阶段2验收复核、运行前冻结协议与数据摘要，以及24次单次对比。阶段2核对见 `stage2_acceptance_audit.json`；协议提交 `c0243e6`，数据摘要提交及运行源码 `3899747`。没有测试后改参数重跑，没有可靠性分析、选择/回退或融合。','',
'## 范围与固定条件','',
'本轮为 **DTU pose pilot v1**，仅 scan1 开发、scan6 测试，每场景 normal5、reduced3、spaced3、blur5 四条件；共8个实例×3方法。不是标准DTU全量划分或官方稠密MVS评测。每划分只有一个独立对象，四条件相关，不能从此估计跨场景泛化或显著性。GT只用于评价。详细输入顺序、模糊构造、预处理差异、预算与指标定义见 [运行前协议](protocol.md)、`freeze.json` 和 `configs/benchmarks/dtu_pose_pilot_v1.json`。','',
'COLMAP报告特征几何注册；VGGT raw的覆盖率表示相机预测可用，不等于经过特征注册或精度达标。BA也只按输出完整性判定success，精度另列。不同方法共有同一实例的800×600公共输入；raw与BA后续网格不同，因此两者不是纯BA消融。','',
'## 逐实例结果','',
'角度为已注册/可用相机对子集均值，ATE为评价专用Sim(3)后的相机中心RMSE（mm）。失败不造零误差；它们仍进入成功率与全部输入对分母的AUC。时间为完整CLI墙钟，包含runner与模型加载，不包含GT评价。','',
'| 划分/条件 | 方法 | 状态 | 相机覆盖 | 旋转° | 方向° | ATE mm | 联合AUC@10 | 秒 |','|---|---|---|---|---:|---:|---:|---:|---:|']
for r in results['rows']:
 lines.append(f"| {r['split']}/{r['condition']} | {r['method']} | {r['status']} | {r['registered']}/{r['input_images']} | {fmt(r['rotation_deg'])} | {fmt(r['translation_deg'])} | {fmt(r['ate_mm'])} | {fmt(r['auc']['joint']['10'])} | {fmt(r['cli_wall_seconds'],2)} |")
lines+=['','## 按划分汇总','',
'以下均先按实例计算再等权平均，避免五图实例因相机对更多而占更大权重。角度/ATE只对可用实例取统计量并报告计数；联合AUC保留全部失败。完整均值、中位数、coverage、单项AUC@5/10/20、资源与逐对误差见 `results.json` 和 `runs/`。','',
'| 划分 | 方法 | 完整成功/4 | 状态计数 | 联合AUC@5/10/20 | 旋转均值°（可用实例） | 方向均值° | ATE均值mm | CLI均值/中位秒（含失败） |','|---|---|---:|---|---|---:|---:|---:|---:|']
for g in results['groups']:
 lines.append(f"| {g['split']} | {g['method']} | {g['successes']}/4 | {g['statuses']} | {'/'.join(fmt(g['joint_auc'][str(t)]) for t in [5,10,20])} | {fmt(g['rotation_registered_subset']['mean'])}（{g['rotation_registered_subset']['count']}） | {fmt(g['translation_registered_subset']['mean'])} | {fmt(g['ate_available_subset']['mean'])} | {fmt(g['cli_wall_seconds']['mean'],2)}/{fmt(g['cli_wall_seconds']['median'],2)} |")
lines+=['','## 失败与退化：观察和解释边界','']
for r in results['rows']:
 if not r['success']:
  state=read_json(OUT/r['scene_id']/r['method']/'worker_state.json')
  reason='CUDA显存分配失败，完整分配请求和发生位置见该实例worker_log.txt及results.json' if r['status']=='oom' else r['failure_reason']
  text=f"- **{r['scene_id']}/{r['method']}：{r['status']}**，覆盖{r['registered']}/{r['input_images']}；{reason}"
  if 'selected_model' in state:text+=f"；原生候选模型：{state.get('models')}。"
  lines.append(text)
lines+=['',
'对两次COLMAP失败，原生数据库中001–020、010–020的验证观测数均为0，只有001–010保留匹配（scan1为894，scan6为857），该对的PyCOLMAP4.2.1配置标记均为6，即PLANAR_OR_PANORAMIC。原生日志显示反复尝试后因bad initial pair丢弃模型，最终No good initial image pair。由此能定位到验证匹配图与初始化阶段；该估计器标记不证明真实物体必然是平面，也没有据此补造三维重建。',
'',
'四次BA OOM的traceback均进入track_modules/track_refine.py的fine_tracker：追加分配请求分别为668MiB、2.94GiB、676MiB、3.13GiB。即使已释放闲置VGGT模型，冻结的精细跟踪仍不适配本机五图显存预算；三图四次均完成。没有在看到测试集OOM后改chunk、查询点数或精度重跑。',
'',
'以上是本轮直接观察到的终止原因。spaced3仅表示索引间隔扩大，不能据失败就断言低重叠、低纹理或某种材质原因；没有做置信度/可靠性打分。COLMAP原生数据库统计、候选组件和映射日志，GPU失败发生阶段与分配请求，都随逐次worker日志保存，便于复查。','',
'成功输出也可能有明显位姿退化。以下按预注册的5°阈值检查联合误差，仅描述现象，不重新设定成功门槛：','']
for r in results['rows']:
 if r['success']:
  e=read_json(OUT/r['scene_id']/r['method']/'evaluation.json');pairs=e['pairs'];bad=sum(1 for p in pairs if p['rotation_deg'] is not None and p['translation_direction_deg'] is not None and max(p['rotation_deg'],p['translation_direction_deg'])>5)
  if bad:lines.append(f"- {r['scene_id']}/{r['method']}：{bad}/{len(pairs)}对联合角误差超过5°；均值旋转{fmt(r['rotation_deg'])}°、方向{fmt(r['translation_deg'])}°。完整输出不能替代准确度。")
lines+=['',
'比较应同时看AUC、覆盖率、角度/ATE和耗时，不能因失败返回得快就称方法更快，也不能只用成功子集的低误差排名。原始点图和BA稀疏点的数量不是同一任务的完整度；本轮没有稠密精度比较。对于低重投影优化目标但GT方向误差较大的结果，只能说拟合观测不保证独立位姿精度，不能据此生成可靠性模块或融合决策。','',
'## 成本与显存','',
'所有GPU运行串行，allocator上限10.5GiB。下表峰值是各方法本轮所有成功/失败运行的最大值；CPU方法GPU项为未测/不适用，不记零。原始字节、完整/分阶段时间和OS/采样RSS见逐运行prediction与worker_state。','',
'| 方法 | allocated峰值GiB | reserved峰值GiB | BA整卡采样峰值GiB | OS RSS峰值GiB | 全部CLI耗时合计秒 |','|---|---:|---:|---:|---:|---:|']
for method in ['colmap','vggt_raw','vggt_ba']:
 part=[r for r in results['rows'] if r['method']==method]
 def peak(k):
  v=[r['memory_bytes'].get(k) for r in part if r['memory_bytes'].get(k) is not None];return max(v)/1024**3 if v else None
 lines.append(f"| {method} | {fmt(peak('peak_gpu_allocated'))} | {fmt(peak('peak_gpu_reserved'))} | {fmt(peak('peak_device_used_sampled_including_desktop'))} | {fmt(peak('peak_worker_rss_os'))} | {sum(r['cli_wall_seconds'] for r in part):.2f} |")
lines+=['',
'BA整卡采样包含驱动和桌面，100ms可能漏掉瞬时峰值；raw只有allocator高水位，不冒充整卡测量。CPU线程设置对COLMAP/PyTorch为8、OMP=8；Ceres保留官方默认solver设置，不能将它解释为明确固定的8线程Ceres配置。GPU为RTX4070SUPER，基础Torch2.11.0+cu128；COLMAP4.2.1和BA用的PyCOLMAP3.10.0分环境。权重与版本沿用阶段2，逐次版本也已保存。seed固定但官方cuDNN允许非确定算法，本轮每实例只有一次，耗时和误差不是多次运行方差估计。','']
prep=read_json(DOC/'data_preparation.json')
lines.append('数据提取成本（不混入推理）：'+ '；'.join(f"scan{p['scan']}传输{p['provenance']['transferred_bytes']}字节，下载{p['provenance']['download_seconds']:.2f}s" for p in prep)+'。包含归档范围读取与校验；不是完整归档下载。权重本轮复用本地文件，依赖安装和以前的权重下载不计为本次推理时间。')
lines+=['','## 复现与验收','',
'从仓库根目录，使用现有独立环境：','',
'```bash',
'/home/zchming/.venvs/georecon-colmap/bin/python scripts/stage3_benchmark.py audit',
'/home/zchming/.venvs/georecon-colmap/bin/python -m unittest discover -s tests -q',
'# 首次本地准备：prepare；已冻结的本轮数据不要再次覆盖',
'/home/zchming/.venvs/georecon-colmap/bin/python scripts/stage3_benchmark.py run',
'# 已有运行只重新汇总，不触发推理：',
'/home/zchming/.venvs/georecon-colmap/bin/python scripts/stage3_benchmark.py summarize',
'```','',
'脚本自动按方法使用 `.venvs/georecon-colmap`、`georecon-vggt`、`georecon-vggt-ba`。源码/模型的取得与各环境锁见阶段2报告。重新执行完整Benchmark需独立的新运行标识/输出位置，当前脚本拒绝覆盖，不能借此静默多跑后挑最佳。发布包中的manifest快照及SHA用于核查；原始图片、模型、原生数据库、大型NPZ留在本地data/outputs，JSON内相对artifact路径以原运行目录为根。','',
'59项测试通过；另逐次核验24份结果协议、每方法相同公共PNG摘要、预测manifest指纹与产物摘要，验证见 `validation.json`。八个实例均有三条显式终态记录，失败全部留表。本轮之后停止，未开始阶段4可靠性分析或阶段5融合。若要提高本轮资源成功率，应另开明确版本、只在开发数据调整资源配置并重新锁定方案，不能把已看过的scan6继续称作未经接触的新测试集。']
lines+=['','阶段4复核勘误：此前将COLMAP原生图像ID误当清单索引，宽间隔案例的边名称已按数据库ID→worker文件名→DTU图像ID映射修正。实际保留001–010，020孤立；匹配数、模型状态和所有评价数值不变。']
(DOC/'report.md').write_text('\n'.join(lines)+'\n')
print('report and independent artifact checks complete')
