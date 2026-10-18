"""Read-only Stage7A input, solver-artifact and metric verification; no geometry solving."""
import argparse
import json,hashlib,pathlib,numpy as np
from georecon.matching.evaluate import truth_for_case
from georecon.matching.metrics import pose_metrics
from georecon.geometry.camera import Camera
parser=argparse.ArgumentParser(description=__doc__)
for key in ['cache','stability','refinement','source','weights','config','output']:parser.add_argument('--'+key,type=pathlib.Path,required=True)
args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=args.output
if (out/'stage7a_closeout.json').exists():raise ValueError('preserve earlier audit')
out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(root/'configs/experiments/stage7a_protocol_v1.json')
run=args.stability;bundle=read(run/'bundle.json');ev=read(run/'evaluation.json');print(ev.keys())
lookup={(c['case_id'],c['scan']):c for c in bundle['cases']}
count=0;diff=[]
for r in ev['rows']:
 if r['arm'] not in ['C_old','C_magsac','C_msac']:continue
 c=lookup[r['case_id'],r['scan']];folder=run/'fits'/r['case_id']/r['arm'];record=next(v for v in read(folder/'record.json')['rows'] if v['seed']==r['seed']);path=folder/f"seed_{r['seed']}.npz"
 assert sha(path)==record['arrays_sha256']
 with np.load(path) as z:
  p=pose_metrics(*truth_for_case(root,protocol,c),z['accepted_R'] if record['status']=='success' else None,z['accepted_t'] if record['status']=='success' else None,record['status']=='success')
 if p!=r['pose']:diff.append([r['case_id'],r['arm'],r['seed']])
 count+=1
ref=args.refinement;e=read(ref/'evaluation.json');print('reproduction',e['reproduction'])
assert e['reproduction']['passed'] and not e['reproduction']['mismatches']
changes=[]
for c in bundle['cases']:
 for seed in range(42,52):
  a=next(r for r in e['rows'] if r['case_id']==c['case_id'] and r['seed']==seed and r['arm']=='base')
  b=next(r for r in e['rows'] if r['case_id']==c['case_id'] and r['seed']==seed and r['arm']=='noise1')
  if a['pose']['correct']!=b['pose']['correct']:changes.append({'case':c['case_id'],'seed':seed,'before':a['pose']['joint_deg'],'after':b['pose']['joint_deg'],'rescue':b['pose']['correct']})
severe={}
for a in ['base','noise1']:
 f=ref/'fits/scan6_blur_002_005'/a;record=next(r for r in read(f/'record.json')['rows'] if r['seed']==51)
 with np.load(f/'seed_51.npz') as z:severe[a]={'statistics':record['statistics'],'E':z['E'].tolist(),'singular_values':np.linalg.svd(z['E'])[1].tolist(),'essential_mask':z['essential_mask'].tolist()}
cache=args.cache;original=read(cache/'bundle.json');locks={}
for c in bundle['cases']:
 with np.load(run/'inputs'/f"{c['case_id']}.npz",allow_pickle=False) as inputs:
  for arm in ['B','C']:
   with np.load(cache/'main'/c['case_id']/arm/'matches.npz',allow_pickle=False) as matches:
    for name in ['pixels_first','pixels_second']:np.testing.assert_array_equal(inputs[f'{arm}_{name}'],matches[name])
  original_case=next(v for v in original['cases'] if v['case_id']==c['case_id'])
  for k,key in zip(c['K'],original_case['image_keys']):np.testing.assert_array_equal(k,original['images'][key]['K'])
for p in [cache/n for n in ['bundle.json','input_hashes.json','implementation.json']]+list((cache/'features').glob('*/features.npz'))+list((cache/'features').glob('*/record.json'))+[cache/'main'/c['case_id']/a/n for c in bundle['cases'] for a in ['B','C'] for n in ['matches.npz','record.json']]:locks[p.relative_to(cache).as_posix()]=sha(p)
aggregate=hashlib.sha256(json.dumps(locks,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert aggregate=='4c593e8db86bf65559624c9b65023d7219b9e3e4c47b2f87f72b104e74b729e7'
for p,d in read(cache/'input_hashes.json').items():assert sha(root/p)==d
lg=protocol['lightglue'];vendor=args.source
for p,d in lg['source_sha256'].items():assert sha(vendor/p)==d
assert sha(args.weights)==lg['weights_sha256']
assert not diff
result={'checked_stability_pose_slots':count,'metric_mismatches':diff,'fixed_cache_files':len(locks),'fixed_cache_sha256':aggregate,'all_original_input_hashes_verified':True,'lightglue_source_weights_verified':True,'refinement_base_reproduction':e['reproduction'],'paired_threshold_transitions':changes,'severe_case':severe,'audit':'read cached arrays; no new geometric solve','new_config_sha256':sha(args.config)}
(out/'stage7a_closeout.json').write_text(json.dumps(result,indent=2));print('verified',count,'transitions',len(changes),'rescues',sum(r['rescue'] for r in changes))
