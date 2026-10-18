"""Verify all planned runs, exact historical geometry, source/input hashes and GT isolation."""
import argparse
import json,pathlib,numpy as np,hashlib,collections
parser=argparse.ArgumentParser(description=__doc__)
for key in ['study','baseline','cache','report']:parser.add_argument('--'+key,type=pathlib.Path,required=True)
args=parser.parse_args()
if args.report.exists():raise ValueError('preserve previous check report')
out=args.study;old=args.baseline;origin=json.load(open(args.cache/'bundle.json'));bundle=json.load(open(out/'inference.json'));e=json.load(open(out/'evaluation.json'))
lookup={tuple(c['image_keys']):c['case_id'] for c in origin['cases']};count=0
for c in bundle['cases']:
 if c['split']!='development':continue
 for arm in ['B','C']:
  for seed in [42,43,44]:
   rec=json.load(open(out/'predictions'/c['id']/arm/str(seed)/'fixed/record.json'))
   for s in rec['signals']:
    i,j=s['pair'];historical=lookup[(c['keys'][i],c['keys'][j])]
    folder=old/'fits'/historical/(arm+'_msac');r=next(v for v in json.load(open(folder/'record.json'))['rows'] if v['seed']==seed)
    assert r['status']==s['status'] and r['reason']==s['reason']
    with np.load(folder/f'seed_{seed}.npz') as a,np.load(out/'predictions'/c['id']/arm/str(seed)/'fixed'/f'candidate_{i}_{j}.npz') as b:
     assert set(a.files)==set(b.files),(a.files,b.files)
     for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    count+=1
assert len(e['rows'])==480
assert all(r['status']=='complete' for r in e['rows'])
assert len({(r['case'],r['arm'],r['seed'],r['strategy']) for r in e['rows']})==480
for r in e['rows']:
 assert len(r['pairs'])==r['planned_pairs']
 assert r['accurate_scene']==all(p['correct'] for p in r['pairs'])
 assert r['registered']>=r['correct_cameras_anchor']>=r['correct_cameras_all_pairs']
 assert abs(r['auc10']-np.mean([p['auc']['10'] for p in r['pairs']]))<1e-14
 assert r['false_pairs']==sum(p['false_accept'] for p in r['pairs'])
 assert r['attempts']<=3
session=json.load(open(out/'prediction_session.json'));assert session['truth_guard']
assert not any(any(w in p.lower() for w in ['truth.json','reference.json','evaluation.json','/calibration/']) for p in session['accesses'])
provenance=json.load(open(out/'prediction_provenance.json'));root=pathlib.Path.cwd()
maintenance=[]
for p,sha in provenance['source_sha256'].items():
 assert hashlib.sha256((out/'sources'/p).read_bytes()).hexdigest()==sha
 if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:
  # Exact, explicit post-run export-path repair; prediction code is unchanged.
  assert p=='scripts/run_stage7b_initialization.py'
  repaired=(out/'sources'/p).read_text().replace("path=Path('docs/results/stage7b_initialization.csv')", "path=out/'summaries.csv'")
  assert repaired==(root/p).read_text()
  maintenance.append({'file':p,'reason':'evaluation CSV now stays inside explicit output; archived inference unchanged','current_sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()})
for p,sha in json.load(open(out/'input_hashes.json')).items():assert hashlib.sha256((out/p).read_bytes()).hexdigest()==sha
print('COMPLETE:480 maps; historical E/H/pose/triangulation exact',count,'shared candidate checks; GT guard and all input/source hashes valid')
result={'all_480_maps_complete_no_process_errors':True,'unique_planned_runs':480,'all_primary_rows':360,'ablation_rows':120,'historical_geometry_arrays_exact':count,'all_input_source_hashes_verified':True,'post_run_export_maintenance':maintenance,'GT_guard_verified':True,'all_planned_pairs_denominator_verified':True,'BA_states':dict(collections.Counter(json.load(open(p))['BA']['reason'] for p in (out/'predictions').glob('*/*/*/*/record.json')))}
args.report.write_text(json.dumps(result,indent=2))
