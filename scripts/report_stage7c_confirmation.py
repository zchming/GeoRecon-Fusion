"""Audit frozen held-out confirmation, including unavailable slots; no new solving."""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from georecon.evaluation.confirmation import confirmation_gate
from georecon.geometry.camera import Camera,relative_pose
from georecon.matching.metrics import pose_metrics

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def export(p,rows):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def report(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);study=a.study.resolve();data=a.data.resolve();cfg=read(a.config)
    freeze=read(study/'freeze.json');data_freeze=read(data/'freeze.json');session=read(study/'session.json');e=read(study/'evaluation.json');bundle=read(data/'inference.json');truth=read(data/'truth.json');planned=read(data/'planned_cases.json')
    if sha(a.config)!=freeze['config_sha256'] or sha(a.config)!=data_freeze['config_sha256']:raise ValueError('configuration drift')
    if freeze['selected_candidate_lock']!=cfg or data_freeze['unix']>=freeze['unix']:raise ValueError('candidate freeze/order mismatch')
    for n,h in freeze['source_sha256'].items():
        if sha(ROOT/n)!=h or sha(study/'sources'/n)!=h:raise ValueError('frozen source drift')
    for n,h in read(study/'input_hashes.json').items():
        if sha(n)!=h:raise ValueError('reconstruction input drift')
    if not session['gt_guard'] or any(any(x in n.lower() for x in ['truth.json','reference.json','evaluation.json','/raw/','groundtruth','cameras.txt','images.txt']) for n in session['file_accesses']):raise ValueError('GT isolation failed')
    rows=e['rows'];lookup={(r['case'],r['arm'],r['seed'],r['ranking'],r['method']):r for r in rows};missing=[];slots=0;seed_checks=0
    if len(lookup)!=len(rows):raise ValueError('duplicate outcomes')
    for case in planned:
        for arm in cfg['frontends']:
            for seed in cfg['geometry_seeds']:
                for ranking in cfg['rankings']:
                    for method in cfg['pnp_candidates']:
                        key=case['id'],arm,seed,ranking,method
                        if key not in lookup:missing.append({'case':case['id'],'family':case['family'],'arm':arm,'seed':seed,'ranking':ranking,'method':method,'status':'not_run_data_unavailable','planned_pairs':case['image_count']*(case['image_count']-1)//2})
    if len(rows)+len(missing)!=cfg['planned_runs']:raise ValueError('unexpected matrix slots')
    actual_runs=len(rows)
    by_case={c['id']:c for c in planned}
    for item in missing:
        case=by_case[item['case']]
        row={**item,'scene':case['scene'],'condition':case['condition'],'registered':0,'accurate_scene':False,'correct_cameras':0,'correct_init':False,
             'accepted_pairs':0,'false_pairs':0,'severe_pairs':0,'unavailable_pairs':item['planned_pairs'],'auc5':0.,'auc10':0.,'auc20':0.,'seconds':0.,'BA':'not_run','pairs':[]}
        rows.append(row);lookup[row['case'],row['arm'],row['seed'],row['ranking'],row['method']]=row
    for r in rows:
        if r['status']=='not_run_data_unavailable':continue
        p=study/'predictions'/r['case']/r['arm']/str(r['seed'])/(r['ranking']+'_'+r['method'])/'record.json';rec=read(p);case=rec['case'];ks=[np.array(bundle['images'][k]['K']) for k in case['keys']];tk=str(case['scan']) if case['family']=='dtu' else case['scene']
        gt=[Camera.from_dict(truth[tk][v]) if truth[tk][v] is not None else None for v in case['view_ids']]
        cams={int(v):Camera(ks[int(v)],np.array(p['R']),np.array(p['t']),bundle['images'][case['keys'][int(v)]]['width'],bundle['images'][case['keys'][int(v)]]['height']) for v,p in rec['poses'].items()}
        for (i,j),saved in zip(combinations(range(len(gt)),2),r['pairs']):
            slots+=1
            if gt[i] is None or gt[j] is None:
                if not saved['unavailable']:raise ValueError('missing reference hidden')
                continue
            rr,tt=relative_pose(cams[i],cams[j]) if i in cams and j in cams else (None,None);actual=pose_metrics(gt[i],gt[j],rr,tt,rr is not None)
            for k in ['rotation_deg','translation_deg','joint_deg']:
                x,y=actual[k],saved[k]
                if (x is None)!=(y is None) or (x is not None and abs(x-y)>1e-8):raise ValueError('pose metric mismatch')
        if r['method']=='ap3p':
            other=read(study/'predictions'/r['case']/r['arm']/str(r['seed'])/(r['ranking']+'_legacy')/'record.json')
            if rec['selection']['selected_seed']!=other['selection']['selected_seed']:raise ValueError('PnP change selected different two-view seed')
            for v in rec['selection']['selected_seed'] or []:
                if rec['before_poses'][str(v)]!=other['before_poses'][str(v)]:raise ValueError('seed pose changed between PnP arms')
            seed_checks+=1
    summaries=[]
    for family in ['dtu','tum_rgbd']:
        for arm in cfg['frontends']:
            for ranking in cfg['rankings']:
                for method in cfg['pnp_candidates']:
                    rs=[r for r in rows if (r['family'],r['arm'],r['ranking'],r['method'])==(family,arm,ranking,method)]
                    angles=[p['joint_deg'] for r in rs for p in r['pairs'] if p['joint_deg'] is not None]
                    summaries.append({'family':family,'arm':arm,'ranking':ranking,'method':method,'runs':len(rs),'actual_runs':sum(r['status']!='not_run_data_unavailable' for r in rs),'scenes':len({r['scene'] for r in rs}),
                                      **{k:float(np.mean([r[k] for r in rs])) for k in ['auc5','auc10','auc20','accurate_scene','correct_cameras','registered','correct_init','seconds']},
                                      **{k:sum(r[k] for r in rs) for k in ['planned_pairs','unavailable_pairs','accepted_pairs','false_pairs','severe_pairs']},'p95_deg':float(np.percentile(angles,95)) if angles else None,'max_deg':max(angles) if angles else None})
    look={(s['family'],s['arm'],s['ranking'],s['method']):s for s in summaries};contrasts=[];gates={};changes=[]
    complete=bool(e['complete'] and session['complete'] and not missing and len(rows)==cfg['planned_runs'] and all(r['status']=='complete' and r['unavailable_pairs']==0 and r['BA']!='wall_budget_exceeded' for r in rows))
    for arm in cfg['frontends']:
        for ranking in cfg['rankings']:
            for family in ['dtu','tum_rgbd']:
                before,after=look[family,arm,ranking,'legacy'],look[family,arm,ranking,'ap3p'];base=[r for r in rows if r['family']==family and r['arm']==arm and r['ranking']==ranking and r['method']=='legacy']
                pairs=[(b,lookup[b['case'],arm,b['seed'],ranking,'ap3p']) for b in base]
                scene_d={s:float(np.mean([n['auc10']-b['auc10'] for b,n in pairs if b['scene']==s])) for s in {b['scene'] for b in base}}
                severe_d={s:sum(n['severe_pairs']-b['severe_pairs'] for b,n in pairs if b['scene']==s) for s in scene_d}
                seed_d={str(seed):float(np.mean([n['auc10']-b['auc10'] for b,n in pairs if b['seed']==seed])) for seed in cfg['geometry_seeds']}
                gate=confirmation_gate(before,after,scene_score_deltas=scene_d,scene_severe_deltas=severe_d,seed_score_deltas=seed_d,complete=complete,replication=family=='tum_rgbd')
                gates[family+':'+arm+':'+ranking]={**gate,'scene_score_deltas':scene_d,'scene_severe_deltas':severe_d,'seed_score_deltas':seed_d}
                contrasts.append({'family':family,'arm':arm,'ranking':ranking,'score10_before':before['auc10'],'score10_after':after['auc10'],'score10_delta':after['auc10']-before['auc10'],'accurate_before':before['accurate_scene'],'accurate_after':after['accurate_scene'],'false_before':before['false_pairs'],'false_after':after['false_pairs'],'severe_before':before['severe_pairs'],'severe_after':after['severe_pairs'],'registered_before':before['registered'],'registered_after':after['registered'],'passed':gate['passed']})
                for b,n in pairs:changes.append({'case':b['case'],'family':family,'arm':arm,'ranking':ranking,'seed':b['seed'],'score10_before':b['auc10'],'score10_after':n['auc10'],'delta':n['auc10']-b['auc10'],'false_before':b['false_pairs'],'false_after':n['false_pairs'],'severe_before':b['severe_pairs'],'severe_after':n['severe_pairs'],'registered_before':b['registered'],'registered_after':n['registered']})
    primary=all(gates[f+':C:'+ranking]['passed'] for f in ['dtu','tum_rgbd'] for ranking in cfg['rankings']);tum=all(gates['tum_rgbd:C:'+r]['passed'] for r in cfg['rankings']);secondary=all(c['severe_after']<=c['severe_before'] for c in contrasts if c['arm']=='B')
    decision={'status':'BOUNDED_CONFIRMATION' if primary else 'CONDITIONAL_TUM_ONLY' if tum else 'NO_GO_CONFIRMATION','technical_complete':complete,'primary_C_all_gates':primary,'TUM_C_both_rankings_replication':tum,'B_severe_nonincrease':secondary,'default_changed':False,'retained_rankings':cfg['rankings'],'selected_candidate':'ap3p','new_geographic_independence_proven':False,'planned_runs':cfg['planned_runs'],'actual_runs':actual_runs,'unrun_slots':len(missing),'verified_pose_slots':slots,'shared_seed_checks':seed_checks,'gates':gates,'source_sha256':sha(Path(__file__))}
    write(out/'checks_and_decision.json',decision);write(out/'unrun.json',missing);export(out/'summaries.csv',summaries);export(out/'contrasts.csv',contrasts);export(out/'paired_changes.csv',changes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');primary_rows=[c for c in contrasts if c['arm']=='C'];labels=[c['family']+'\n'+c['ranking'] for c in primary_rows];x=np.arange(len(labels))
    for ax,k,title in zip(axes,['score10','false','severe'],['Joint pose linear score@10','False accepted pair-seed slots >5°','Severe pair-seed slots >20°']):
        ax.bar(x-.18,[r[k+'_before'] for r in primary_rows],width=.36,label='EPnP');ax.bar(x+.18,[r[k+'_after'] for r in primary_rows],width=.36,label='AP3P');ax.set_xticks(x,labels,rotation=12);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Stage7C held-out objects/recordings · fixed LightGlue · correlated repetitions');fig.savefig(out/'confirmation.svg');fig.savefig(out/'confirmation.png',dpi=150)
    print(json.dumps({'decision':{k:v for k,v in decision.items() if k!='gates'},'contrasts':contrasts},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['study','data','config','output']:p.add_argument('--'+key,type=Path,required=True)
    report(p.parse_args())
