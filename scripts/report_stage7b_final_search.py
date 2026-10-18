"""Check final-search provenance, retain tradeoffs and write concise real metrics."""
import argparse
from collections import Counter
from itertools import combinations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def report(args):
    from georecon.geometry.camera import Camera,relative_pose
    from georecon.matching.metrics import pose_metrics
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    selection=read(args.development/'selection.json');selected=selection['selected'];rows=[];summaries=[];sessions=[];verified=0;original_baselines=0;baseline_differences=[];source_bundles={}
    for study in args.study:
        e=read(study/'evaluation.json');f=read(study/'freeze.json');session=read(study/'session.json')
        if not e['complete'] or not session['complete']:raise ValueError('incomplete matrix')
        protocol=read(study/'protocol.json')
        if sha(ROOT/'configs/experiments'/(protocol['id']+'.json'))!=f['config_sha256']:raise ValueError('frozen configuration drift')
        if not session['gt_guard'] or any(any(w in n.lower() for w in ['truth.json','reference.json','evaluation.json','/raw/','_par.txt','.camera','groundtruth','cameras.txt','images.txt']) for n in session['file_accesses']):raise ValueError('reference read guard invalid')
        if f['selected_candidate_lock']!=selection:raise ValueError('external candidate selection differs')
        if f['unix']<=selection['unix']:raise ValueError('external run began before development lock')
        for n,d in f['source_sha256'].items():
            if sha(study/'sources'/n)!=d or sha(ROOT/n)!=d:raise ValueError('source provenance drift')
        for n,d in read(study/'input_hashes.json').items():
            if sha(n)!=d:raise ValueError('input hash drift')
        for r in e['rows']:
            rec=read(r['record']);source=rec['source']
            if source not in source_bundles:source_bundles[source]=(read(Path(source)/'inference.json'),read(Path(source)/'truth.json'))
            b,truth=source_bundles[source];c=r['case'];case=rec['case'];tk=str(case['scan']) if case['family']=='dtu' else case['scene'];gt=[Camera.from_dict(truth[tk][v]) if truth[tk][v] is not None else None for v in case['view_ids']]
            cameras={int(v):Camera(np.array(b['images'][case['keys'][int(v)]]['K']),np.array(p['R']),np.array(p['t']),b['images'][case['keys'][int(v)]]['width'],b['images'][case['keys'][int(v)]]['height']) for v,p in rec['poses'].items()}
            for pair,saved in zip(combinations(range(len(gt)),2),r['pairs']):
                i,j=pair
                if gt[i] is None or gt[j] is None:
                    if not saved['unavailable']:raise ValueError('unknown reference hidden')
                    continue
                rr,tt=relative_pose(cameras[i],cameras[j]) if i in cameras and j in cameras else (None,None);actual=pose_metrics(gt[i],gt[j],rr,tt,rr is not None)
                for metric in ['rotation_deg','translation_deg','joint_deg']:
                    a,z=actual[metric],saved[metric]
                    if (a is None)!=(z is None) or (a is not None and abs(a-z)>1e-8):raise ValueError('pose metric mismatch')
                verified+=1
            if r['method']=='legacy' and r['ranking'] in ['fixed','inliers','quality']:
                old=Path(source)/'predictions'/c/r['arm']/str(r['seed'])/r['ranking']/'record.json'
                if old.exists():
                    orig=read(old);original_baselines+=1
                    if orig['before_poses']!=rec['before_poses'] or orig['selection']['selected_seed']!=rec['selection']['selected_seed']:raise ValueError('legacy reconstruction before BA changed')
                    # Budget-limited BA can differ under shared machine load; retain any difference.
                    if orig['poses']!=rec['poses']:baseline_differences.append({'record':r['record'],'old':str(old),'reason':'same inputs/solver, final BA comparison requires scrutiny'})
        rows+=e['rows'];summaries+=e['summaries'];sessions.append({'study':str(study),**session})
    from georecon.data.motion_benchmarks import colmap_references,tum_reference
    from georecon.geometry.camera import Camera
    reference_reparse_count=0
    for source,(bundle,truth) in source_bundles.items():
        source=Path(source)
        if not (source/'metadata_transfer.json').exists():continue
        parsed=colmap_references((source/'cameras.txt').read_text(),(source/'images.txt').read_text())
        for scene,table in truth.items():
            if scene.startswith('tum_rgbd_'):
                folder=source/'raw'/scene;selected_frames=read(folder/'selected.json')
                for name,entry in table.items():
                    actual,gap=tum_reference((folder/'groundtruth.txt').read_text(),float(Path(name).stem),bundle['config']['data']['tum']['K'])
                    if (actual is None)!=(entry is None):raise ValueError('missing TUM reference hidden')
                    if actual is None:continue
                    if abs(gap-selected_frames['reference_gaps_seconds'][name])>1e-12:raise ValueError('TUM association drift')
                    original=Camera.from_dict(entry)
                    np.testing.assert_allclose(actual.R,original.R,atol=1e-12,rtol=0);np.testing.assert_allclose(actual.t,original.t,atol=1e-12,rtol=0)
                    reference_reparse_count+=1
            elif scene=='south_building_south_building':
                for name,entry in table.items():
                    original=Camera.from_dict(entry);actual=parsed[name][0]
                    np.testing.assert_allclose(actual.R,original.R,atol=1e-12,rtol=0);np.testing.assert_allclose(actual.t,original.t,atol=1e-12,rtol=0)
                    reference_reparse_count+=1
    families=sorted({r['family'] for r in rows})
    if families!=['dtu','middlebury','south_building','strecha','tum_rgbd']:raise ValueError('five distinct planned dataset families required')
    if len(rows)!=5760:raise ValueError('fixed external denominator5760 maps required')
    look={(s['family'],s['arm'],s['ranking'],s['method']):s for s in summaries};gates={};tradeoffs={}
    for family in families:
        old=look[family,'C','inliers','legacy'];new=look[family,'C','inliers',selected]
        family_rows=[r for r in rows if r['family']==family and r['arm']=='C' and r['ranking']=='inliers'];lookup={(r['case'],r['seed'],r['method']):r for r in family_rows}
        deltas=[lookup[r['case'],r['seed'],selected]['auc10']-r['auc10'] for r in family_rows if r['method']=='legacy']
        seed_deltas={str(seed):float(np.mean([lookup[r['case'],r['seed'],selected]['auc10']-r['auc10'] for r in family_rows if r['method']=='legacy' and r['seed']==seed])) for seed in range(42,47)}
        scene_deltas={scene:float(np.mean([lookup[r['case'],r['seed'],selected]['auc10']-r['auc10'] for r in family_rows if r['method']=='legacy' and r['scene']==scene])) for scene in {r['scene'] for r in family_rows}}
        severe_scene_ok=all(sum(r['severe_pairs'] for r in family_rows if r['scene']==scene and r['method']==selected)<=sum(r['severe_pairs'] for r in family_rows if r['scene']==scene and r['method']=='legacy') for scene in scene_deltas)
        gates[family]={'score_nonnegative':new['auc10']>=old['auc10']-1e-12,'accurate_nonnegative':new['accurate_scene']>=old['accurate_scene']-1e-12,'false_nonincrease':new['false_pairs']<=old['false_pairs'],'severe_nonincrease':new['severe_pairs']<=old['severe_pairs'],'scene_severe_nonincrease':severe_scene_ok,'available':new['unavailable_pairs']==0,'strict_score_improvement':new['auc10']>old['auc10']+1e-12}
        tradeoffs[family]={'score10_delta':new['auc10']-old['auc10'],'accurate_map_delta':new['accurate_scene']-old['accurate_scene'],'false_pair_delta':new['false_pairs']-old['false_pairs'],'severe_pair_delta':new['severe_pairs']-old['severe_pairs'],'registered_delta':new['registered']-old['registered'],'seconds_delta':new['seconds']-old['seconds'],'scene_score_deltas':scene_deltas,'seed_score_deltas':seed_deltas,'paired_better':sum(d>1e-10 for d in deltas),'paired_worse':sum(d< -1e-10 for d in deltas)}
    material=float(np.mean([d['score10_delta'] for d in tradeoffs.values()]))>=.01 or (sum(look[f,'C','inliers',selected]['false_pairs'] for f in families)<=.9*sum(look[f,'C','inliers','legacy']['false_pairs'] for f in families) and all(g['score_nonnegative'] for g in gates.values()))
    stable=all(all(v for k,v in g.items() if k!='strict_score_improvement') for g in gates.values()) and all(gates[f]['strict_score_improvement'] for f in ['tum_rgbd','south_building']) and material
    # Safety against fixed reference is a separate, mandatory promotion check.
    safe_fixed=all(look[f,'C','inliers',selected]['severe_pairs']<=look[f,'C','fixed','legacy']['severe_pairs'] and look[f,'C','inliers',selected]['false_pairs']<=look[f,'C','fixed','legacy']['false_pairs'] for f in families)
    exploratory=[]
    for ranking in ['inliers','quality']:
        for method in ['lm','ap3p','ap3p_lm']:
            contrasts={}
            for family in families:
                a,z=look[family,'C',ranking,'legacy'],look[family,'C',ranking,method]
                contrasts[family]={'score10_delta':z['auc10']-a['auc10'],'accurate_map_delta':z['accurate_scene']-a['accurate_scene'],'false_delta':z['false_pairs']-a['false_pairs'],'severe_delta':z['severe_pairs']-a['severe_pairs']}
            exploratory.append({'ranking':ranking,'method':method,'family_contrasts':contrasts,'equal_family_score10_delta':float(np.mean([c['score10_delta'] for c in contrasts.values()])), 'confirmatory_selected':ranking=='inliers' and method==selected})
    observed_best=max((c for c in exploratory if c['ranking']=='inliers'),key=lambda c:c['equal_family_score10_delta'])
    failure_rows=[];index={(r['case'],r['arm'],r['seed'],r['ranking'],r['method']):r for r in rows}
    for r in rows:
        if r['arm']!='C' or r['ranking']!='inliers' or r['method']!='legacy':continue
        n=index[r['case'],r['arm'],r['seed'],r['ranking'],selected];delta=n['auc10']-r['auc10']
        if abs(delta)>1e-10 or n['severe_pairs']!=r['severe_pairs'] or n['false_pairs']!=r['false_pairs']:
            failure_rows.append({'case':r['case'],'family':r['family'],'seed':r['seed'],'score10_before':r['auc10'],'score10_after':n['auc10'],'delta':delta,'false_before':r['false_pairs'],'false_after':n['false_pairs'],'severe_before':r['severe_pairs'],'severe_after':n['severe_pairs'],'registered_before':r['registered'],'registered_after':n['registered'],'baseline_record':r['record'],'candidate_record':n['record']})
    observed_pairs=[]
    for r in rows:
        if r['arm']!='C' or r['ranking']!='inliers' or r['method']!='legacy':continue
        n=index[r['case'],r['arm'],r['seed'],r['ranking'],observed_best['method']]
        observed_pairs.append({'case':r['case'],'seed':r['seed'],'delta':n['auc10']-r['auc10'],'base_score':r['auc10'],'new_score':n['auc10'],
                               **{k+'_'+label:row[k+'_pairs'] for k in ['false','severe'] for label,row in [('before',r),('after',n)]},
                               'registered_before':r['registered'],'registered_after':n['registered'],
                               'baseline_record':str(Path(r['record']).relative_to(ROOT)), 'candidate_record':str(Path(n['record']).relative_to(ROOT))})
    decision={'selected_candidate':selected,'development_winner':selection['development_winner'],'stable_benefit_gate':stable,'material_gain_gate':material,'safe_against_fixed':safe_fixed,'promotion':stable and safe_fixed,'status':'GO_BOUNDED_EVIDENCE' if stable and safe_fixed else 'NO_GO_STABLE_PROMOTION','retain':['inliers','quality'],'gates':gates,'tradeoffs':tradeoffs,'all_candidate_contrasts':exploratory,'observed_best_exploratory':{**observed_best,'independent_confirmation':False,'default_changed':False,'selection_basis':'post-hoc highest observed family-macro gain among3 predeclared variants under inlier ranking; candidate for review only'},'qualifier':'Small fixed subsets only. South Building SfM poses are proxy references, not independent truth. Seeds/conditions correlated. No novelty or universal generalization established.'}
    write(out/'checks_and_decision.json',{'report_source_sha256':sha(Path(__file__)),'decision':decision,'verified_pose_slots':verified,'new_reference_cameras_reparsed':reference_reparse_count,'legacy_before_BA_equal_replays':original_baselines,'BA_output_differences':baseline_differences,'runs':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),'BA_statuses':dict(Counter(r['BA'] for r in rows)),'sessions':[{'study':s['study'],'runs':s['runs'],'seconds':s['seconds'],'peak_rss_bytes':s['peak_rss_bytes']} for s in sessions]})
    for name,rs in [('summaries.csv',summaries),('paired_changes.csv',failure_rows),('observed_candidate_pairs.csv',observed_pairs)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(15,8),layout='constrained');labels=['DTU','Middlebury','South Building*','Strecha','TUM RGB-D'];x=np.arange(5)
    for ax,metric,title in zip(axes.flat,['auc10','false_pairs','severe_pairs','seconds'],['Joint pose linear score@10 (higher better)','False accepted pairs >5° (lower better)','Severe pairs >20° (lower better)','Seconds/map incl shared seed checks']):
        for offset,(ranking,method,label) in enumerate([('inliers','legacy','inliers/EPnP'),('quality','legacy','quality/EPnP'),('inliers',selected,f'inliers/{selected.upper()} [locked]'),('inliers',observed_best['method'],f"inliers/{observed_best['method'].upper()} [exploratory]"),('quality',observed_best['method'],f"quality/{observed_best['method'].upper()} [exploratory]")]):
            ax.bar(x+(offset-2)*.16,[look[f,'C',ranking,method][metric] for f in families],width=.16,label=label)
        ax.set_xticks(x,labels,rotation=15);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Final bounded search · LightGlue C · *SfM reference, not independent GT')
    fig.savefig(out/'final_search.svg');fig.savefig(out/'final_search.png',dpi=160)
    print(json.dumps({'decision':decision,'runs':len(rows),'verified_pose_slots':verified,'legacy_replays':original_baselines,'BA_differences':len(baseline_differences)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,action='append',required=True);p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);report(p.parse_args())
