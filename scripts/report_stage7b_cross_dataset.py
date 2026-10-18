"""Independent provenance checks, scene-level contrasts and non-tuned retention decision."""
import argparse
from pathlib import Path
import hashlib
import json
import csv
from itertools import combinations
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def report(study,prior,output):
    from georecon.matching.features import validate_pairs
    from georecon.geometry.camera import Camera,relative_pose
    from georecon.matching.metrics import pose_metrics
    from georecon.data.calibrated_benchmarks import strecha_camera,middlebury_cameras
    study=Path(study).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    e=read(study/'evaluation.json');b=read(study/'inference.json');cfg=b['config'];session=read(study/'session.json');prov=read(study/'provenance.json')
    expected=len(b['cases'])*len(cfg['frontends'])*len(cfg['geometry_seeds'])*len(cfg['strategies'])
    assert expected==720 and e['complete'] and session['complete'] and len(e['rows'])==expected
    assert len({(r['case'],r['arm'],r['seed'],r['strategy']) for r in e['rows']})==expected
    assert all(r['status']=='complete' for r in e['rows'])
    for p,d in prov['source_sha256'].items():assert sha(ROOT/p)==sha(study/'sources'/p)==d
    for p,d in cfg['algorithm_locks'].items():assert sha(ROOT/p)==d
    assert read(study/'protocol.json')==cfg
    for p,d in read(study/'input_hashes.json').items():assert sha(study/p)==d
    assert session['gt_guard'] and not any(any(w in p.lower() for w in ['truth.json','reference.json','evaluation.json','/raw/','.camera']) for p in session['file_accesses'])
    def load(p):
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    checked=0
    for key,path in b['pair_paths'].items():
        arm,a,c=key.split(':');x,y=load(b['feature_paths'][a]),load(b['feature_paths'][c]);m=load(path)
        validate_pairs(m['indices'],len(x['keypoints']),len(y['keypoints']))
        np.testing.assert_array_equal(m['pixels_first'],x['keypoints'][m['indices'][:,0]])
        np.testing.assert_array_equal(m['pixels_second'],y['keypoints'][m['indices'][:,1]]);checked+=1
    truth=read(study/'truth.json');checks=0;reference_checks=0
    import urllib.parse
    for item in read(study/'preparation.json')['downloads']:
        if 'raw.githubusercontent.com/' in item['url']:
            parts=urllib.parse.unquote(item['url']).split('/');name=parts[-1];scene=parts[parts.index('Benchmarking_Camera_Calibration_2008')+1]
            raw=study/'raw'/('strecha_'+scene)/name
        else:
            dataset=next(d for d in cfg['data'] if d.get('url')==item['url']);raw=study/'raw'/('middlebury_'+dataset['scene'])/'source.zip'
        assert sha(raw)==item['sha256']
    for scene,camera_table in truth.items():
        if scene.startswith('middlebury_'):parsed=middlebury_cameras((study/'raw'/scene/'parameters.txt').read_text())
        for name,camera in camera_table.items():
            imported=strecha_camera((study/'raw'/scene/(name+'.camera')).read_text())[0] if scene.startswith('strecha_') else parsed[name][0]
            for k in ['K','R','t']:np.testing.assert_array_equal(imported.as_dict()[k],camera[k])
            reference_checks+=1
    for row in e['rows']:
        case=next(c for c in b['cases'] if c['id']==row['case']);record=read(study/'predictions'/row['case']/row['arm']/str(row['seed'])/row['strategy']/'record.json')
        gt=[Camera.from_dict(truth[case['scene']][v]) for v in case['view_ids']];poses={int(v):(np.array(p['R']),np.array(p['t'])) for v,p in record['poses'].items()}
        values=[]
        for i,j in combinations(range(len(gt)),2):
            if i in poses and j in poses:
                r=poses[j][0]@poses[i][0].T;t=poses[j][1]-r@poses[i][1]
            else:r,t=None,None
            value=pose_metrics(gt[i],gt[j],r,t,r is not None);values.append(value)
            stored=next(v for v in row['pairs'] if v['pair']==[i,j])
            assert all(stored[k]==v for k,v in value.items());assert value['GT_pose_defined'];checks+=1
        assert len(values)==row['planned_pairs'];assert row['accurate_scene']==all(v['correct'] for v in values)
        assert abs(row['auc10']-np.mean([v['auc']['10'] for v in values]))<1e-14
        assert row['false_pairs']==sum(v['false_accept'] for v in values)
    scene_rows=[];seed_rows=[]
    for arm in cfg['frontends']:
        for scene in sorted({r['scene'] for r in e['rows']}):
            rows={s:[r for r in e['rows'] if r['arm']==arm and r['scene']==scene and r['strategy']==s] for s in ['inliers','quality']}
            a,c=rows['inliers'],rows['quality'];family=a[0]['family']
            scene_rows.append({'family':family,'scene':scene,'arm':arm,
                'inliers_auc10':float(np.mean([r['auc10'] for r in a])),'quality_auc10':float(np.mean([r['auc10'] for r in c])),
                'delta_auc10':float(np.mean([r['auc10'] for r in c])-np.mean([r['auc10'] for r in a])),
                **{s+'_'+k:sum(r[k] for r in values) for s,values in rows.items() for k in ['accurate_scene','false_pairs','severe_pairs']}})
        for family in ['strecha','middlebury']:
            for seed in cfg['geometry_seeds']:
                values={s:[r for r in e['rows'] if r['arm']==arm and r['family']==family and r['seed']==seed and r['strategy']==s] for s in ['inliers','quality']}
                seed_rows.append({'arm':arm,'family':family,'seed':seed,'quality_minus_inliers_score10':float(np.mean([r['auc10'] for r in values['quality']])-np.mean([r['auc10'] for r in values['inliers']]))})
    old=read(Path(prior)/'evaluation.json');old_quality=[r for r in old['rows'] if r['arm']=='C' and r['strategy']=='quality'];old_inliers=[r for r in old['rows'] if r['arm']=='C' and r['strategy']=='inliers']
    old_severe={'inliers':sum(r['severe_pairs'] for r in old_inliers),'quality':sum(r['severe_pairs'] for r in old_quality)}
    assert old_severe['quality']<=old_severe['inliers'],'do not promote without disclosing prior harm'
    checks_report={'complete_runs':expected,'pair_metric_recomputations':checks,'matching_identity_checks':checked,'reference_cameras_reparsed':reference_checks,'all_download_hashes_verified':True,'all_source_input_hashes_verified':True,'no_pose_truth_read_in_prediction':True,'no_unknown_GT_baselines':True,'prior_DTU_severe_counts':old_severe,'decision':e['decision'],'scene_contrasts':scene_rows,'seed_contrasts':seed_rows}
    write(output/'checks_and_contrasts.json',checks_report)
    with (output/'scene_contrasts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(scene_rows[0]));w.writeheader();w.writerows(scene_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.9));colors={'fixed':'#777777','inliers':'#2775ad','quality':'#ce6a29'}
    for strategy in cfg['strategies']:
        values=[p['joint_deg'] for r in e['rows'] if r['arm']=='C' and r['strategy']==strategy for p in r['pairs'] if p['joint_deg'] is not None];values.sort()
        n=sum(r['planned_pairs'] for r in e['rows'] if r['arm']=='C' and r['strategy']==strategy)
        if values:axes[0].plot(values,np.arange(1,len(values)+1)/n,label=strategy,color=colors[strategy])
    axes[0].set(xlim=(0,30),ylim=(0,1),xlabel='Joint pose error (degrees)',ylabel='Fraction of ALL planned pairs',title='External data: LightGlue');axes[0].legend()
    for i,family in enumerate(['strecha','middlebury']):
        for j,strategy in enumerate(cfg['strategies']):
            row=next(s for s in e['summaries'] if s['family']==family and s['arm']=='C' and s['strategy']==strategy)
            axes[1].bar(i+(j-1)*.23,row['accurate_scene'],width=.22,color=colors[strategy],label=strategy if i==0 else None)
    axes[1].set(xticks=[0,1],xticklabels=['Strecha (4 sequences)','Middlebury (2 objects)'],ylim=(0,1),ylabel='Accurate complete map rate',title='All views registered, all pairs <=5 deg')
    rr=[s for s in scene_rows if s['arm']=='C'];axes[2].barh(range(len(rr)),[s['delta_auc10'] for s in rr],color=['#2775ad' if s['delta_auc10']>=0 else '#ce6a29' for s in rr]);axes[2].axvline(0,color='gray',linewidth=.7)
    axes[2].set(yticks=range(len(rr)),yticklabels=[s['scene'].replace('middlebury_','').replace('strecha_','') for s in rr],xlabel='Quality minus inliers: score@10',title='Each scene, 4 conditions x 5 seeds')
    fig.tight_layout();fig.savefig(output/'cross_dataset.svg');fig.savefig(output/'cross_dataset.png',dpi=140);plt.close(fig)
    print(json.dumps({'decision':e['decision'],'source_and_pairs_verified':checks,'scene_contrasts':scene_rows,'seed_contrasts':seed_rows},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.study,a.prior,a.output)
