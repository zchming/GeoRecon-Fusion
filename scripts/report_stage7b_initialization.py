"""Offline real-data figures and COLMAP comparison; never changes predictions."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import csv
import numpy as np
from georecon.geometry.camera import Camera,relative_pose
from georecon.matching.metrics import pose_metrics


def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def report(study,colmap,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    bundle=read(study/'inference.json');ev=read(study/'evaluation.json');truth=read(study/'truth.json');cfg=bundle['config'];refs=[]
    for c in bundle['cases']:
        path=colmap/c['id']/'run/prediction.json'
        pred=read(path) if path.exists() else {'images':[],'status':'not_run','timing_seconds':{},'memory_bytes':{}}
        cams={v['image_id']:Camera.from_dict(v['camera']) for v in pred['images'] if v['registered']};metrics=[]
        for i,j in combinations(c['view_ids'],2):
            r,t=relative_pose(cams[i],cams[j]) if i in cams and j in cams else (None,None)
            metrics.append(pose_metrics(Camera.from_dict(truth[str(c['scan'])][i]),Camera.from_dict(truth[str(c['scan'])][j]),r,t,r is not None))
        angles=[v['joint_deg'] for v in metrics if v['joint_deg'] is not None]
        refs.append({'case':c['id'],'scan':c['scan'],'split':c['split'],'status':pred['status'],'registered':len(cams),'accurate_scene':all(v['correct'] for v in metrics),
            **{f'auc{t}':float(np.mean([v['auc'][str(t)] for v in metrics])) for t in [5,10,20]},
            'false_pairs':sum(v['false_accept'] for v in metrics),'severe_pairs':sum(v['joint_deg'] is not None and v['joint_deg']>20 for v in metrics),
            'false_rate':float(np.mean([v['false_accept'] for v in metrics])),'severe_rate':float(np.mean([v['joint_deg'] is not None and v['joint_deg']>20 for v in metrics])),
            'worst_deg':max(angles) if angles else None,'seconds':pred['timing_seconds'].get('total'),'peak_rss_bytes':pred['memory_bytes'].get('peak_worker_rss_os'),
            'correct_init':None,'protocol':'self-estimated shared K, official default frontend; separate reference, not controlled strategy comparator'})
    reference_path=study/'colmap_evaluation.json'
    if reference_path.exists():
        if read(reference_path)!=refs:raise ValueError('existing COLMAP evaluation differs')
    else:reference_path.write_text(json.dumps(refs,indent=2))
    with (output/'colmap_reference.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(refs[0]));w.writeheader();w.writerows(refs)
    # Full pose distribution: all accepted pairs, with missing pairs explicitly in adjacent rates.
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for strategy in cfg['strategies']:
        rows=[r for r in ev['rows'] if r['split']=='independent' and r['arm']=='C' and r['strategy']==strategy]
        vals=sorted(p['joint_deg'] for r in rows for p in r['pairs'] if p['joint_deg'] is not None)
        planned=sum(r['planned_pairs'] for r in rows)
        if vals:axes[0].plot(vals,np.arange(1,len(vals)+1)/planned,label=strategy)
    axes[0].set(xlim=(0,30),ylim=(0,1),xlabel='Joint pose error (degrees)',ylabel='Fraction of ALL planned pairs',title='Independent objects: failure-aware CDF');axes[0].legend()
    summaries=[s for s in ev['summaries'] if s['split']=='independent' and s['arm']=='C' and s['strategy'] in cfg['strategies']]
    x=np.arange(len(summaries));axes[1].bar(x,[s['accurate_scene'] for s in summaries],label='Accurate complete map');axes[1].set(xticks=x,xticklabels=[s['strategy'] for s in summaries],ylim=(0,1),title='Independent reconstruction success')
    for idx,s in enumerate(summaries):axes[1].text(idx,s['accurate_scene']+.02,f"{s['accurate_scene']:.1%}",ha='center')
    for strategy in cfg['strategies']:
        xs=[];ys=[]
        for c in bundle['cases']:
            if c['split']!='independent':continue
            rr=[r for r in ev['rows'] if r['case']==c['id'] and r['arm']=='C' and r['strategy']==strategy]
            xs.append(c['id'].replace('scan',''));ys.append(np.mean([r['auc10'] for r in rr]))
        axes[2].plot(range(len(xs)),ys,'o-',label=strategy)
    axes[2].set(xticks=range(len(xs)),xticklabels=xs,ylabel='Linear pose score @10',title='All independent conditions');axes[2].tick_params(axis='x',rotation=65)
    fig.tight_layout();fig.savefig(output/'independent_results.svg');fig.savefig(output/'independent_results.png',dpi=140);plt.close(fig)
    # Select representative by predeclared largest absolute C-minus-B score change; keep benefit and harm.
    representative=[]
    for sign in [1,-1]:
        contrasts=[]
        for c in bundle['cases']:
            for seed in cfg['geometry_seeds']:
                rr={r['strategy']:r for r in ev['rows'] if r['case']==c['id'] and r['arm']=='C' and r['seed']==seed}
                contrasts.append((sign*(rr['quality']['auc10']-rr['inliers']['auc10']),c,seed))
        _,c,seed=max(contrasts,key=lambda r:r[0]);representative.append((c,seed,'benefit' if sign==1 else 'harm'))
    for case,seed,label in representative:
        rec=read(study/'predictions'/case['id']/'C'/str(seed)/'quality/record.json');signals=rec['signals'];n=len(case['keys']);theta=np.linspace(0,2*np.pi,n,endpoint=False);pos=np.c_[np.cos(theta),np.sin(theta)]
        fig=plt.figure(figsize=(12,3.8));ax=fig.add_subplot(131)
        for s in signals:
            i,j=s['pair'];ax.plot(pos[[i,j],0],pos[[i,j],1],color='green' if s['status']=='success' else 'red',alpha=.45,linewidth=1+min(s['essential_inliers']/200,3))
        for i,v in enumerate(pos):ax.text(*v,case['view_ids'][i],ha='center',bbox=dict(facecolor='white',edgecolor='gray'))
        ax.set(title=f"{case['id']}, seed {seed}\nGreen: fitted geometry accepted",xlim=(-1.3,1.3),ylim=(-1.3,1.3));ax.axis('off')
        from georecon.teaching.initialization import rank_seeds
        ordered=rank_seeds(signals,'quality');ax=fig.add_subplot(132);labels=[f"{case['view_ids'][s['pair'][0]]}-{case['view_ids'][s['pair'][1]]}" for s in ordered]
        ax.barh(range(len(ordered)),[s['essential_inliers'] for s in ordered],color=['steelblue' if s['status']=='success' else 'salmon' for s in ordered])
        ax.set(yticks=range(len(ordered)),yticklabels=labels,xlabel='E inliers (quality rank order)',title='Graph reach / occupancy')
        for i,s in enumerate(ordered):
            occupied=f"{s['coverage']:.3f}" if s['coverage'] is not None else 'unknown'
            ax.text(s['essential_inliers'],i,f" {s['reachable_other_views']} / {occupied}",fontsize=7)
        ax.set_xlim(0,max(s['essential_inliers'] for s in ordered)*1.4+1)
        ax.invert_yaxis();ax=fig.add_subplot(133,projection='3d');points=load(study/'predictions'/case['id']/'C'/str(seed)/'quality/map.npz')['xyz']
        if len(points):ax.scatter(*points.T,s=1,alpha=.4)
        for v,p in rec['poses'].items():center=-np.array(p['R']).T@np.array(p['t']);ax.scatter(*center,c='red',s=20);ax.text(*center,v)
        row=next(r for r in ev['rows'] if r['case']==case['id'] and r['seed']==seed and r['arm']=='C' and r['strategy']=='quality')
        ax.set(title=f"Actual sparse map: score10={row['auc10']:.3f}",xlabel='X (seed units)',ylabel='Y',zlabel='Z')
        fig.tight_layout();fig.savefig(output/f'representative_{label}.svg');fig.savefig(output/f'representative_{label}.png',dpi=140);plt.close(fig)
    # Signal redundancy, no labels used to alter scoring.
    rows=[read(p)['signals'] for p in (study/'predictions').glob('*/C/42/fixed/record.json')]
    signal=[s for group in rows for s in group if s['status']=='success']
    keys=['essential_inliers','coverage','reachable_other_views','kept_points','positive_fraction','parallax_deg','homography_fraction'];a=np.array([[s[k] if s[k] is not None else np.nan for k in keys] for s in signal])
    from scipy.stats import spearmanr
    corr=spearmanr(a,axis=0,nan_policy='omit').statistic
    (output/'signal_correlations.json').write_text(json.dumps({'keys':keys,'spearman':[[float(v) if np.isfinite(v) else None for v in row] for row in corr],'selection_unchanged':True},indent=2,allow_nan=False))
    print('figures',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['study','colmap','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();report(a.study,a.colmap,a.output)
