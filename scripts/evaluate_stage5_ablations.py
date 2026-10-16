"""Posthoc GT evaluation of frozen diagnostic controls, never rule fitting."""
import time,resource
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from stage5_points import Evaluator
from stage5_fusion import measure
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_acceptance';OUT=ROOT/'outputs/stage5_acceptance_v1'


def evaluate():
    start=time.perf_counter();protocol=read_json(ROOT/'configs/ablation/stage5_acceptance.json');rows=[]
    points=read_json(ROOT/'docs/stage5_points/comparison.json');fusion=read_json(ROOT/'docs/stage5_fusion/comparison.json');joint=read_json(ROOT/'docs/stage5_joint/comparison.json')
    for scan in [1,6]:
        ev=Evaluator(scan)
        for scene in protocol['cases']:
            if not scene.startswith(f'dtu_scan{scan}_'):continue
            ev.prepare(scene);dest=OUT/scene;run=read_json(dest/'run.json')
            assert run['protocol_sha256']==sha256(ROOT/'configs/ablation/stage5_acceptance.json')
            a=next(r for r in points['rows'] if r['scene_id']==scene);b=next(r for r in fusion['rows'] if r['scene_id']==scene);c=next(r for r in joint['rows'] if r['scene_id']==scene)
            methods={'raw_selector':a['before'],'filter':a['after'],'fusion':b['weighted_fused'],'unweighted_fusion':b['unweighted'],'grouped_initial':c['grouped_initial'],'joint':c['optimized']}
            for row in run['rows']:
                name=row['name'];p=dest/(name+'.npz');assert sha256(p)==row['artifact_sha256']
                if name=='filter_keep_unknown':
                    keep=np.zeros(ev.n,bool)
                    with np.load(p) as z:keep[z['source_point_indices']]=True
                    methods[name]=ev.measure(keep)
                else:methods[name]=measure(ev,p)
            rows.append({'scene_id':scene,'split':'dev' if scan==1 else 'test','methods':methods,'selection_control':read_json(dest/'selection_control.json'),'run':run,'GT_point_alignment':ev.alignment})
            print(scene,{m:round(v['thresholds_mm']['2']['f1'],8) for m,v in methods.items()},flush=True)
    write_json(DOC/'ablation_comparison.json',{'rows':rows,'protocol_sha256':sha256(ROOT/'configs/ablation/stage5_acceptance.json'),'configuration_fit':False,'offline_evaluation_seconds':time.perf_counter()-start,'peak_evaluator_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2})

if __name__=='__main__':evaluate()
