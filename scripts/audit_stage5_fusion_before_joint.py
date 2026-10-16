"""Recompute published fusion metrics and local failures into a new audit."""
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from georecon.geometry.alignment import Similarity
from stage5_points import Evaluator
from stage5_fusion import ROOT,OUT,CONDITIONS,measure

def audit():
    comparison=read_json(ROOT/'docs/stage5_fusion/comparison.json');failures=read_json(ROOT/'docs/stage5_fusion/failure_analysis.json');rows=[]
    for scan in [1,6]:
        ev=Evaluator(scan)
        for cond in CONDITIONS:
            scene=f'dtu_scan{scan}_{cond}';ev.prepare(scene);dest=OUT/'frozen'/scene
            row=next(r for r in comparison['rows'] if r['scene_id']==scene)
            run=read_json(dest/'run.json')
            assert sha256(dest/'fused.npz')==run['rows'][0]['sha256']
            assert ev.measure(np.ones(ev.n,bool))==row['raw']
            assert measure(ev,dest/'fused.npz')==row['weighted_fused']
            assert measure(ev,dest/'unweighted.npz')==row['unweighted']
            a=ev.alignment;sim=Similarity(a['scale'],np.array(a['R']),np.array(a['t']))
            with np.load(dest/'anchors.npz') as z:ids=z['source_point_indices']
            with np.load(ROOT/'outputs/stage5_v1'/scene/'selected_artifacts/raw_predictions.npz') as z:before=z['world_points'].reshape(-1,3)[ids]
            with np.load(dest/'fused.npz') as z:after=z['xyz'][ids]
            db=ev.tree.query(sim.apply(before),workers=8)[0] if len(ids) else np.array([])
            da=ev.tree.query(sim.apply(after),workers=8)[0] if len(ids) else np.array([])
            f=next(r for r in failures['rows'] if r['scene_id']==scene)
            assert int((da>db+1e-9).sum())==f['distance_worsened']
            assert int(((db<=2)&(da>2)).sum())==f['good_to_bad_2mm']
            rows.append({'scene_id':scene,'published_metrics_recomputed_identical':True,'affected_points':len(ids),'worsened_points_recomputed':f['distance_worsened'],'good_to_bad_2mm':f['good_to_bad_2mm'],'fallback':run['fallback']})
    write_json(ROOT/'docs/stage5_joint/fusion_audit.json',{'rows':rows,'previous_comparison_sha256':sha256(ROOT/'docs/stage5_fusion/comparison.json'),'previous_failures_sha256':sha256(ROOT/'docs/stage5_fusion/failure_analysis.json'),'conclusion':'weighting did not consistently beat unweighted; only 502 points in two normal5 cases; joint step stays on same correspondences','previous_test_already_seen':True})
if __name__=='__main__':audit()
