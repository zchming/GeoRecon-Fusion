"""Recompute old filter evaluation and failure counts into NEW audit only."""
from pathlib import Path
import numpy as np
from stage5_points import Evaluator,ROOT,CONDITIONS
from georecon.data.protocol import read_json,write_json,sha256

def audit():
    prior=read_json(ROOT/'docs/stage5_points/comparison.json');fail=read_json(ROOT/'docs/stage5_points/failure_analysis.json');rows=[]
    for scan in [1,6]:
        ev=Evaluator(scan)
        for condition in CONDITIONS:
            scene=f'dtu_scan{scan}_{condition}';ev.prepare(scene)
            old=next(r for r in prior['rows'] if r['scene_id']==scene)
            base=ROOT/'outputs/stage5_points_v1/frozen'/scene
            run=read_json(base/'run.json');assert sha256(ROOT/run['source'])==run['source_sha256']
            assert sha256(base/'filtered.npz')==run['rows'][0]['sha256']
            keep=np.zeros(ev.n,bool)
            with np.load(base/'filtered.npz') as a:keep[a['source_point_indices']]=True
            for side,k in [('before',np.ones(ev.n,bool)),('after',keep)]:
                assert ev.measure(k)==old[side]
            selected=keep[ev.ids];good=np.zeros(len(ev.ids),bool)
            good[ev.in_mask]=ev.tree.query(ev.xyz[ev.in_mask],workers=8)[0]<=2
            with np.load(base/'evidence.npz') as a:unknown=(a['finite']&(a['observed']==0))[ev.ids]
            f=next(r for r in fail['rows'] if r['scene_id']==scene)
            assert int((ev.in_mask&good&~selected).sum())==f['good_points_removed']
            assert int((ev.in_mask&good&unknown).sum())==f['good_unknown_removed']
            rows.append({'scene_id':scene,'old_metrics_recomputed_and_identical':True,'old_good_removed_count_recomputed':f['good_points_removed'],'old_good_unknown_removed_count_recomputed':f['good_unknown_removed']})
    write_json(ROOT/'docs/stage5_fusion/filter_audit.json',{'rows':rows,'prior_source_sha256':sha256(ROOT/'docs/stage5_points/comparison.json'),'prior_failures_sha256':sha256(ROOT/'docs/stage5_points/failure_analysis.json'),'response':'start from UNFILTERED raw; unavailable correspondences retain exact XYZ','prior_test_already_seen':True})
if __name__=='__main__':audit()
