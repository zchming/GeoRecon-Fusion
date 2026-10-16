"""Verify stage boundaries, immutable selection outputs, point provenance and diagnostics."""
from pathlib import Path
import copy,sys,subprocess
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_points'


def audit(final=False):
    protocol=read_json(ROOT/'configs/filtering/stage5_points_protocol.json')
    assert sha256(ROOT/'docs/stage3/freeze.json')==protocol['stage3_freeze_sha256']
    assert sha256(ROOT/'configs/selection/stage5_v1.json')==protocol['selection_policy_sha256']
    lock=read_json(ROOT/'docs/stage5/input_lock.json');rows=[]
    for case in lock['cases']:
        for name,digest in case['files'].items(): assert sha256(ROOT/case['bundle']/name)==digest
        scene=case['scene_id'];old=ROOT/'outputs/stage5_v1'/scene
        pred=read_json(old/'prediction.json'); decision=read_json(old/'decision.json')
        assert decision==read_json(ROOT/'docs/stage5/runs'/scene/'decision.json')
        selected=decision['selected_method'];assert selected=='vggt_raw'
        native=read_json(ROOT/case['bundle']/(selected+'.json'))
        assert pred['images']==native['images']
        p=old/pred['artifacts']['pointcloud']['path'];assert sha256(p)==pred['artifacts']['pointcloud']['sha256']
        row={'scene_id':scene,'split':case['split'],'selection':decision,'source_sha256':sha256(p)}
        if final:
            dest=ROOT/'outputs/stage5_points_v1/frozen'/scene
            run=read_json(dest/'run.json');output=validate_result(read_json(dest/'prediction.json'))
            assert run['source_sha256']==sha256(p) and output['images']==pred['images']
            for a in output['artifacts'].values():
                if a is not None:assert sha256(dest/a['path'])==a['sha256']
            with np.load(dest/'evidence.npz') as ev,np.load(dest/'filtered.npz') as a,np.load(p) as source:
                ids=a['source_point_indices'];assert len(ids)==len(np.unique(ids)) and np.all(np.diff(ids)>0)
                assert np.array_equal(a['xyz'],source['world_points'].reshape(-1,3)[ids])
                rule=read_json(ROOT/'configs/filtering/stage5_points_v1.json')
                expected=ev['finite']&(ev['support'][0]>=rule['min_support'])
                assert np.array_equal(ids,np.flatnonzero(expected))
                assert run['rows'][0]['unknown']==int((ev['finite']&(ev['observed']==0)).sum())
            row['filter_run']=run
            snap=DOC/'runs'/scene;snap.mkdir(parents=True,exist_ok=True)
            for filename in ['run.json','prediction.json']:
                write_json(snap/filename,read_json(dest/filename))
        rows.append(row)
    faults=read_json(ROOT/'docs/stage5/probes.json') if (ROOT/'docs/stage5/probes.json').exists() else None
    write_json(DOC/('validation.json' if final else 'selection_audit.json'),{'rows':rows,'immutable_selection_inputs_verified':True,'complete_camera_outputs_unchanged':True,'natural_fallbacks':sum(r['selection']['fallback_used'] for r in rows),
        'counterexamples':['blur COLMAP can pass internal geometry checks but have bad GT pose','all natural outputs selected raw; fallback tested only by counterfactual dev probes','point/depth heads are correlated; agreement is not correctness','occlusion and missing views can remove valid geometry'],
        'point_subset_bytes_verified':final,'weighted_fusion_or_joint_optimization':False})


if __name__=='__main__':audit('--final' in sys.argv)
