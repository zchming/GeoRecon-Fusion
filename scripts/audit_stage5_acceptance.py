"""Read-only audit of archived stage5 output files and metadata; new audit only."""
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.evaluation.poses import evaluate_poses
from georecon.selection.policy import choose
from stage5_points import Evaluator
from stage5_fusion import measure
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5_acceptance'


def audit():
    source={};rows=[]
    for stage,key in [('stage5_points','source_files'),('stage5_fusion','source_files_sha256'),('stage5_joint','source_sha256')]:
        lock=read_json(ROOT/'docs'/stage/'input_lock.json')
        for name,digest in lock[key].items():assert sha256(ROOT/name)==digest
        source[stage]=sha256(ROOT/'docs'/stage/'input_lock.json')
    selection=read_json(ROOT/'docs/stage5/input_lock.json');freeze=read_json(ROOT/'docs/stage3/freeze.json')
    policy=read_json(ROOT/'configs/selection/stage5_v1.json')
    oldpoints=read_json(ROOT/'docs/stage5_points/comparison.json');oldfusion=read_json(ROOT/'docs/stage5_fusion/comparison.json');oldjoint=read_json(ROOT/'docs/stage5_joint/comparison.json')
    for scan in [1,6]:
        ev=Evaluator(scan)
        for condition in ['blur5','normal5','reduced3','spaced3']:
            scene=f'dtu_scan{scan}_{condition}';ev.prepare(scene)
            case=next(c for c in selection['cases'] if c['scene_id']==scene);bundle=ROOT/case['bundle']
            for name,digest in case['files'].items():assert sha256(bundle/name)==digest
            predictions={m:read_json(bundle/(m+'.json')) for m in policy['priority']};hashes={m:sha256(bundle/(m+'.json')) for m in policy['priority']}
            manifest=read_json(bundle/'manifest.json');diagnostics=read_json(bundle/'reliability.json')
            decision=choose(predictions,diagnostics,policy,scene,sha256(bundle/'manifest.json'),[i['image_id'] for i in manifest['images']],hashes)
            assert decision==read_json(ROOT/'outputs/stage5_v1'/scene/'decision.json')
            a=next(r for r in oldpoints['rows'] if r['scene_id']==scene);b=next(r for r in oldfusion['rows'] if r['scene_id']==scene);c=next(r for r in oldjoint['rows'] if r['scene_id']==scene)
            assert ev.measure(np.ones(ev.n,bool))==a['before']==b['raw']==c['raw']
            base=ROOT/'outputs/stage5_points_v1/frozen'/scene
            run=read_json(base/'run.json');assert sha256(base/'filtered.npz')==run['rows'][0]['sha256']
            with np.load(base/'filtered.npz') as z:ids=z['source_point_indices']
            mask=np.zeros(ev.n,bool);mask[ids]=True;assert ev.measure(mask)==a['after']
            for name,old in [('stage5_fusion_v1',b),('stage5_joint_v1',c)]:
                folder=ROOT/'outputs'/name/'frozen'/scene;run=read_json(folder/'run.json')
                artifact='fused.npz' if name=='stage5_fusion_v1' else 'optimized.npz'
                digest=run['rows'][0]['sha256' if name=='stage5_fusion_v1' else 'artifact_sha256']
                assert sha256(folder/artifact)==digest
                expected=old['weighted_fused' if name=='stage5_fusion_v1' else 'optimized'];assert measure(ev,folder/artifact)==expected
                pred=validate_result(read_json(folder/'prediction.json'))
                for desc in pred['artifacts'].values():
                    if desc is not None:assert sha256(folder/desc['path'])==desc['sha256']
            truth=read_json(ROOT/'data/stage3_dtu'/scene.removeprefix('dtu_')/'reference.json')
            pred=read_json(ROOT/'outputs/stage5_joint_v1/frozen'/scene/'prediction.json');assert evaluate_poses(pred,truth)==c['poses_after']
            rows.append({'scene_id':scene,'selection_decision_recomputed':True,'raw_filter_fusion_joint_point_metrics_recomputed':True,'joint_pose_metrics_recomputed':True,'artifact_hashes_verified':True})
    rules=[ROOT/'configs'/d/f for d,f in [('selection','stage5_v1.json'),('filtering','stage5_points_v1.json'),('fusion','stage5_v1.json'),('optimization','stage5_v1.json')]]
    write_json(DOC/'prior_audit.json',{'rows':rows,'prior_source_locks':source,'frozen_rule_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in rules},'all_prior_outputs_unchanged':True,'prior_test_already_seen':True})
if __name__=='__main__':audit()
