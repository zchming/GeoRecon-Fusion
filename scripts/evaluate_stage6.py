"""Offline pose/fixture evaluation; inference never imports this module or reads truth."""
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from georecon.evaluation.poses import rotation_error_deg,translation_direction_error_deg
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6';OUT=ROOT/'outputs/stage6_two_view_v1'


def evaluate():
    lock=read_json(DOC/'input_lock.json');rows=[]
    for case in lock['cases']:
        base=OUT/case['case_id'];p=read_json(base/'prediction.json');stats=read_json(base/'steps.json')
        ref=ROOT/case['reference'];assert sha256(ref)==case['reference_sha256'];truth=read_json(ref)
        row={'case_id':case['case_id'],'kind':p['provenance']['data_kind'],'input_mode':stats['mode'],'status':p['status'],'failure_reason':p['failure_reason'],
            'features':stats['feature_stage'],'geometry':stats['geometry_stage'],'expected_status':truth['expected_status'],'expected_status_met':truth['expected_status'] is None or truth['expected_status']==p['status'],
            'rotation_error_deg':None,'translation_direction_error_deg':None,'metric_scale_evaluated':False,'reference_access':'offline evaluation only'}
        if p['status']=='success' and 'relative_R' in truth:
            c=p['images'][1]['camera'];r=np.array(c['R']);t=np.array(c['t']);gt=np.array(truth['relative_t'])
            row['rotation_error_deg']=rotation_error_deg(r,np.array(truth['relative_R']));row['translation_direction_error_deg']=translation_direction_error_deg(t,gt)
            if 'first_camera_xyz' in truth and np.linalg.norm(gt)>0:
                with np.load(base/'sparse_points.npz') as a:xyz=a['xyz'];ids=a['match_indices']
                true_mask=np.array(truth['true_match_mask'],bool)[ids];target=np.array(truth['first_camera_xyz'])[ids]/np.linalg.norm(gt)
                distances=np.linalg.norm(xyz[true_mask]-target[true_mask],axis=1)
                row['synthetic_oracle_scale_shape_error']={'scope':'synthetic only, GT baseline used only for evaluation; exclude injected false pairs','matched_true_points':int(true_mask.sum()),'mean':float(distances.mean()),'median':float(np.median(distances)),'unit':'unit-baseline world','false_pairs_kept':int((~true_mask).sum())}
        rows.append(row)
        print(case['case_id'],p['status'],row['rotation_error_deg'],row['translation_direction_error_deg'],flush=True)
    write_json(DOC/'evaluation.json',{'rows':rows,'not_a_benchmark_or_model_ranking':True,'known_K_assumption_differs_from_stage3_self_calibrated_COLMAP':True,'cases_with_expected_status':sum(r['expected_status'] is not None for r in rows),'all_expected_statuses_met':all(r['expected_status_met'] for r in rows)})

if __name__=='__main__':evaluate()
