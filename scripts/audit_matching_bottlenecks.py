"""Read-only reuse of old databases and reports: no matching, backend or GT rerun."""
from pathlib import Path
import sqlite3
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/sift_lightglue_preflight'
def audit():
    source={};cases=[]
    for scan in [1,6]:
        for condition in ['normal5','reduced3','spaced3','blur5']:
            scene=f'dtu_scan{scan}_{condition}';base=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap';path=base/'database.db';before=sha256(path);source[str(path.relative_to(ROOT))]=before;mapping=read_json(base/'input_mapping.json');lookup={x['worker_name']:x['image_id'] for x in mapping};pred=read_json(base/'prediction.json');evaluation=read_json(base/'evaluation.json')
            with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
                images={int(i):lookup[name] for i,name in db.execute('SELECT image_id,name FROM images')};kp={int(i):int(n) for i,n in db.execute('SELECT image_id,rows FROM keypoints')};raw={int(i):int(n) for i,n in db.execute('SELECT pair_id,rows FROM matches')};verified={int(i):(int(n),int(config)) for i,n,config in db.execute('SELECT pair_id,rows,config FROM two_view_geometries')}
            pairs=[]
            for pair in sorted(set(raw)|set(verified)):
                i=pair//2147483647;j=pair%2147483647;v,c=verified.get(pair,(0,None));pairs.append({'ids':[images[i],images[j]],'raw_matches':raw.get(pair,0),'verified_rows':v,'geometry_config_numeric':c})
            edges=[p['ids'] for p in pairs if p['verified_rows']>0];isolated=[name for name in images.values() if not any(name in e for e in edges)];assert sha256(path)==before
            cases.append({'scene_id':scene,'historical_status':pred['status'],'keypoints_by_image':{images[i]:v for i,v in kp.items()},'pairs':pairs,'isolated_images_in_nonzero_verified_graph':isolated,'old_offline_pose_evaluation':evaluation,'diagnostic_scope':'raw/verified correspondences and initialization support; no new pose or causal attribution'})
            for p in [base/'prediction.json',base/'evaluation.json',base/'worker.log',base/'input_mapping.json']:source[str(p.relative_to(ROOT))]=sha256(p)
    for p in ['docs/stage3/report.md','docs/stage4/report.md','docs/stage6/summary.json']:source[p]=sha256(ROOT/p)
    write_json(DOC/'bottleneck_audit.json',{'cases':cases,'known_two_view_pair_failure':read_json(ROOT/'docs/stage6/summary.json')['real_matches_and_points'],'input_hashes':source,'all_databases_opened_readonly_and_hashes_unchanged':True,'limitations':['verified graph is estimator output, not measured physical overlap','PLANAR_OR_PANORAMIC label is not proof of actual scene degeneracy','blur wrong poses despite matches: intrinsics, geometry and front end not causally isolated','no guarantee more learned matches correct these failures']})
    print('Read-only bottleneck audit: 8 existing COLMAP databases and historical pair reports verified.')
if __name__=='__main__':audit()
