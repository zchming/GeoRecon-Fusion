"""Audit incremental acceptance; lock exact baseline maps and BA configuration."""
from pathlib import Path
from georecon.data.protocol import read_json,write_json,sha256
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_ba'
def prepare():
    if (DOC/'input_lock.json').exists():raise ValueError('BA input lock already exists')
    old=read_json(ROOT/'docs/stage6_incremental/source_lock.json');accept=read_json(ROOT/'docs/stage6_incremental/acceptance.json');assert accept['engineering_acceptance']=='passed'
    for p,h in old['source_sha256'].items():assert sha256(ROOT/p)==h
    cfg={'schema_version':1,'scope':'stage6 minimal teaching calibrated BA only; no next stage','known_K':'unchanged per-image supplied calibration, zero-based 800x600 pixels; no distortion fit','solver':'SciPy sparse least_squares trf/lsmr, 2-point Jacobian','loss':'soft_l1 on individual pixel residual components','loss_scale_px':1.,'gauge':'fix first seed pose and largest nonzero initial second seed center coordinate relative to first; all other poses/points free','min_points':20,'max_points':3000,'max_observations':20000,'min_registered_cameras':3,'max_nfev':80,'wall_budget_seconds':60.,'ftol':1e-8,'xtol':1e-8,'gtol':1e-8,'CPU_threads':8,'GPU_budget_GiB':12,'GPU_allocated_bytes':0,'fit_on_GT':False,'filter_observations':False,'filter_points':False,'priors':False,'retry_or_parameter_search':False,'acceptance':'finite solution, all observed depths positive, robust objective nonincreasing; max_nfev status0 may be accepted with explicit nonconverged label; timeout rejected','fallback':'retain original registration, poses, points, tracks; preserve rejected candidate'}
    write_json(ROOT/'configs/teaching/stage6_ba.json',cfg);rows=[]
    for case in read_json(ROOT/'docs/stage6_incremental/input_lock.json')['cases']:
        for key in ['input','reference']:assert sha256(ROOT/case[key])==case[key+'_sha256']
        folder=ROOT/'outputs/stage6_incremental_v1'/case['case_id'];pred=read_json(folder/'prediction.json');assert pred==read_json(ROOT/'docs/stage6_incremental/runs'/case['case_id']/'prediction.json')
        for artifact in pred['artifacts'].values():
            if artifact:assert sha256(folder/artifact['path'])==artifact['sha256']
        rows.append({**case,'baseline':str(folder.relative_to(ROOT)),'baseline_files_sha256':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()}})
    write_json(DOC/'previous_audit.json',{'acceptance_sha256':sha256(ROOT/'docs/stage6_incremental/acceptance.json'),'source_hashes_verified':old['source_sha256'],'previous_replay_log':'previous_check.txt','previous_failures_preserved':['001/020 seed has 4 matches, no map','blank 002 has 0 links, partial 2-camera map with 404 points'],'old_files_modified':False})
    write_json(DOC/'input_lock.json',{'config_sha256':sha256(ROOT/'configs/teaching/stage6_ba.json'),'cases':rows,'known_K_and_all_original_3_or_5_views_preserved':True,'no_new_downloads':True,'data_profile':'exact previous incremental maps/observations, no new feature extraction or map re-selection'})
if __name__=='__main__':prepare()
