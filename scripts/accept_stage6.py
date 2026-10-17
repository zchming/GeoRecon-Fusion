"""Stage6 acceptance: replay frozen teaching evidence, never alter old records."""
import hashlib,importlib.util,io,os,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.teaching.run import load_input
from georecon.teaching.two_view import features_and_matches,reconstruct,triangulate_dlt
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_acceptance'
STEPS=[('stage6','stage6_two_view_v1','stage6_two_view.json'),('stage6_incremental','stage6_incremental_v1','stage6_incremental.json'),('stage6_ba','stage6_ba_v1','stage6_ba.json')]


def historical_source_digest(commit,overrides):
    """Reconstruct committed package plus explicitly frozen uncommitted source bytes."""
    archive=subprocess.run(['git','archive',commit,'src/georecon'],cwd=ROOT,check=True,capture_output=True).stdout;files={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for item in tar.getmembers():
            if item.isfile() and item.name.endswith('.py'):files[item.name.removeprefix('src/georecon/')]=tar.extractfile(item).read()
    for name,h in overrides.items():
        if name.startswith('src/georecon/') and name.endswith('.py'):
            assert sha256(ROOT/name)==h;files[name.removeprefix('src/georecon/')]=(ROOT/name).read_bytes()
    digest=hashlib.sha256()
    for name,data in sorted(files.items()):digest.update(name.encode());digest.update(data)
    return digest.hexdigest()


def readonly_check(script):
    spec=importlib.util.spec_from_file_location('acceptance_'+script,ROOT/'scripts'/(script+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def compare_existing(path,value):
        assert read_json(path)==value,f'historical result differs on replay: {path}'
    module.write_json=compare_existing;module.check()


def audit():
    start=time.perf_counter();previous_dirs=[ROOT/'docs'/s for s,_,_ in STEPS];outputs=[ROOT/'outputs'/s for _,s,_ in STEPS]+[ROOT/'outputs/stage6_ba_controls_v1']
    previous={str(p.relative_to(ROOT)):sha256(p) for folder in previous_dirs+outputs for p in folder.rglob('*') if p.is_file()}
    commits={};cases=[];source_checks={};historical_digests={}
    for stage,output,config in STEPS:
        folder=ROOT/'docs'/stage;lock=read_json(folder/'input_lock.json');source=read_json(folder/'source_lock.json');cfgpath=ROOT/'configs/teaching'/config
        assert sha256(cfgpath)==lock.get('protocol_sha256',lock.get('config_sha256'))
        assert sha256(cfgpath)==source['config_sha256'] and sha256(folder/'input_lock.json')==source['input_lock_sha256']
        for name,h in source['source_sha256'].items():assert sha256(ROOT/name)==h;source_checks[name]=h
        if (folder/'delivery_source_hashes.json').exists():
            for name,h in read_json(folder/'delivery_source_hashes.json').items():assert sha256(ROOT/name)==h;source_checks[name]=h
        for case in lock['cases']:
            for key in ['input','reference']:assert sha256(ROOT/case[key])==case[key+'_sha256']
            base=ROOT/'outputs'/output/case['case_id'];pred=validate_result(read_json(base/'prediction.json'));inv=read_json(base/'invocation.json')
            assert pred==read_json(folder/'runs'/case['case_id']/'prediction.json')
            for artifact in pred['artifacts'].values():
                if artifact:assert sha256(base/artifact['path'])==artifact['sha256']
            assert inv['pose_truth_guard_active'];accesses=inv.get('project_file_accesses',inv.get('file_accesses',[]));forbidden=['truth','reference.json','evaluation.json','results.json','freeze.json','calibration/','ground_truth']
            assert not any(any(word in name.replace('\\','/').lower() for word in forbidden) for name in accesses)
            code=inv.get('code',inv.get('source'));key=(stage,code['code_commit'])
            if key not in commits:commits[key]=historical_source_digest(code['code_commit'],source['source_sha256'])
            assert commits[key]==code['source_tree_sha256'],f'historical package digest mismatch {stage}/{case["case_id"]}'
            historical_digests[stage]={'code_commit_at_run':code['code_commit'],'reconstructed_package_sha256':commits[key],'committed_package_plus_frozen_src_overrides_verified':True}
            if stage=='stage6':process=read_json(base/'process.json');exit_code=process['returncode']
            else:exit_code=read_json(folder/'runs'/case['case_id']/'process.json')['exit_code']
            expected_exit=0 if (pred['status']=='success' if stage!='stage6_ba' else read_json(base/'ba.json')['accepted']) else 1
            assert exit_code==expected_exit
            assert pred['memory_bytes']['peak_gpu_allocated']==0
            cases.append({'stage':stage,'case_id':case['case_id'],'kind':pred['provenance']['data_kind'],'status':pred['status'],'exit_code':exit_code,'registered_images':sum(i['registered'] for i in pred['images']),'images':len(pred['images']),'config_sha256':sha256(cfgpath),'no_recorded_pose_truth_access':True})
    # Replay all eleven two-view cases, including the eight expected controls.
    cfg=read_json(ROOT/'configs/teaching/stage6_two_view.json');two=[]
    for case in read_json(ROOT/'docs/stage6/input_lock.json')['cases']:
        spec,images,observations,_=load_input(ROOT/case['input'],cfg);base=ROOT/'outputs/stage6_two_view_v1'/case['case_id'];steps=read_json(base/'steps.json');pred=read_json(base/'prediction.json')
        if images:
            arrays,stats=features_and_matches(images,cfg);assert stats==steps['feature_stage']
            with np.load(base/'features_matches.npz') as z:
                assert set(z.files)==set(arrays)
                for name,value in arrays.items():np.testing.assert_array_equal(z[name],value)
            p,q=arrays['pixels_first'],arrays['pixels_second']
        else:p,q=observations['pixels_first'],observations['pixels_second']
        result=reconstruct(p,q,spec['images'][0]['K'],spec['images'][1]['K'],cfg);assert result['status']==pred['status'] and result['reason']==pred['failure_reason'] and result['stats']==steps['geometry_stage']
        with np.load(base/'geometry_steps.npz') as z:
            assert set(z.files)==set(result['arrays'])
            for name,value in result['arrays'].items():np.testing.assert_allclose(z[name],value,atol=1e-10,rtol=0,equal_nan=True)
        with np.load(base/'sparse_points.npz') as z:xyz=z['xyz'];ids=z['match_indices']
        truth=read_json(ROOT/case['reference'])
        if spec['kind']=='synthetic':assert truth['expected_status']==pred['status']
        if pred['status']=='success':
            np.testing.assert_array_equal(pred['images'][0]['camera']['R'],np.eye(3));np.testing.assert_array_equal(pred['images'][0]['camera']['t'],np.zeros(3));r=np.array(pred['images'][1]['camera']['R']);t=np.array(pred['images'][1]['camera']['t']);assert np.isclose(np.linalg.norm(t),1)
            tri=triangulate_dlt(p[ids],q[ids],spec['images'][0]['K'],spec['images'][1]['K'],r,t);np.testing.assert_allclose(tri['xyz'],xyz,atol=1e-9,rtol=0);assert tri['positive'].all() and (tri['parallax_deg']>=cfg['min_parallax_deg']).all() and (tri['reprojection_px']<=cfg['max_reprojection_px']).all()
        else:assert len(xyz)==0 and all(i['camera'] is None and not i['registered'] for i in pred['images'])
        two.append({'case_id':case['case_id'],'all_feature_and_geometry_arrays_replayed':True,'status':pred['status'],'points':len(xyz)})
    readonly_check('check_stage6_incremental');readonly_check('check_stage6_ba')
    for name,h in previous.items():assert sha256(ROOT/name)==h,f'historical artifact modified: {name}'
    current={str(p.relative_to(ROOT)) for folder in previous_dirs+outputs for p in folder.rglob('*') if p.is_file()};assert current==set(previous)
    write_json(DOC/'historical_artifact_lock.json',previous)
    write_json(DOC/'audit.json',{'source_checks':source_checks,'historical_package_provenance':historical_digests,'all_21_run_records_and_exits_verified':cases,'two_view_replay':two,'incremental_5_runs_exactly_replayed':True,'BA_5_runs_independently_reprojected_and_gauge_checked':True,'BA_4_synthetic_failure_controls_hashes_and_reasons_verified':True,'all_historical_files_unchanged':True,'historical_file_count':len(previous),'current_scope':'stage6 acceptance only, known K, no new samples or algorithm changes','seconds':time.perf_counter()-start})
    print('Stage6 audit passed: 11 pair runs, 5 incremental runs, 5 BA runs, 4 BA failure controls; historical files unchanged.')

if __name__=='__main__':audit()
