"""GT-blocked conditional cross-fit copies over original 502 positions."""
import argparse,resource,sys,time
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from georecon.fusion.anchors import predicted_alignment
from heldout_fusion_math import split_groups,project,depth_support,construct,witness_score
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/heldout_fusion_502'

def generate(out):
    start=time.perf_counter();out=Path(out).resolve()
    if out.exists():raise ValueError('preserve prior attempts: output exists')
    out.mkdir(parents=True);lock=read_json(DOC/'input_lock.json');protocol=read_json(DOC/'protocol.json');assert sha256(DOC/'protocol.json')==lock['protocol_sha256'];accesses=[]
    def guard(event,args):
        if event=='open' and isinstance(args[0],str):
            name=args[0].replace('\\','/').lower()
            if any(token in name for token in lock['generation_forbidden_reads']):raise PermissionError('GT/evaluation reads blocked during diagnostic candidate construction')
            accesses.append(args[0])
    sys.addaudithook(guard)
    cfg=read_json(ROOT/'configs/fusion/stage5_protocol.json');rule=read_json(ROOT/'configs/fusion/stage5_v1.json');assert sha256(ROOT/'configs/fusion/stage5_v1.json')==lock['old_rule_sha256']
    paths=['scripts/generate_heldout_fusion.py','scripts/heldout_fusion_math.py','scripts/prepare_heldout_fusion.py','tests/test_heldout_fusion.py'];source={p:sha256(ROOT/p) for p in paths};write_json(out/'source_lock.json',source);write_json(out/'protocol.json',protocol);write_json(out/'input_lock.json',lock)
    counts=[]
    for scene in protocol['objects']:
        folder=ROOT/'outputs/stage5_fusion_v1/frozen'/scene;dest=out/scene;dest.mkdir()
        with np.load(folder/'anchors.npz') as z:anchors={k:z[k] for k in z.files}
        with np.load(folder/'fused.npz') as z:oldfused=z['xyz'][anchors['source_point_indices']]
        with np.load(ROOT/'outputs/stage5_v1'/scene/'selected_artifacts/raw_predictions.npz') as z:data={k:z[k] for k in ['world_points','depth','intrinsics','extrinsics']}
        depth=data['depth'][...,0];n,h,w=depth.shape;raw=data['world_points'].reshape(-1,3)[anchors['source_point_indices']]
        with np.load(ROOT/'data/stage5_fusion_tracks'/scene/'observations.npz') as z:tracks={k:z[k] for k in z.files}
        cm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/input_mapping.json');rm=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'vggt_raw/input_mapping.json');bycm={v['image_id']:v for v in cm};vlookup={v['image_id']:i for i,v in enumerate(rm)}
        col=read_json(ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap/prediction.json');rawpred=read_json(ROOT/'outputs/stage5_v1'/scene/'prediction.json');colcams={}
        for item in col['images']:
            if item['registered']:
                c=item['camera'];A=np.array(bycm[item['image_id']]['pixel_transform']);colcams[item['image_id']]={'K':A@c['K'],'R':np.array(c['R']),'t':np.array(c['t'])}
        rawcams={name:{'K':data['intrinsics'][v].astype(float),'R':data['extrinsics'][v,:,:3].astype(float),'t':data['extrinsics'][v,:,3].astype(float)} for name,v in vlookup.items()}
        sims={};alignments={}
        for held in [None,*vlookup]:
            cp={**col,'images':[i for i in col['images'] if i['image_id']!=held]};rp={**rawpred,'images':[i for i in rawpred['images'] if i['image_id']!=held]};sims[held],alignments[str(held)]=predicted_alignment(cp,rp,cfg['alignment'])
        folds=[];xyz=[];fullxyz=[];point_rows=[]
        for p,(grid,pid) in enumerate(zip(anchors['source_point_indices'],anchors['point_id'])):
            source_frame=int(grid//(h*w));source=rm[source_frame]['image_id'];global_rows=np.flatnonzero(tracks['point_id']==pid);names=tracks['image_id'][global_rows];common=tracks['xy'][global_rows]-.5;native=[]
            for name,uv in zip(names,common):
                T=np.array(rm[vlookup[str(name)]]['original_to_model_transform'])@np.linalg.inv(np.array(bycm[str(name)]['pixel_transform']));q=T@np.r_[uv,1.];native.append(q[:2]/q[2])
            native=np.array(native);source_candidates=[]
            for i in np.flatnonzero(names==source):
                cell=np.floor(native[i]+.5).astype(int);projected,z=project(anchors['anchor_xyz'][p],rawcams[source])
                if source_frame*h*w+cell[1]*w+cell[0]==grid and z>0 and np.linalg.norm(projected-native[i])<=cfg['correspondence']['max_aligned_anchor_reprojection_error_native_px']:source_candidates.append(int(i))
            assert source_candidates;source_row=source_candidates[0];source_uv=native[source_row];allrows=np.arange(len(names));support_all,_=depth_support(raw[p],source_frame,None,depth,data['extrinsics'],data['intrinsics'])
            full,fullinfo=construct(allrows,names,common,colcams,sims[None],raw[p],rawcams[source],source_uv,support_all,n,rule['strength'],cfg['correspondence'])
            point_rows.append({'point_index':p,'source_point_index':int(grid),'point_id':int(pid),'source_frame':source_frame,'source_image':source,'global_track_rows':global_rows.tolist(),'source_local_row':source_row,'track_rows':len(names),'distinct_views':len(set(names.tolist())),'full_DLT_control':fullinfo})
            for witness in sorted(set(names.tolist())-{source}):
                train,test,feasible=split_groups(names,source,witness);v=vlookup[witness];support,observed=depth_support(raw[p],source_frame,v,depth,data['extrinsics'],data['intrinsics'])
                candidate,info=construct(train,names,common,colcams,sims[witness],raw[p],rawcams[source],source_uv,support,n,rule['strength'],cfg['correspondence']) if feasible else (None,{'reason':'geometrically_infeasible','training_rows':train.tolist()})
                score,reason=witness_score(raw[p],candidate,native[test],rawcams[witness],w,h);fs,fr=witness_score(raw[p],full,native[test],rawcams[witness],w,h);original_score,original_reason=witness_score(raw[p],oldfused[p],native[test],rawcams[witness],w,h)
                folds.append({'fold_index':len(folds),'point_index':p,'point_id':int(pid),'source_image':source,'witness_image':witness,'train_local_rows':train.tolist(),'witness_local_rows':test.tolist(),'train_global_rows':global_rows[train].tolist(),'witness_global_rows':global_rows[test].tolist(),'geometrically_feasible':feasible,'construction':info,'witness_unavailable_reason':reason,'score':score,'raw_support_views_excluded':[source,witness],'raw_supported_construction_views':support,'raw_observed_construction_views':observed,'full_DLT_score':fs,'full_DLT_witness_reason':fr,'original_frozen_score':original_score,'original_frozen_witness_reason':original_reason,'alignment_camera_ids':alignments[witness].get('shared_ids',[])})
                xyz.append(np.full(3,np.nan) if candidate is None else candidate);fullxyz.append(np.full(3,np.nan) if full is None else full)
        write_json(dest/'folds.json',folds);write_json(dest/'positions.json',point_rows);write_json(dest/'alignments.json',alignments);np.savez_compressed(dest/'candidates.npz',raw_xyz=raw,crossfit_xyz=np.array(xyz),full_DLT_xyz=np.array(fullxyz),original_frozen_xyz=oldfused,source_point_indices=anchors['source_point_indices'],point_id=anchors['point_id'])
        counts.append({'scene_id':scene,'positions':len(raw),'folds':len(folds),'geometrically_feasible_folds':sum(f['geometrically_feasible'] for f in folds),'constructed_folds':sum(f['construction']['reason'] is None for f in folds),'scorable_folds':sum(f['score'] is not None for f in folds)})
    write_json(out/'execution.json',{'wall_seconds':time.perf_counter()-start,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_task_allocated_bytes':0,'CPU_threads':8,'counts':counts})
    write_json(out/'invocation.json',{'GT_guard_active':True,'project_file_accesses':sorted({p for p in accesses if p.startswith(str(ROOT))}),'source_sha256':source,'scope':'fixed-camera conditional heldout diagnostics only'})
    print(counts)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);generate(p.parse_args().output_root)
