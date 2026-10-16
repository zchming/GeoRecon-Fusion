"""Read immutable native COLMAP model; export measured observations, never rerun SfM."""
import argparse,sys,time,resource
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from georecon.reliability.analyze import deny_ground_truth
ROOT=Path(__file__).resolve().parents[1]

def export(scene):
    import pycolmap
    start=time.perf_counter();base=ROOT/'outputs/stage3_dtu_pose_pilot_v1'/scene/'colmap'
    dest=ROOT/'data/stage5_fusion_tracks'/scene
    if dest.exists():raise ValueError('preserve existing track export')
    dest.mkdir(parents=True)
    pred=read_json(base/'prediction.json');mapping=read_json(base/'input_mapping.json')
    files={str(p.relative_to(ROOT)):sha256(p) for p in [base/'prediction.json',base/'input_mapping.json']}
    rows=[];model_paths=[]
    if pred['status'] in ['success','partial']:
        models=sorted((base/'sparse').glob('*/images.bin'))
        # Match the exported selected component rather than assuming component 0 is best.
        expected={m['worker_name'] for m in mapping if next(i for i in pred['images'] if i['image_id']==m['image_id'])['registered']}
        possible=[]
        for model in models:
            recon=pycolmap.Reconstruction(model.parent)
            if {i.name for i in recon.images.values()}==expected:possible.append((model.parent,recon))
        if len(possible)!=1:raise ValueError('cannot uniquely identify selected sparse model')
        path,recon=possible[0];model_paths=[path/name for name in ['cameras.bin','images.bin','points3D.bin']]
        sparse=np.load(base/'sparse_points.npz');lookup={int(i):p for i,p in zip(sparse['point_ids'],sparse['xyz'])}
        files[str((base/'sparse_points.npz').relative_to(ROOT))]=sha256(base/'sparse_points.npz')
        names={m['worker_name']:m['image_id'] for m in mapping}
        for image in recon.images.values():
            for point2d in image.points2D:
                if not point2d.has_point3D():continue
                pid=int(point2d.point3D_id);point=recon.points3D[pid]
                assert np.array_equal(point.xyz,lookup[pid])
                rows.append((names[image.name],pid,point2d.xy,point.xyz,point.error,point.track.length()))
    for path in model_paths:files[str(path.relative_to(ROOT))]=sha256(path)
    rows.sort(key=lambda r:(r[0],r[1],*r[2]))
    np.savez_compressed(dest/'observations.npz',image_id=np.array([r[0] for r in rows],dtype='U8'),point_id=np.array([r[1] for r in rows],dtype=np.int64),
            xy=np.array([r[2] for r in rows]).reshape(-1,2),xyz=np.array([r[3] for r in rows]).reshape(-1,3),error=np.array([r[4] for r in rows]),track_length=np.array([r[5] for r in rows],dtype=np.int32))
    write_json(dest/'export.json',{'scene_id':scene,'upstream_status':pred['status'],'observations':len(rows),'files_sha256':files,'artifact_sha256':sha256(dest/'observations.npz'),
              'pycolmap':pycolmap.__version__,'wall_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'device':'CPU','gpu_allocated_bytes':0})
    print(scene,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--dev-only',action='store_true');args=p.parse_args()
    if args.dev_only:
        from fit_stage5 import dev_only_guard
        sys.addaudithook(dev_only_guard)
    sys.addaudithook(deny_ground_truth);export(args.scene)
