"""Stage 5 point subset experiment: separate GT-free filtering and offline evaluation."""
import argparse, os, sys, time, resource, subprocess, copy
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree
from georecon.data.protocol import read_json,write_json,sha256,validate_result
from georecon.filtering.points import evidence,classify
from georecon.geometry.alignment import align_points
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/stage5_points'; OUT=ROOT/'outputs/stage5_points_v1'
PROTOCOL=ROOT/'configs/filtering/stage5_points_protocol.json'
RULE=ROOT/'configs/filtering/stage5_points_v1.json'
CONDITIONS=['blur5','normal5','reduced3','spaced3']


def source(scene):
    base=ROOT/'outputs/stage5_v1'/scene
    decision=read_json(base/'decision.json'); pred=read_json(base/'prediction.json')
    if decision['selected_method']!='vggt_raw':
        raise ValueError('not_supported: selected method has no compatible dense raw point grid')
    p=base/'selected_artifacts/raw_predictions.npz'
    assert sha256(p)==pred['artifacts']['pointcloud']['sha256']
    return p,pred


def generate(scene,fit=False):
    start=time.perf_counter(); protocol=read_json(PROTOCOL)
    p,pred=source(scene); rule=None if fit else read_json(RULE)
    dest=OUT/('dev_candidates' if fit else 'frozen')/scene
    if dest.exists(): raise ValueError('preserve existing output directory')
    dest.mkdir(parents=True)
    tolerances=sorted(set(c['relative_z_tolerance'] for c in protocol['candidates'])) if fit else [rule['relative_z_tolerance']]
    with np.load(p) as a:
        xyz=a['world_points']; shape=xyz.shape
        ev=evidence(xyz,a['depth'],a['extrinsics'],a['intrinsics'],tolerances,protocol['runtime']['batch_points'])
    np.savez_compressed(dest/'evidence.npz',finite=ev['finite'],observed=ev['observed'],support=ev['support'])
    rows=[]
    for i,c in enumerate(protocol['candidates'] if fit else [rule]):
        keep,unknown=classify(ev,tolerances.index(c['relative_z_tolerance']),c['min_support'])
        ids=np.flatnonzero(keep); name=f'candidate_{i}' if fit else 'filtered'
        np.savez_compressed(dest/(name+'.npz'),source_point_indices=ids,xyz=xyz.reshape(-1,3)[ids])
        rows.append({'candidate':i,'relative_z_tolerance':c['relative_z_tolerance'],'min_support':c['min_support'],
                     'points':len(keep),'kept':len(ids),'unknown':int(unknown.sum()),'invalid':int((~ev['finite']).sum()),
                     'observed_but_not_kept':int((ev['finite']&~unknown&~keep).sum()),'retention':float(keep.mean()),'artifact':name+'.npz','sha256':sha256(dest/(name+'.npz'))})
    if not fit:
        from georecon.selection.run import link_or_copy
        output=copy.deepcopy(pred)
        for a in output['artifacts'].values():
            if a is not None:
                src=ROOT/'outputs/stage5_v1'/scene/a['path']
                target=dest/a['path']
                if not target.exists(): link_or_copy(src,target)
        output['method_id']='scene_selector_point_filter_v1'
        output['config']={'selection':pred['config'],'point_filter':rule}
        output['provenance']['point_filter']={'protocol_sha256':sha256(PROTOCOL),'rule_sha256':sha256(RULE),'source_sha256':sha256(p),'gt_access':'none; runtime open guard active','camera_and_world_frame':'unchanged; subset only'}
        output['artifacts']['pointcloud']={'path':'filtered.npz','sha256':rows[0]['sha256'],'semantics':'unchanged source point-head XYZ subset, key xyz; source_point_indices maps C-order frame/y/x in original raw grid','arrays':{'xyz':[rows[0]['kept'],3],'source_point_indices':[rows[0]['kept']]}}
        output['timing_seconds']={'incremental_filter':time.perf_counter()-start}
        output['memory_bytes']={'peak_process_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_gpu_allocated':0}
        validate_result(output)
        write_json(dest/'prediction.json',output)
    write_json(dest/'run.json',{'scene_id':scene,'source':str(p.relative_to(ROOT)),'source_sha256':sha256(p),'prediction_sha256':sha256(ROOT/'outputs/stage5_v1'/scene/'prediction.json'),'source_shape':list(shape),'flattening':'C order: frame, native_y, native_x; integer native pixel centers',
               'rows':rows,'tolerances':tolerances,'protocol_sha256':sha256(PROTOCOL),'rule_sha256':None if fit else sha256(RULE),
               'wall_seconds':time.perf_counter()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'device':'CPU','gpu_allocated_bytes':0,'cameras_unchanged':True})
    print(scene,rows,flush=True)


def ply_points(path):
    # Official binary PLY vertex scalar records; refuse unknown layouts instead of guessing.
    types={'float':'f4','float32':'f4','double':'f8','uchar':'u1','uint8':'u1','char':'i1','int':'i4','uint':'u4','short':'i2','ushort':'u2'}
    with open(path,'rb') as f:
        assert f.readline().strip()==b'ply'
        props=[]; vertex=False; count=None; endian=None
        while True:
            line=f.readline().decode('ascii').strip(); parts=line.split()
            if not line: raise ValueError('truncated PLY')
            if parts[0]=='format':
                endian={'binary_little_endian':'<','binary_big_endian':'>'}.get(parts[1])
            elif parts[0]=='element':
                vertex=parts[1]=='vertex'
                if vertex: count=int(parts[2])
            elif parts[0]=='property' and vertex:
                if parts[1] not in types: raise ValueError('unsupported PLY vertex property')
                props.append((parts[2],types[parts[1]]))
            elif line=='end_header': break
        if endian is None or count is None: raise ValueError('expected binary vertex PLY')
        arr=np.fromfile(f,dtype=np.dtype([(n,endian+t) for n,t in props]),count=count)
        assert len(arr)==count
        return np.stack([arr[k] for k in ['x','y','z']],axis=1).astype(float)


def sample_ids(n,cap=200000):
    return np.linspace(0,n-1,min(n,cap),dtype=np.int64) if n else np.array([],dtype=np.int64)


class Evaluator:
    def __init__(self,scan):
        base=ROOT/'data/stage5_point_gt'/f'scan{scan}'
        self.provenance=read_json(base/'provenance.json')
        for row in self.provenance['files']: assert sha256(ROOT/row['local'])==row['sha256']
        gt=ply_points(base/f'stl{scan:03}_total.ply');gt=gt[sample_ids(len(gt))]
        self.tree=cKDTree(gt)
        plane=loadmat(base/f'Plane{scan}.mat')['P'].reshape(-1)
        self.recall_gt=gt[(gt@plane[:3]+plane[3])>0]
        self.mask=loadmat(base/f'ObsMask{scan}_10.mat')
        self.scan=scan

    def prepare(self,scene):
        p,pred=source(scene)
        truth=read_json(ROOT/'data/stage3_dtu'/scene.removeprefix('dtu_')/'reference.json')
        def centers(obj):
            return {i['image_id']:-np.array(i['camera']['R']).T@np.array(i['camera']['t']) for i in obj['images'] if i['registered']}
        a,b=centers(pred),centers(truth);keys=sorted(a)
        sim=align_points([a[k] for k in keys],[b[k] for k in keys])
        with np.load(p) as z: xyz=z['world_points'].reshape(-1,3)
        self.ids=sample_ids(len(xyz));self.xyz=sim.apply(xyz[self.ids]); self.n=len(xyz)
        mask=self.mask; bb=mask['BB'];res=float(mask['Res'].item());obs=mask['ObsMask']
        # MATLAB round is half away from zero, followed by its one-based index.
        q=(self.xyz-bb[0])/res
        safe=np.where(np.isfinite(q),q,-1e9)
        idx=(np.sign(safe)*np.floor(np.abs(safe)+.5)).astype(np.int64)
        inside=(idx>=0).all(1)&(idx<obs.shape).all(1)&np.isfinite(self.xyz).all(1)
        self.in_mask=np.zeros(len(idx),bool);self.in_mask[inside]=obs[tuple(idx[inside].T)].astype(bool)
        self.alignment={'scale':sim.scale,'R':sim.R.tolist(),'t':sim.t.tolist(),'source':'predicted and GT camera centers; evaluation only, no ICP'}

    def measure(self,keep):
        selected=keep[self.ids]&np.isfinite(self.xyz).all(1)
        precision_points=self.xyz[selected&self.in_mask]
        d=self.tree.query(precision_points,workers=8)[0] if len(precision_points) else np.array([])
        r=cKDTree(self.xyz[selected]).query(self.recall_gt,workers=8)[0] if selected.any() else np.full(len(self.recall_gt),np.inf)
        metrics={}
        for t in [1,2,5]:
            precision=float(np.mean(d<=t)) if len(d) else 0.
            recall=float(np.mean(r<=t)) if len(r) else 0.
            metrics[str(t)]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
        return {'retained_full_count':int(keep.sum()),'prediction_sample_count':int(selected.sum()),'precision_mask_count':len(d),'recall_reference_count':len(r),
                'mean_accuracy_mm':float(d.mean()) if len(d) else None,'mean_completeness_mm':float(r.mean()) if np.isfinite(r).all() and len(r) else None,'thresholds_mm':metrics,'empty_output':not bool(selected.any())}


def fit():
    from fit_stage5 import dev_only_guard
    files=[]
    def guard(event,args):
        dev_only_guard(event,args)
        if event=='open' and isinstance(args[0],str) and args[0].startswith(str(ROOT)):
            files.append(str(Path(args[0]).relative_to(ROOT)))
    sys.addaudithook(guard)
    if RULE.exists(): raise ValueError('rule already frozen')
    protocol=read_json(PROTOCOL);ev=Evaluator(1);rows=[]
    for cond in CONDITIONS:
        scene='dtu_scan1_'+cond;ev.prepare(scene)
        dest=OUT/'dev_candidates'/scene
        run=read_json(dest/'run.json');assert run['protocol_sha256']==sha256(PROTOCOL)
        row={'scene_id':scene,'before':ev.measure(np.ones(ev.n,bool)),'candidates':[],'alignment':ev.alignment}
        for c in run['rows']:
            keep=np.zeros(ev.n,bool)
            with np.load(dest/c['artifact']) as a: keep[a['source_point_indices']]=True
            row['candidates'].append(ev.measure(keep))
        rows.append(row)
    scores=[float(np.mean([r['candidates'][i]['thresholds_mm']['2']['f1'] for r in rows])) for i in range(len(protocol['candidates']))]
    best=int(np.argmax(scores));rule=dict(protocol['candidates'][best],protocol_sha256=sha256(PROTOCOL),fit_split='dev_only',candidate_index=best)
    write_json(RULE,rule)
    write_json(DOC/'fit_report.json',{'rows':rows,'macro_dev_f1_2mm':scores,'winner':best,'project_file_accesses':sorted(set(files)),'test_access_blocked_by_audit_hook':True,'gt_provenance':ev.provenance})
    print('frozen',rule,'scores',scores,flush=True)


def evaluate():
    rows=[]
    for scan in [1,6]:
        ev=Evaluator(scan)
        for cond in CONDITIONS:
            scene=f'dtu_scan{scan}_'+cond;ev.prepare(scene);dest=OUT/'frozen'/scene
            run=read_json(dest/'run.json');assert run['rule_sha256']==sha256(RULE)
            keep=np.zeros(ev.n,bool)
            with np.load(dest/'filtered.npz') as a:
                ids=a['source_point_indices'];keep[ids]=True
                with np.load(ROOT/run['source']) as z: assert np.array_equal(a['xyz'],z['world_points'].reshape(-1,3)[ids])
            rows.append({'scene_id':scene,'split':'dev' if scan==1 else 'test','before':ev.measure(np.ones(ev.n,bool)),'after':ev.measure(keep),'alignment':ev.alignment,'run':run})
    write_json(DOC/'comparison.json',{'rows':rows,'protocol_sha256':sha256(PROTOCOL),'rule_sha256':sha256(RULE),'test_rule_retuning':False})
    for r in rows: print(r['scene_id'],r['before']['thresholds_mm']['2'],r['after']['thresholds_mm']['2'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','fit','evaluate']);p.add_argument('--scene');p.add_argument('--dev-candidates',action='store_true');p.add_argument('--output-root',type=Path);args=p.parse_args()
    if args.output_root is not None: OUT=args.output_root.resolve()
    if args.action=='generate':
        from georecon.reliability.analyze import deny_ground_truth
        if args.dev_candidates:
            from fit_stage5 import dev_only_guard
            sys.addaudithook(dev_only_guard)
        sys.addaudithook(deny_ground_truth)
        accesses=[]
        def trace(event,values):
            if event=='open' and isinstance(values[0],str) and values[0].startswith(str(ROOT)):
                accesses.append(str(Path(values[0]).relative_to(ROOT)))
        sys.addaudithook(trace)
        generate(args.scene,args.dev_candidates)
        dest=OUT/('dev_candidates' if args.dev_candidates else 'frozen')/args.scene
        from georecon.provenance import code_provenance
        write_json(dest/'invocation.json',{'argv':sys.argv,'project_file_accesses':sorted(set(accesses)),'code':code_provenance('real'),'gt_guard_active':True,'numpy_version':np.__version__})
    elif args.action=='fit': fit()
    else: evaluate()
