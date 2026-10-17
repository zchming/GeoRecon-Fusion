"""Seeded synthetic BA failure fixtures; separate from real DTU results."""
from pathlib import Path
import numpy as np
from georecon.data.protocol import read_json,write_json,sha256
from georecon.teaching.bundle_adjustment import prepare,solve,decode
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage6_ba';OUT=ROOT/'outputs/stage6_ba_controls_v1'
def controls():
    if OUT.exists():raise ValueError('preserve synthetic attempts')
    OUT.mkdir(parents=True);cfg=read_json(ROOT/'configs/teaching/stage6_ba.json');rng=np.random.default_rng(42);m=40
    x=rng.uniform([-1,-1,4],[1,1,7],(m,3));r=np.tile(np.eye(3),(3,1,1));t=np.array([[0.,0,0],[-1,0,0],[-.4,-.5,0]]);k=np.tile([[700.,2,320],[0,710,240],[0,0,1.]],(3,1,1));f=np.repeat(np.arange(3),m);j=np.tile(np.arange(m),3);rows=[]
    for name in ['collinear','zero_baseline','disconnected','forced_wall_budget']:
        xx=x.copy();tt=t.copy();ff=f.copy();jj=j.copy();settings=cfg.copy()
        if name=='collinear':xx=np.c_[np.linspace(-1,1,m),np.zeros(m),np.full(m,5.)]
        if name=='zero_baseline':tt[1]=0
        if name=='disconnected':mask=ff<2;ff=ff[mask];jj=jj[mask]
        if name=='forced_wall_budget':settings['wall_budget_seconds']=0.
        cam=xx[jj]+tt[ff];q=np.einsum('nij,nj->ni',k[ff],cam);uv=q[:,:2]/q[:,2:3]
        problem=prepare(r,tt,k,xx,ff,jj,uv,[0,1]);delta,stats=solve(problem,settings);rr,trans,points=decode(problem,delta)
        expected={'collinear':'collinear_structure','zero_baseline':'zero_seed_baseline','disconnected':'disconnected_observation_graph','forced_wall_budget':'wall_budget_exceeded'}[name];assert not stats['accepted'] and stats['reason']==expected
        np.savez_compressed(OUT/(name+'.npz'),input_R=r,input_t=tt,K=k,xyz=xx,frames=ff,point_ids=jj,pixels=uv,candidate_delta=delta,candidate_R=rr,candidate_t=trans,candidate_xyz=points)
        rows.append({'case_id':name,'kind':'synthetic','expected_reason':expected,'stats':stats,'actual_config':settings,'explicit_profile_change':'wall_budget_seconds=0 only for deliberate timeout control' if name=='forced_wall_budget' else None,'artifact_sha256':sha256(OUT/(name+'.npz'))})
    write_json(DOC/'synthetic_controls.json',rows)
if __name__=='__main__':controls()
