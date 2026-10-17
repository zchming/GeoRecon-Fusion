import unittest
import numpy as np
try:
    from georecon.teaching.bundle_adjustment import prepare,decode,residuals,sparsity,solve
    from scipy.spatial.transform import Rotation
except ImportError:prepare=None

@unittest.skipIf(prepare is None,'SciPy optimization extra not installed')
class TeachingBATests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4);self.x=rng.uniform([-1,-1,4],[1,1,7],(35,3));self.r=Rotation.from_rotvec([[0,0,0],[0,.05,0],[.01,-.03,.01]]).as_matrix();self.c=np.array([[0.,0,0],[1.,.01,0],[.4,.7,.02]]);self.t=-np.einsum('nij,nj->ni',self.r,self.c);self.k=np.tile([[700.,2,320],[0,710,240],[0,0,1.]],(3,1,1));self.f=np.repeat(np.arange(3),len(self.x));self.j=np.tile(np.arange(len(self.x)),3)
        camera=np.einsum('nij,nj->ni',self.r[self.f],self.x[self.j])+self.t[self.f];q=np.einsum('nij,nj->ni',self.k[self.f],camera);self.uv=q[:,:2]/q[:,2:3]
        self.cfg={'min_points':10,'max_points':3000,'max_observations':20000,'loss_scale_px':1.,'max_nfev':80,'wall_budget_seconds':30.,'ftol':1e-8,'xtol':1e-8,'gtol':1e-8}
    def problem(self,**kw):
        args=dict(rotations=self.r,translations=self.t,intrinsics=self.k,xyz=self.x,frames=self.f,point_ids=self.j,pixels=self.uv,seed=[0,1]);args.update(kw);return prepare(**args)
    def test_joint_recovery_and_gauge(self):
        rng=np.random.default_rng(5);r=self.r.copy();r[1:]=Rotation.from_rotvec([[.01,0,0],[0,.01,0]]).as_matrix()@r[1:];c=self.c.copy();c[2]+=[.03,-.02,.01];t=-np.einsum('nij,nj->ni',r,c)
        p=self.problem(rotations=r,translations=t,xyz=self.x+rng.normal(0,.02,self.x.shape));d,info=solve(p,self.cfg);self.assertTrue(info['accepted']);self.assertLess(info['final_rmse_px'],info['initial_rmse_px']*.02);rr,tt,xx=decode(p,d);np.testing.assert_array_equal(rr[0],r[0]);np.testing.assert_array_equal(tt[0],t[0]);centers=-np.einsum('nji,nj->ni',rr,tt);self.assertAlmostEqual(centers[1,p['anchor_axis']],p['C'][1,p['anchor_axis']],places=12);self.assertLess(np.mean(np.linalg.norm(xx-self.x,axis=1)),.001)
    def test_jacobian_sparsity_covers_numeric_derivatives(self):
        p=self.problem();base=np.zeros(p['size']);pattern=sparsity(p).toarray().astype(bool);v=residuals(p,base)[0]
        for col in range(p['size']):
            q=base.copy();q[col]=1e-7;changed=np.abs(residuals(p,q)[0]-v)>1e-8;self.assertFalse(np.any(changed&~pattern[:,col]))
    def test_collinear_rejected(self):
        x=np.c_[np.linspace(0,1,len(self.x)),np.zeros(len(self.x)),np.full(len(self.x),5.)];_,s=solve(self.problem(xyz=x),self.cfg);self.assertEqual(s['reason'],'collinear_structure')
    def test_zero_baseline_rejected(self):
        t=self.t.copy();t[1]=0;_,s=solve(self.problem(translations=t),self.cfg);self.assertEqual(s['reason'],'zero_seed_baseline')
    def test_wall_budget_is_failure_with_preserved_candidate(self):
        p=self.problem();d,s=solve(p,{**self.cfg,'wall_budget_seconds':0});self.assertFalse(s['accepted']);self.assertEqual(s['reason'],'wall_budget_exceeded');np.testing.assert_array_equal(d,0)
    def test_duplicate_observations_rejected(self):
        with self.assertRaises(ValueError):self.problem(frames=np.r_[self.f,0],point_ids=np.r_[self.j,0],pixels=np.vstack([self.uv,self.uv[0]]))
    def test_single_view_points_rejected(self):
        mask=(self.j!=0)|(self.f==0);_,s=solve(self.problem(frames=self.f[mask],point_ids=self.j[mask],pixels=self.uv[mask]),self.cfg);self.assertEqual(s['reason'],'point_has_fewer_than_two_views')
    def test_underconstrained_partial_rejected(self):
        mask=self.f<2;p=prepare(self.r[:2],self.t[:2],self.k[:2],self.x,self.f[mask],self.j[mask],self.uv[mask],[0,1]);_,s=solve(p,self.cfg);self.assertEqual(s['reason'],'requires_at_least_three_registered_cameras')
if __name__=='__main__':unittest.main()
