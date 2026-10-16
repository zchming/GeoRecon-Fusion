import unittest
import numpy as np
try:
    from georecon.optimization.joint import decode,terms,sparsity,objective,solve,scatter
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'SciPy joint optimization dependency unavailable')
class JointOptimizationTest(unittest.TestCase):
    def fixture(self):
        from pathlib import Path
        import json
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/optimization/stage5_protocol.json').read_text());cfg['fusion_strength']=.5
        n=3;points=np.array([[-.3,-.2,3],[.3,-.2,3.2],[-.3,.2,3.1],[.3,.2,3.3]])
        r=np.broadcast_to(np.eye(3),(n,3,3)).copy();t=np.array([[0.,0,0],[-.3,0,0],[0,-.3,0]])
        k=np.broadcast_to(np.array([[200.,0,100],[0,200,100],[0,0,1]]),(n,3,3)).copy()
        f=np.repeat(np.arange(n),len(points));j=np.tile(np.arange(len(points)),n)
        target_t=t.copy();target_t[1,0]+=.001;target_t[2,1]-=.001
        c=points[j]+target_t[f];q=np.einsum('nij,nj->ni',k[f],c);uv=q[:,:2]/q[:,2:3]
        p={'ids':np.arange(4),'point_ids':np.arange(4),'inverse':np.arange(4),'raw_prior':points.copy(),'raw_weight':np.ones(4)*2,
           'anchor_prior':points.copy(),'classical_weight':np.ones(4),'x0':points+np.array([.002,-.001,.003]),'R0':r,'t0':t,'K':k,'obs_frame':f,'obs_point':j,'uv':uv,'normalization':3.}
        return p,cfg

    def test_joint_solve_reduces_cost_and_changes_pose_and_points(self):
        p,c=self.fixture();solution,record=solve(p,c,1)
        self.assertTrue(record['accepted'],record)
        self.assertLess(record['final_objective'],record['initial_objective'])
        self.assertGreater(np.linalg.norm(solution['pose_delta'][1:]),1e-8)
        self.assertGreater(np.linalg.norm(solution['points']-p['x0']),1e-8)
        np.testing.assert_array_equal(solution['R'][0],p['R0'][0]);np.testing.assert_array_equal(solution['t'][0],p['t0'][0])

    def test_sparse_jacobian_covers_numerical_dependencies(self):
        p,c=self.fixture();pattern=sparsity(p).toarray().astype(bool)
        x=np.zeros(pattern.shape[1]);r=np.concatenate(terms(p,x,c,1)[0])
        for col in range(len(x)):
            y=x.copy();y[col]=1e-6;delta=np.concatenate(terms(p,y,c,1)[0])-r
            self.assertFalse(np.any((np.abs(delta)>1e-9)&~pattern[:,col]))

    def test_bounds_and_first_pose_gauge(self):
        p,c=self.fixture();solution,record=solve(p,c,10)
        self.assertTrue(record['accepted'])
        d=solution['pose_delta'];self.assertLessEqual(np.max(np.abs(d[:,:3])),np.radians(c['solver']['pose_rotation_component_bound_deg'])+1e-12)
        self.assertLessEqual(np.max(np.abs(d[:,3:])),c['solver']['pose_translation_component_bound_over_depth']+1e-12)
        self.assertTrue(np.all(d[0]==0))

    def test_solver_budget_failure_keeps_attempt_for_caller(self):
        p,c=self.fixture();c['solver']['max_nfev']=1
        solution,record=solve(p,c,1)
        self.assertFalse(record['accepted']);self.assertEqual(record['status'],0);self.assertIsNotNone(solution)

    def test_no_or_excess_constraints_do_not_run(self):
        p,c=self.fixture();c['solver']['max_unique_points']=3
        solution,record=solve(p,c,1)
        self.assertIsNone(solution);self.assertFalse(record['accepted'])
        c['solver']['max_unique_points']=2000;p['obs_point']=p['obs_point'][:2];p['obs_frame']=p['obs_frame'][:2];p['uv']=p['uv'][:2]
        self.assertIsNone(solve(p,c,1)[0])

    def test_scatter_changes_only_allowed_grid_indices_and_preserves_dtype(self):
        before=np.arange(30,dtype=np.float32).reshape(-1,3);p={'ids':np.array([2,7]),'inverse':np.array([0,0])}
        out=scatter(before,p,np.array([[.1,.2,.3]]));mask=np.ones(len(before),bool);mask[p['ids']]=False
        self.assertEqual(out.dtype,before.dtype);self.assertEqual(out[mask].tobytes(),before[mask].tobytes())
        np.testing.assert_array_equal(out[2],out[7]);np.testing.assert_array_equal(scatter(before,None,None),before)

    def test_normalized_objective_matches_soft_l1_definition(self):
        x=np.array([0.,1.,3.]);self.assertAlmostEqual(objective(x),float(np.sum(np.sqrt(1+x*x)-1)))

    def test_zero_parameters_preserve_baseline_projection(self):
        p,c=self.fixture();x=np.zeros(12+12);r,t,pts,d=decode(p,x)
        np.testing.assert_array_equal(r,p['R0']);np.testing.assert_array_equal(t,p['t0']);np.testing.assert_array_equal(pts,p['x0'])

if __name__=='__main__':unittest.main()
