"""Contracts for the frozen E backend contrast and failure-aware decision."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from georecon.matching.backend_control import (ARMS, usac_parameters, replace_essential, reconstruct_new,
                                               compare_legacy, summarize, decision)
from georecon.teaching.two_view import reconstruct, normalize

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/experiments/stage7a_backend_control_v1.json').read_text())
BACKEND=json.loads((ROOT/'configs/teaching/stage6_two_view.json').read_text())
CALLS=0
OPENCV=importlib.util.find_spec('cv2') is not None


def scene():
    rng=np.random.default_rng(192)
    xyz=np.c_[rng.uniform(-1,1,(100,2)),rng.uniform(4,8,100)]
    k1=np.array([[700.,3,400],[0,680,300],[0,0,1]])
    k2=np.array([[720.,-2,390],[0,710,310],[0,0,1]])
    theta=.08;r=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    t=np.array([-.8,0,.02])
    def project(x,k):
        h=x@k.T;return h[:,:2]/h[:,2:3]
    return project(xyz,k1),project(xyz@r.T+t,k2),k1,k2,r,t


def fit(p,q,k1,k2,seed=42):
    global CALLS
    CALLS+=1
    return reconstruct_new(p,q,k1,k2,dict(BACKEND,seed=seed),CFG['backend']['new']['params'])


@unittest.skipUnless(OPENCV,'OpenCV required for runtime USAC contracts')
class BackendContracts(unittest.TestCase):
    def test_explicit_parameters_and_seed_are_not_global_rng(self):
        import cv2
        for seed in [42,43,44]:
            p=usac_parameters(CFG['backend']['new']['params'],seed)
            self.assertEqual(p.score,cv2.SCORE_METHOD_MAGSAC)
            self.assertEqual(p.loMethod,cv2.LOCAL_OPTIM_SIGMA)
            self.assertEqual(p.randomGeneratorState,seed)
            self.assertEqual(p.threshold,1.5)
            self.assertFalse(p.isParallel)

    def test_pixel_K_and_threshold_forwarding_without_fitting(self):
        import cv2
        p,q,k1,k2,_,_=scene();seen=[]
        def native(*args):
            seen.append(args);return np.eye(3),np.ones((len(p),1),np.uint8)
        with patch.object(cv2,'findEssentialMat',native):
            with replace_essential(p,q,k1,k2,BACKEND,CFG['backend']['new']['params']):
                e,m=cv2.findEssentialMat(normalize(p,k1),normalize(q,k2),np.eye(3),method=cv2.RANSAC,
                    prob=.999,threshold=1.5/np.mean([k1[0,0],k1[1,1],k2[0,0],k2[1,1]]),maxIters=10000)
            self.assertIs(cv2.findEssentialMat,native)
        args=seen[0]
        for a,b in zip(args[:4],[p,q,k1,k2]):np.testing.assert_array_equal(a,b)
        self.assertEqual(args[4:6],(None,None))
        self.assertEqual(args[6].threshold,1.5)
        self.assertEqual(m.shape,(100,1))

    def test_runtime_essential_with_skew_different_K_all_seeds(self):
        p,q,k1,k2,r,t=scene()
        for seed in [42,43,44]:
            result=fit(p,q,k1,k2,seed)
            self.assertEqual(result['status'],'success',result['reason'])
            self.assertEqual(result['essential_calls'],1)
            np.testing.assert_allclose(result['R'],r,atol=2e-4,rtol=0)
            self.assertGreater(np.dot(result['t'],t/np.linalg.norm(t)),.9999)
            self.assertEqual(result['arrays']['essential_mask'].shape,(100,))
            self.assertAlmostEqual(result['stats']['normalized_ransac_threshold'],1.5/702.5)

    def test_same_seed_is_independent_of_global_cv_rng(self):
        import cv2
        p,q,k1,k2,_,_=scene()
        q=q.copy();q[::5]+=[10.,-7.]
        cv2.setRNGSeed(901);first=fit(p,q,k1,k2)
        cv2.setRNGSeed(1701);second=fit(p,q,k1,k2)
        self.assertEqual(first['status'],second['status'])
        np.testing.assert_array_equal(first['arrays']['essential_mask'],second['arrays']['essential_mask'])
        np.testing.assert_array_equal(first['arrays']['essential_candidates'],second['arrays']['essential_candidates'])

    def test_homography_is_unchanged_and_original_callable_restored(self):
        import cv2
        p,q,k1,k2,_,_=scene();native=cv2.findEssentialMat
        old=reconstruct(p,q,k1,k2,BACKEND);new=fit(p,q,k1,k2)
        self.assertIs(cv2.findEssentialMat,native)
        np.testing.assert_array_equal(old['arrays']['H'],new['arrays']['H'])
        np.testing.assert_array_equal(old['arrays']['homography_mask'],new['arrays']['homography_mask'])

    def test_too_few_and_identical_inputs_keep_original_rejections(self):
        p,q,k1,k2,_,_=scene()
        for a,b,k,reason in [(p[:4],q[:4],k2,'insufficient_matches'),(p,p,k1,'identical_correspondences_no_baseline')]:
            result=fit(a,b,k1,k)
            self.assertEqual(result['reason'],reason)
            self.assertEqual(result['essential_calls'],0)
        with self.assertRaises(ValueError):fit(p*np.nan,q,k1,k2)

    def test_exception_restores_backend_and_does_not_fallback(self):
        import cv2
        p,q,k1,k2,_,_=scene()
        def unavailable(*a,**kw):raise RuntimeError('USAC unavailable')
        with patch.object(cv2,'findEssentialMat',unavailable):
            with self.assertRaisesRegex(RuntimeError,'USAC unavailable'):fit(p,q,k1,k2)
            self.assertIs(cv2.findEssentialMat,unavailable)


class AuditAndMetrics(unittest.TestCase):
    def test_reproduction_detects_failure_mask_and_pose_changes(self):
        historical={'status':'success','reason':None,'statistics':{'matches':20,'essential_inliers':20,'homography_inliers':2,'kept_points':20}}
        value={'status':'success','reason':None,'stats':historical['statistics']}
        arrays={'essential_mask':np.ones(20,bool),'accepted_R':np.eye(3),'accepted_t':np.array([1.,0,0])}
        self.assertEqual(compare_legacy(value,arrays,historical,arrays),[])
        changed={**arrays,'essential_mask':np.zeros(20,bool),'accepted_t':np.array([-1.,0,0])}
        self.assertEqual(set(compare_legacy(value,changed,historical,arrays)),{'arrays.essential_mask','accepted_t'})
        self.assertIn('status',compare_legacy(dict(value,status='no_reconstruction'),arrays,historical,arrays))

    def fixture(self):
        cases=[{'case_id':f's{s}_{i}','scan':s,'conditions':['normal5','reduced3','spaced3','blur5']} for s in [1,6] for i in [0,1]]
        seeds=[42,43,44];rows=[]
        for c in cases:
            for a in ARMS:
                good=a=='C_new' or c['case_id'].endswith('_0')
                e=1. if good else 8.
                for seed in seeds:
                    rows.append({'case_id':c['case_id'],'arm':a,'seed':seed,'geometry_seconds':.01,
                        'pose':{'accepted':True,'correct':good,'false_accept':not good,'GT_pose_defined':True,
                                'joint_deg':e,'auc':{str(t):max(0,1-e/t) for t in [5,10,20]}}})
        return cases,seeds,rows

    def test_gates_need_both_objects_and_protect_normal_pairs(self):
        cases,seeds,rows=self.fixture()
        def verdict():return decision(rows,cases,seeds,summarize(rows,cases,seeds),True)
        self.assertEqual(verdict()['verdict'],'GO_FEASIBILITY')
        for r in rows:
            if r['case_id']=='s6_0' and r['arm']=='C_new':
                r['pose'].update(correct=False,false_accept=True,joint_deg=30.,auc={str(t):0 for t in [5,10,20]})
        result=verdict()
        self.assertEqual(result['verdict'],'NO_GO_EVIDENCE')
        self.assertFalse(result['gates']['G2'])
        self.assertTrue(result['severe_new'])
        self.assertTrue(result['normal_regressions'])

    def test_unknown_failure_denominators_and_blocked_priority(self):
        cases,seeds,rows=self.fixture()
        sums=summarize(rows,cases,seeds)
        self.assertEqual(decision(rows,cases,seeds,sums,False)['verdict'],'BLOCKED')
        self.assertEqual(decision(rows,cases,seeds,sums,True,True)['verdict'],'NO_GO_RESOURCE')
        self.assertEqual(decision(rows,cases,seeds,sums,True,False,True)['verdict'],'INCONCLUSIVE')
        rows[0]['pose']['GT_pose_defined']=False
        self.assertEqual(decision(rows,cases,seeds,sums,True)['verdict'],'INCONCLUSIVE')
        with self.assertRaises(ValueError):decision(rows[:-1],cases,seeds,sums,True)

    def test_guard_rejects_truth_models_old_predictions_and_network(self):
        spec=importlib.util.spec_from_file_location('backend_runner',ROOT/'scripts/run_stage7a_backend_control.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        for name in ['/x/reference.json','/x/evaluation.json','/x/stage7a_v1_after_guard_fix/main/a/geometry_42.npz','/x/model.pth']:
            with self.assertRaises(PermissionError):runner.guard('open',(name,'r',0))
        with self.assertRaises(PermissionError):runner.guard('import',('torch.cuda',))
        with self.assertRaises(PermissionError):runner.guard('socket.connect',())
        runner.guard('open',('/new/inputs/synthetic.npz','r',0))


if __name__=='__main__':unittest.main()
