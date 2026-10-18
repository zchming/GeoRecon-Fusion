"""Cached-rank permutation, mask restoration and unchanged downstream contracts."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from georecon.matching.msac_sampling import rank_order,restore_mask,reconstruct_sampling,scorecards

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/experiments/stage7a_msac_sampling_v1.json').read_text())
BACKEND=json.loads((ROOT/'configs/teaching/stage6_two_view.json').read_text())
CALLS=0


class RankingContracts(unittest.TestCase):
    def test_descending_quality_ties_resolve_by_indices_not_input_position(self):
        scores=np.array([.5,.9,.5,.5]);ids=np.array([[2,4],[9,1],[1,3],[1,2]])
        np.testing.assert_array_equal(rank_order(scores,ids,'score'),[1,3,2,0])
        np.testing.assert_array_equal(rank_order(scores,ids,'lex'),np.arange(4))
        np.testing.assert_array_equal(restore_mask(np.array([1,0,1,0]),np.array([1,3,2,0])).ravel(),[0,1,1,0])

    def test_bad_score_or_mask_is_rejected_without_silent_filter(self):
        ids=np.array([[0,0],[1,1]])
        for scores in [[.2,np.nan],[.2,1.1],[.2,-.1],[.2]]:
            with self.assertRaises(ValueError):rank_order(scores,ids,'score')
        for mask,order in [(np.ones(2),[0,0]),(np.ones(1),[1,0]),([2,0],[0,1])]:
            with self.assertRaises(ValueError):restore_mask(mask,np.array(order))
        self.assertIsNone(restore_mask(None,np.array([0,1])))
        self.assertEqual(rank_order(np.empty(0),np.empty((0,2),int),'score').size,0)

    def test_scorecard_reports_tradeoffs_without_automatic_priority_change(self):
        cases=[{'case_id':str(s),'scan':s,'conditions':['normal5','reduced3','spaced3','blur5']} for s in [1,6]]
        cfg={**CFG,'seeds':[42,43]};rows=[]
        for c in cases:
            for arm in cfg['arms']:
                for seed in cfg['seeds']:
                    error=1. if arm=='P_score' and seed==42 else 30. if arm=='P_score' else 4.
                    rows.append({'case_id':c['case_id'],'scan':c['scan'],'conditions':c['conditions'],'arm':arm,'seed':seed,
                        'status':'success','geometry_seconds':.01,'pose':{'correct':error<=5,'accepted':True,'false_accept':error>5,'joint_deg':error,
                                                                        'auc':{str(t):max(0,1-error/t) for t in [5,10,20]}}})
        result=scorecards(rows,cases,cfg)
        self.assertFalse(result['notices']['accuracy'])
        self.assertTrue(any(r['severe_events'] for r in result['cards']))
        self.assertTrue(any(r['transitions'] for r in result['contrasts']))
        self.assertIn('retained',result['priority_candidate'])
        with self.assertRaises(ValueError):scorecards(rows[:-1],cases,cfg)


@unittest.skipUnless(importlib.util.find_spec('cv2') is not None,'OpenCV required for ranked E runtime contracts')
class SamplingRuntimeContracts(unittest.TestCase):
    def test_four_cells_synthetic_pose_mask_mapping_and_H_invariance(self):
        global CALLS
        from test_stage7a_backend_control import scene
        p,q,k1,k2,r,t=scene();rng=np.random.default_rng(99)
        scores=rng.uniform(.1,.9,len(p));ids=np.c_[np.arange(len(p)),np.arange(len(p))]
        values=[]
        for name,cell in CFG['arms'].items():
            CALLS+=1;v=reconstruct_sampling(p,q,k1,k2,BACKEND,CFG['base_params'],cell,scores,ids);values.append(v)
            self.assertEqual(v['status'],'success',(name,v['reason']))
            np.testing.assert_allclose(v['R'],r,atol=2e-4,rtol=0)
            self.assertGreater(np.dot(v['t'],t/np.linalg.norm(t)),.9999)
            a=v['arrays'];np.testing.assert_array_equal(a['pixels_first'],p)
            np.testing.assert_array_equal(a['essential_mask'],restore_mask(a['estimation_native_mask'],a['estimation_order']).ravel().astype(bool))
            np.testing.assert_array_equal(a['homography_mask'],values[0]['arrays']['homography_mask'])
            np.testing.assert_array_equal(a['H'],values[0]['arrays']['H'])

    def test_insufficient_matches_early_return_and_native_function_restoration(self):
        global CALLS
        import cv2
        from test_stage7a_backend_control import scene
        p,q,k1,k2,_,_=scene();native=cv2.findEssentialMat
        CALLS+=1;v=reconstruct_sampling(p[:4],q[:4],k1,k2,BACKEND,CFG['base_params'],CFG['arms']['P_score'],np.ones(4),np.c_[np.arange(4),np.arange(4)])
        self.assertEqual(v['reason'],'insufficient_matches');self.assertIs(cv2.findEssentialMat,native)
        with self.assertRaises(ValueError):reconstruct_sampling(p*np.nan,q,k1,k2,BACKEND,CFG['base_params'],CFG['arms']['P_score'],np.ones(len(p)),np.c_[np.arange(len(p)),np.arange(len(p))])
        self.assertIs(cv2.findEssentialMat,native)


if __name__=='__main__':unittest.main()
