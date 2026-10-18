"""Frozen one-factor profiles and failure-aware improvement notices."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from georecon.matching.backend_control import reconstruct_new,usac_parameters
from georecon.matching.msac_refinement import scorecards

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/experiments/stage7a_msac_refinement_v1.json').read_text())
BACKEND=json.loads((ROOT/'configs/teaching/stage6_two_view.json').read_text())
CALLS=0


class RefinementPlanTests(unittest.TestCase):
    def test_each_candidate_changes_exactly_one_effort_control(self):
        changes={'sample50':('loSampleSize',50),'lo50':('loIterations',50),'iter50k':('maxIterations',50000)}
        for arm,(key,value) in changes.items():
            diff={k for k in CFG['profiles']['base'] if CFG['profiles'][arm][k]!=CFG['profiles']['base'][k]}
            self.assertEqual(diff,{key});self.assertEqual(CFG['profiles'][arm][key],value)
        self.assertEqual(CFG['profiles']['base']['threshold'],1.5)
        self.assertEqual(CFG['profiles']['base']['sampler'],'SAMPLING_UNIFORM')

    def fixture(self):
        cases=[{'case_id':str(s),'scan':s,'conditions':['normal5','reduced3','spaced3','blur5']} for s in [1,6]]
        cfg={**CFG,'seeds':[42,43]};rows=[]
        for c in cases:
            for arm in cfg['profiles']:
                for seed in cfg['seeds']:
                    error=1. if arm=='sample50' else 4. if seed==42 else 8.
                    accepted=arm!='lo50'
                    rows.append({'case_id':c['case_id'],'scan':c['scan'],'conditions':c['conditions'],'arm':arm,'seed':seed,
                        'status':'success' if accepted else 'no_reconstruction','geometry_seconds':.01,
                        'pose':{'accepted':accepted,'correct':accepted and error<=5,'false_accept':accepted and error>5,
                                'joint_deg':error if accepted else None,'auc':{str(t):max(0,1-error/t) if accepted else 0 for t in [5,10,20]}}})
        return rows,cases,cfg

    def test_notices_describe_gain_without_changing_priority(self):
        rows,cases,cfg=self.fixture();r=scorecards(rows,cases,cfg)
        self.assertTrue(r['notices']['sample50']['mean'])
        self.assertTrue(r['notices']['sample50']['variation'])
        self.assertFalse(r['notices']['lo50']['false_acceptance'])
        self.assertIn('original',r['priority_candidate'])
        self.assertTrue(any(c['transitions'] for c in r['contrasts']))
        self.assertFalse(r['unseen_objects_tested'])

    def test_missing_duplicate_and_rejection_denominators(self):
        rows,cases,cfg=self.fixture()
        with self.assertRaises(ValueError):scorecards(rows[:-1],cases,cfg)
        with self.assertRaises(ValueError):scorecards(rows+[rows[0]],cases,cfg)
        r=scorecards(rows,cases,cfg)
        self.assertTrue(all(c['macro']['auc10']['mean']==0 and c['accepted_events']==0 for c in r['cards'] if c['arm']=='lo50'))


@unittest.skipUnless(importlib.util.find_spec('cv2') is not None,'OpenCV required for effort profile contracts')
class RefinementRuntimeTests(unittest.TestCase):
    def test_actual_fields_synthetic_pose_and_shared_H(self):
        global CALLS
        from test_stage7a_backend_control import scene
        p,q,k1,k2,r,t=scene();values=[]
        for arm,spec in CFG['profiles'].items():
            params=usac_parameters(spec,42)
            self.assertEqual(params.loSampleSize,spec['loSampleSize'])
            self.assertEqual(params.loIterations,spec['loIterations'])
            self.assertEqual(params.maxIterations,spec['maxIterations'])
            CALLS+=1;v=reconstruct_new(p,q,k1,k2,BACKEND,spec);values.append(v)
            self.assertEqual(v['status'],'success',(arm,v['reason']))
            np.testing.assert_allclose(v['R'],r,atol=2e-4,rtol=0)
            self.assertGreater(np.dot(v['t'],t/np.linalg.norm(t)),.9999)
            np.testing.assert_array_equal(v['arrays']['H'],values[0]['arrays']['H'])
            np.testing.assert_array_equal(v['arrays']['homography_mask'],values[0]['arrays']['homography_mask'])

    def test_invalid_and_few_inputs_preserve_failure_and_callable(self):
        global CALLS
        import cv2
        from test_stage7a_backend_control import scene
        p,q,k1,k2,_,_=scene();native=cv2.findEssentialMat
        CALLS+=1;v=reconstruct_new(p[:4],q[:4],k1,k2,BACKEND,CFG['profiles']['sample50'])
        self.assertEqual(v['reason'],'insufficient_matches');self.assertIs(cv2.findEssentialMat,native)
        with self.assertRaises(ValueError):reconstruct_new(p*np.nan,q,k1,k2,BACKEND,CFG['profiles']['iter50k'])
        self.assertIs(cv2.findEssentialMat,native)


    def test_additional_local_optimizer_families_on_same_synthetic_observations(self):
        global CALLS
        from test_stage7a_backend_control import scene
        cfg=json.loads((ROOT/'configs/experiments/stage7a_msac_refinement_v2.json').read_text())
        p,q,k1,k2,r,t=scene()
        for name in ['inner','none','graphcut']:
            spec=cfg['profiles'][name]
            self.assertEqual({k for k in spec if spec[k]!=cfg['profiles']['base'][k]},{'loMethod'})
            CALLS+=1;v=reconstruct_new(p,q,k1,k2,BACKEND,spec)
            self.assertEqual(v['status'],'success',(name,v['reason']))
            np.testing.assert_allclose(v['R'],r,atol=2e-4,rtol=0)
            self.assertGreater(np.dot(v['t'],t/np.linalg.norm(t)),.9999)



    def test_new_E_noise_profiles_do_not_change_H_threshold(self):
        global CALLS
        from test_stage7a_backend_control import scene
        cfg=json.loads((ROOT/'configs/experiments/stage7a_msac_refinement_v3.json').read_text())
        p,q,k1,k2,r,t=scene();Hs=[]
        for name in ['noise075','noise1','noise2']:
            spec=cfg['profiles'][name]
            self.assertEqual({k for k in spec if spec[k]!=cfg['profiles']['base'][k]},{'threshold'})
            self.assertEqual(usac_parameters(spec,42).threshold,spec['threshold'])
            CALLS+=1;v=reconstruct_new(p,q,k1,k2,BACKEND,spec)
            self.assertEqual(v['status'],'success',(name,v['reason']))
            Hs.append(v['arrays']['H'])
            np.testing.assert_array_equal(Hs[-1],Hs[0])
            self.assertEqual(BACKEND['essential_ransac_px'],1.5)


if __name__=='__main__':unittest.main()


class DevelopmentSelectionTests(unittest.TestCase):
    def test_selection_uses_development_rows_only_and_keeps_rejection_cost(self):
        from georecon.matching.msac_refinement import choose_development
        cfg=json.loads((ROOT/'configs/experiments/stage7a_msac_refinement_v3.json').read_text())
        cfg={**cfg,'seeds':[42,43]};rows=[]
        for scan in [1,6]:
            for i in range(23):
                for arm in cfg['profiles']:
                    for seed in cfg['seeds']:
                        error=4. if arm=='base' else 1. if arm=='noise1' else 30.
                        rows.append({'case_id':str(i),'scan':scan,'conditions':['normal5','reduced3','spaced3','blur5'],
                            'arm':arm,'seed':seed,'pose':{'GT_pose_defined':True,'correct':error<=5,'accepted':True,
                            'false_accept':error>5,'joint_deg':error,'auc':{'10':max(0,1-error/10)}}})
        choice=choose_development(rows,cfg)
        self.assertEqual(choice['selected_arm'],'noise1');self.assertFalse(choice['uses_validation_GT'])
        for row in rows:
            if row['scan']==6:row['pose'].update(correct=False,false_accept=True,joint_deg=179.,auc={'10':0.})
        self.assertEqual(choice,choose_development(rows,cfg))
        with self.assertRaises(ValueError):choose_development(rows[1:],cfg)
