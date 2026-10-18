"""Runtime profile contracts and transparent gain/regression scorecards."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from georecon.matching.backend_stability import reconstruct_profile,scorecards
from georecon.matching.backend_control import usac_parameters

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/experiments/stage7a_backend_stability_v1.json').read_text())
BACKEND=json.loads((ROOT/'configs/teaching/stage6_two_view.json').read_text())
CALLS=0
CV=importlib.util.find_spec('cv2') is not None


@unittest.skipUnless(CV,'OpenCV required for MSAC runtime contracts')
class ProfileContracts(unittest.TestCase):
    def test_msac_profile_fields_and_seed(self):
        import cv2
        p=usac_parameters(CFG['profiles']['msac'],51)
        self.assertEqual(p.score,cv2.SCORE_METHOD_MSAC)
        self.assertEqual(p.loMethod,cv2.LOCAL_OPTIM_INNER_AND_ITER_LO)
        self.assertEqual((p.loSampleSize,p.loIterations,p.randomGeneratorState),(12,20,51))

    def test_runtime_msac_skew_different_intrinsics_new_seed_range(self):
        global CALLS
        from test_stage7a_backend_control import scene
        p,q,k1,k2,r,t=scene()
        for seed in [42,45,51]:
            CALLS+=1
            v=reconstruct_profile(p,q,k1,k2,dict(BACKEND,seed=seed),'msac',CFG['profiles']['msac'])
            self.assertEqual(v['E_backend'],'msac')
            self.assertEqual(v['status'],'success',v['reason'])
            np.testing.assert_allclose(v['R'],r,atol=2e-4,rtol=0)
            self.assertGreater(float(np.dot(v['t'],t/np.linalg.norm(t))),.9999)

    def test_all_profiles_share_H_and_early_failure_checks(self):
        global CALLS
        from test_stage7a_backend_control import scene
        p,q,k1,k2,_,_=scene();values=[]
        for profile in ['old','magsac','msac']:
            CALLS+=1;v=reconstruct_profile(p,q,k1,k2,BACKEND,profile,CFG['profiles'][profile]);values.append(v)
            CALLS+=1;failure=reconstruct_profile(p[:4],q[:4],k1,k2,BACKEND,profile,CFG['profiles'][profile])
            self.assertEqual(failure['reason'],'insufficient_matches')
        for v in values[1:]:
            np.testing.assert_array_equal(v['arrays']['H'],values[0]['arrays']['H'])
            np.testing.assert_array_equal(v['arrays']['homography_mask'],values[0]['arrays']['homography_mask'])


class ScorecardContracts(unittest.TestCase):
    def fixture(self):
        cfg={**CFG,'seeds':[42,43]}
        cases=[{'case_id':f'{s}_{i}','scan':s,'conditions':['normal5','reduced3','spaced3','blur5']} for s in [1,6] for i in [0,1]]
        rows=[]
        for c in cases:
            for arm in cfg['arms']:
                for seed in cfg['seeds']:
                    error=2. if c['case_id'].endswith('_0') else 8.
                    if arm=='C_magsac':error=1. if c['case_id'].endswith('_0') else 30.
                    if arm=='C_msac':error=1.
                    rows.append({'case_id':c['case_id'],'scan':c['scan'],'conditions':c['conditions'],'arm':arm,'seed':seed,
                        'geometry_seconds':.01,'status':'success',
                        'pose':{'correct':error<=5,'accepted':True,'false_accept':error>5,'joint_deg':error,
                                'auc':{str(t):max(0,1-error/t) for t in [5,10,20]}}})
        return rows,cases,cfg

    def test_notification_preserves_tradeoffs_no_automatic_winner(self):
        rows,cases,cfg=self.fixture();report=scorecards(rows,cases,cfg)
        self.assertTrue(report['notices']['C_msac']['severe_events_halved_within_auc_tolerance'])
        self.assertFalse(report['unseen_objects_tested'])
        for row in rows:
            if row['arm']=='C_msac' and row['seed']==43:
                row['pose'].update(correct=False,false_accept=True,joint_deg=40.,auc={str(t):0 for t in [5,10,20]})
        report=scorecards(rows,cases,cfg)
        msac=[r for r in report['cards'] if r['arm']=='C_msac']
        self.assertTrue(all(r['macro']['auc10']['min']==0 for r in msac))
        self.assertTrue(all(r['severe_events']>0 for r in msac))
        self.assertTrue(any(r['new_severe_pairs'] for r in report['contrasts']))
        self.assertIn('user',report['candidate_decision'])

    def test_duplicate_missing_rows_rejected_and_rejections_keep_denominator(self):
        rows,cases,cfg=self.fixture()
        with self.assertRaises(ValueError):scorecards(rows[:-1],cases,cfg)
        with self.assertRaises(ValueError):scorecards(rows+[rows[0]],cases,cfg)
        for r in rows:
            if r['arm']=='C_msac':r['pose'].update(accepted=False,correct=False,false_accept=False,joint_deg=None,auc={str(t):0 for t in [5,10,20]})
        report=scorecards(rows,cases,cfg)
        self.assertTrue(all(r['macro']['auc10']['mean']==0 for r in report['cards'] if r['arm']=='C_msac'))
        self.assertFalse(report['notices']['C_msac']['severe_events_halved_within_auc_tolerance'])

    def test_dynamic_read_guard_blocks_any_source_run_directory(self):
        spec=importlib.util.spec_from_file_location('stability_runner',ROOT/'scripts/run_stage7a_backend_stability.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        runner.DENIED_ROOTS=(Path('/custom/cached_run'),)
        with self.assertRaises(PermissionError):runner.guard('open',('/custom/cached_run/fits/pose.npz','r',0))
        with self.assertRaises(PermissionError):runner.guard('open',('/x/reference.json','r',0))
        runner.guard('open',('/new/inputs/observations.npz','r',0))


if __name__=='__main__':unittest.main()
