import copy
from itertools import combinations
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from georecon.teaching.initialization import coverage,rank_seeds,initialize,directed_pairs,seed_signals,verify_candidates
from georecon.teaching.incremental import reconstruct_incremental
from georecon.data.protocol import read_json

ROOT=Path(__file__).resolve().parents[1]


class InitializationTests(unittest.TestCase):
    def runner(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('stage7b_runner',ROOT/'scripts/run_stage7b_initialization.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

    def test_protocol_snapshot_whitespace_is_not_configuration_drift(self):
        import tempfile,json
        runner=self.runner()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=root/'config.json';snapshot=root/'snapshot.json';freeze=root/'freeze.json'
            config.write_text('{"x":1}');snapshot.write_text(json.dumps({'x':1},indent=2));freeze.write_text(json.dumps({'config_sha256':runner.sha(config)}))
            runner.verify_protocol(config,snapshot,freeze)
            snapshot.write_text('{"x":2}')
            with self.assertRaises(ValueError):runner.verify_protocol(config,snapshot,freeze)

    def test_blur_uses_original_grid_before_resizing(self):
        from PIL import Image,ImageFilter
        runner=self.runner();rgb=Image.fromarray(np.random.default_rng(4).integers(0,256,(1200,1600,3),dtype=np.uint8))
        expected=rgb.filter(ImageFilter.GaussianBlur(4)).resize((800,600),Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(runner.processed_image(rgb,'blur',4),expected)
        wrong=rgb.resize((800,600),Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        self.assertFalse(np.array_equal(np.asarray(expected),np.asarray(wrong)))

    def test_coverage_normalized_by_image_size_unknown_explicit(self):
        a=np.array([[10.,10.],[310,230],[610,450]])
        self.assertEqual(coverage(a,(640,480)),coverage(a*2,(1280,960)))
        self.assertIsNone(coverage(np.empty((0,2)),(640,480)))
        self.assertIsNone(coverage(np.array([[np.nan,0]]),(640,480)))

    def signals(self):
        return [dict(pair=[0,1],status='success',reason=None,essential_inliers=100,coverage=.1,reachable_other_views=0),
                dict(pair=[0,2],status='success',reason=None,essential_inliers=60,coverage=.4,reachable_other_views=1),
                dict(pair=[1,2],status='no_reconstruction',reason='homography',essential_inliers=500,coverage=1.,reachable_other_views=0)]

    def test_quality_does_not_promote_rejected_geometry_or_use_ground_truth(self):
        signals=self.signals();before=copy.deepcopy(signals)
        self.assertEqual(rank_seeds(signals,'inliers')[0]['pair'],[0,1])
        self.assertEqual(rank_seeds(signals,'quality')[0]['pair'],[0,2])
        self.assertEqual(rank_seeds(signals,'quality')[-1]['reason'],'homography')
        self.assertEqual(signals,before)
        with self.assertRaises(ValueError):rank_seeds(signals,'oracle')

    def test_unknown_is_not_fabricated_zero_and_ties_are_deterministic(self):
        s=self.signals()[:2]
        for v in s:v.update(reachable_other_views=0,essential_inliers=60,coverage=None)
        self.assertEqual(rank_seeds(s[::-1],'quality')[0]['pair'],[0,1])
        s[1]['coverage']=0.
        self.assertEqual(rank_seeds(s,'quality')[0]['pair'],[0,2])

    def test_retry_limit_and_no_fallback_after_accepted_seed(self):
        signals=self.signals()
        for s in signals:s.update(status='no_reconstruction',reason='rejected')
        result,selection=initialize([None]*3,[],[],{}, {},{},signals,'inliers',[0,1],2)
        self.assertEqual(len(selection['attempts']),2);self.assertFalse(result[0])
        with self.assertRaises(ValueError):initialize([],[],[],{},{},{},[],'inliers',[0,1],0)
        signals=self.signals()
        with patch('georecon.teaching.initialization.reconstruct_incremental',return_value='map') as run:
            result,selection=initialize([None]*3,[],[],{}, {},{(0,1):{}},signals,'inliers',[0,1],3)
            self.assertEqual(result,'map');self.assertEqual(len(selection['attempts']),1);run.assert_called_once()

    def test_reverse_matches_preserve_keypoint_identity(self):
        m={(0,1):np.array([[0,2],[1,0]])}
        np.testing.assert_array_equal(directed_pairs(m,1,0),[[2,0],[0,1]])
        np.testing.assert_array_equal(m[0,1],[[0,2],[1,0]])

    def test_conflicting_third_view_landmark_links_excluded(self):
        features=[(np.array([[10.,10.],[100,100.]]),None)]*3
        matches={(0,1):np.array([[0,0],[1,1]]),(0,2):np.array([[0,0],[1,1]]),(1,2):np.array([[0,1],[1,0]])}
        fit={'status':'success','reason':None,'stats':{'essential_inliers':2,'homography_inliers':0,'kept_points':2},'arrays':{'kept_match_indices':np.array([0,1]),'essential_mask':np.ones(2,bool)}}
        s=seed_signals((0,1),fit,features,matches,[(640,480)]*3,{'pnp_min_inliers':1})
        self.assertEqual(s['other_view_links'][0]['unambiguous_landmarks'],0);self.assertEqual(s['reachable_other_views'],0)

    def test_cached_interface_rejects_wrong_seed_and_invalid_pairs(self):
        cfg=read_json(ROOT/'configs/teaching/stage6_incremental.json');f=[(np.array([[0.,0.],[1,1.]]),None)]*3
        m={p:np.array([[0,0],[1,1]]) for p in combinations(range(3),2)}
        with self.assertRaises(ValueError):reconstruct_incremental([None]*3,[np.eye(3)]*3,[0,1],cfg,features=f,pair_matches={},seed_result={})
        base={'arrays':{'pixels_first':np.array([[3.,3.],[1,1.]]),'pixels_second':f[1][0]}}
        with self.assertRaises(ValueError):reconstruct_incremental([None]*3,[np.eye(3)]*3,[0,1],cfg,features=f,pair_matches=m,seed_result=base)

    def test_real_msac_degenerate_inputs_reject(self):
        cfg=read_json(ROOT/'configs/teaching/stage6_incremental.json');msac=read_json(ROOT/'configs/experiments/stage7a_backend_stability_v1.json')['profiles']['msac']
        xy=np.c_[np.arange(30.),np.arange(30.)];features=[(xy,None)]*3;m={p:np.c_[np.arange(30),np.arange(30)] for p in combinations(range(3),2)}
        fits,signals,_=verify_candidates(features,m,[np.eye(3)]*3,[(640,480)]*3,cfg,msac)
        self.assertTrue(all(s['reason']=='identical_correspondences_no_baseline' for s in signals))
        self.assertTrue(all(s['reachable_other_views']==0 for s in signals))

if __name__=='__main__':unittest.main()
