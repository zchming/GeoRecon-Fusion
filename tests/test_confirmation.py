import unittest
from georecon.evaluation.confirmation import confirmation_gate

class ConfirmationTests(unittest.TestCase):
    def gate(self,after=None,**kw):
        b=dict(auc10=.5,accurate_scene=.5,false_pairs=20,severe_pairs=2)
        a={**b,'auc10':.52,'false_pairs':18};a.update(after or {})
        args=dict(scene_score_deltas={'a':.02,'b':.02,'c':.02},scene_severe_deltas={'a':0,'b':0,'c':0},seed_score_deltas={i:.02 for i in range(5)},complete=True,replication=True);args.update(kw)
        return confirmation_gate(b,a,**args)
    def test_bounded_gain_passes(self):self.assertTrue(self.gate()['passed'])
    def test_no_posthoc_noninferiority_margin(self):self.assertFalse(self.gate({'auc10':.499999})['passed'])
    def test_average_does_not_hide_severe_or_scene_regression(self):
        self.assertFalse(self.gate({'severe_pairs':3})['passed'])
        self.assertFalse(self.gate(scene_severe_deltas={'a':1,'b':-1,'c':0})['passed'])
        self.assertFalse(self.gate(scene_score_deltas={'a':-.001,'b':.1,'c':.1})['passed'])
    def test_missing_evidence_and_seed_instability_fail(self):
        self.assertFalse(self.gate(complete=False)['passed'])
        self.assertFalse(self.gate(seed_score_deltas={0:.1,1:.1,2:.1,3:0,4:0})['passed'])
    def test_tiny_gain_not_material_and_no_fabricated_false_reduction(self):
        self.assertFalse(self.gate({'auc10':.5001,'false_pairs':20})['passed'])
        self.assertTrue(self.gate({'auc10':.5001,'false_pairs':18})['passed'])
