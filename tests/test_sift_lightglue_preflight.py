"""Input contract guards; optional matcher dependency stays outside core installs."""
import importlib.util
import unittest
from pathlib import Path
try:
    import torch
    import lightglue
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'isolated LightGlue environment required')
class InputContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/preflight_sift_lightglue.py'
        spec=importlib.util.spec_from_file_location('lg_preflight',path)
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)
    def fixture(self):
        return {'keypoints':torch.tensor([[[10.,20.]]]),'descriptors':torch.ones(1,1,128)/(128**.5),'scales':torch.ones(1,1),'oris':torch.ones(1,1),'image_size':torch.tensor([[800.,600.]])}
    def test_contract_accepts_pixel_center_with_rootsift(self):
        self.assertEqual(self.module.validate(self.fixture()),1)
    def test_rejects_swapped_size_outside_height(self):
        f=self.fixture();f['keypoints'][0,0]=torch.tensor([700.,20.]);f['image_size']=torch.tensor([[600.,800.]])
        with self.assertRaises(AssertionError):self.module.validate(f)
    def test_rejects_degrees(self):
        f=self.fixture();f['oris'][0,0]=90
        with self.assertRaises(AssertionError):self.module.validate(f)
    def test_rejects_raw_sift_magnitude(self):
        f=self.fixture();f['descriptors']*=512
        with self.assertRaises(AssertionError):self.module.validate(f)
    def test_rejects_nonfinite_and_wrong_dimension(self):
        for key,value in [('keypoints',torch.tensor([[[float('nan'),0.]]])),('descriptors',torch.ones(1,1,256))]:
            f=self.fixture();f[key]=value
            with self.assertRaises(AssertionError):self.module.validate(f)
if __name__=='__main__':unittest.main()
