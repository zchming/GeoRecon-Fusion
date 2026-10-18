"""Regression for offline evaluation against a prior run without cache_lock.json."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CacheLockRegression(unittest.TestCase):
    def test_reconstructs235_original_cache_locks_without_prior_lock_file(self):
        spec=importlib.util.spec_from_file_location('sampling_cache_runner',ROOT/'scripts/run_stage7a_msac_sampling.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache';cache.mkdir();cases=[{'case_id':str(i)} for i in range(46)]
            for n in ['bundle.json','input_hashes.json','implementation.json']:(cache/n).write_text('{}')
            for i in range(24):
                p=cache/'features'/str(i);p.mkdir(parents=True)
                for n in ['features.npz','record.json']:(p/n).write_bytes(b'test-only')
            for case in cases:
                for arm in ['B','C']:
                    p=cache/'main'/case['case_id']/arm;p.mkdir(parents=True)
                    for n in ['matches.npz','record.json']:(p/n).write_bytes(b'test-only')
            locks=runner.cache_lock(cache,cases)
            self.assertEqual(len(locks),235)
            self.assertTrue(all(len(h)==64 for h in locks.values()))
            (cache/'main'/'0'/'C'/'matches.npz').write_bytes(b'changed')
            self.assertNotEqual(locks,runner.cache_lock(cache,cases))
            (cache/'features'/'0'/'record.json').unlink()
            with self.assertRaises(ValueError):runner.cache_lock(cache,cases)


if __name__=='__main__':unittest.main()
