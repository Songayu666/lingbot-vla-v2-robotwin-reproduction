import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'baseline_improve_pipeline.py'
s=importlib.util.spec_from_file_location('pipeline',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class RetryTests(unittest.TestCase):
    def exercise(self, failures):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=base/'source';source.mkdir()
            calls=[]
            def fake(cmd,path,env):
                calls.append(cmd[-1]);i=len(calls)-1
                if i<len(failures):path.write_text(failures[i]);return 1
                ckpt=base/'accum4/checkpoints/global_step_6750/hf_ckpt';ckpt.mkdir(parents=True)
                (ckpt/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'fake.safetensors'}}))
                (ckpt/'fake.safetensors').write_bytes(b'test')
                return 0
            with patch.object(m,'BASE',base),patch.object(m,'SOURCE',source),patch.object(m,'run_logged',fake),patch.object(m,'announce',lambda _:None):
                error=None
                try:m.train('accum4')
                except RuntimeError as exc:error=exc
                return calls,error,(base/'accum4/.trained').exists(),(base/'accum4/.lowmem').exists()
    def test_cuda_oom_retries_lowmem_once(self):
        calls,error,trained,lowmem=self.exercise(['torch.OutOfMemoryError: CUDA out of memory'])
        self.assertIsNone(error);self.assertTrue(trained and lowmem)
        self.assertEqual(len(calls),2);self.assertTrue(calls[1].endswith('_lowmem.yaml'))
    def test_other_error_does_not_retry(self):
        calls,error,trained,_=self.exercise(['ValueError: invalid data'])
        self.assertIsNotNone(error);self.assertEqual(len(calls),1);self.assertFalse(trained)
    def test_repeated_oom_stops(self):
        calls,error,trained,_=self.exercise(['CUDA out of memory']*2)
        self.assertIsNotNone(error);self.assertEqual(len(calls),2);self.assertFalse(trained)
if __name__=='__main__':unittest.main()
