import importlib.util
from pathlib import Path
import unittest
import torch
p=Path(__file__).resolve().parents[2]/'lingbotvla/optim/accumulation.py'
s=importlib.util.spec_from_file_location('acc',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class AccumulationTests(unittest.TestCase):
    def test_four_microbatches_match_full_batch_gradient(self):
        torch.manual_seed(11)
        a=torch.nn.Linear(5,3).double();b=torch.nn.Linear(5,3).double();b.load_state_dict(a.state_dict())
        x=torch.randn(4,5,dtype=torch.double);y=torch.randn(4,3,dtype=torch.double)
        torch.nn.functional.mse_loss(a(x),y).backward()
        for i in range(4):m.normalized_microbatch_loss(torch.nn.functional.mse_loss(b(x[i:i+1]),y[i:i+1]),4).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad)
        for model in [a,b]:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            m.finite_grad_norm(norm)
            torch.optim.SGD(model.parameters(),lr=.01).step()
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb)
    def test_bad_count_rejected(self):
        with self.assertRaises(RuntimeError):m.check_microbatch_count(1,4)
        with self.assertRaises(ValueError):m.normalized_microbatch_loss(torch.tensor(1.),0)
    def test_nonfinite_gradient_stops_update(self):
        for v in [float('nan'),float('inf')]:
            with self.assertRaises(FloatingPointError):m.finite_grad_norm(v)
        self.assertEqual(m.finite_grad_norm(torch.tensor(2.)),2.)
if __name__=='__main__':unittest.main()
