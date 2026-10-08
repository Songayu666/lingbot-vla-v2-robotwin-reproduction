import importlib.util
import unittest
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('phase_loss',root/'lingbotvla/models/vla/lingbot_vla/phase_loss.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PhaseLossTests(unittest.TestCase):
    def inputs(self):
        loss=torch.ones(2,12,55,requires_grad=True)
        actions=torch.zeros_like(loss)
        mask=torch.zeros_like(loss,dtype=torch.bool);mask[:,:,:12]=True;mask[:,:,28:30]=True
        pad=torch.zeros(2,12,dtype=torch.bool)
        return loss,actions,mask,pad
    def test_unweighted_matches_original_without_padding(self):
        l,a,mk,p=self.inputs();l=l*torch.rand_like(l)
        result,per,_=m.phase_flow_loss(l,a,mk,p,weight=1)
        torch.testing.assert_close(result,(l*mk).sum()/mk.sum())
        torch.testing.assert_close(per,(l*mk).sum((1,2))/mk.sum((1,2)))
    def test_padding_and_invalid_joints_have_zero_gradient(self):
        l,a,mk,p=self.inputs();p[:,8:]=True
        result,_,_=m.phase_flow_loss(l,a,mk,p);result.backward()
        self.assertEqual(l.grad[:,8:].abs().sum().item(),0)
        self.assertEqual(l.grad[:,:,12:28].abs().sum().item(),0)
        self.assertGreater(l.grad[:,:8,:12].sum().item(),0)
    def test_transition_doubles_relative_gradient(self):
        l,a,mk,p=self.inputs();a[:,5:,28]=1
        result,_,metrics=m.phase_flow_loss(l,a,mk,p,radius=0);result.backward()
        torch.testing.assert_close(l.grad[:,5,0],2*l.grad[:,0,0])
        self.assertGreater(metrics['phase_loss/event_fraction'].item(),0)
    def test_padding_edge_is_not_transition(self):
        l,a,mk,p=self.inputs();p[:,6:]=True;a[:,6:,28]=1
        _,_,metrics=m.phase_flow_loss(l,a,mk,p)
        self.assertEqual(metrics['phase_loss/event_fraction'].item(),0)
    def test_all_padded_is_finite_zero(self):
        l,a,mk,p=self.inputs();p[:]=True
        result,_,_=m.phase_flow_loss(l,a,mk,p);result.backward()
        self.assertEqual(result.item(),0);self.assertEqual(l.grad.abs().sum().item(),0)
    def test_missing_padding_fails_loudly(self):
        l,a,mk,p=self.inputs()
        with self.assertRaises(ValueError):m.phase_flow_loss(l,a,mk,None)

if __name__=='__main__':unittest.main()
