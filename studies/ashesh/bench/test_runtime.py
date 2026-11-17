"""CPU torch checks; run in the staged pod environment with CUDA hidden."""
import unittest
from types import SimpleNamespace
import torch
from app.robo_jepa.world_model import WorldModelState
from bench.rollout import Rollout
from bench.harness import bounded_plan
from bench.interface import PlanContext

class ToyModel:
    model = SimpleNamespace(embodiment_list=['franka_omron_state_delta'], action_dim=64)
    def forward(self, features, actions, states, **kwargs):
        # z_{t+1} = z_t + u_t + p_t; p_{t+1} = p_t + u_t.
        value = features + (actions[...,7:10]+states[...,7:10])[:,:,None,None,None,:]
        return value, None, None, None

class RuntimeTests(unittest.TestCase):
    def test_batched_multistep_preserves_state_derivatives(self):
        state = WorldModelState.from_features(torch.zeros(1,1,2,1,1,3),
            actions=torch.zeros(1,0,64), states=torch.zeros(1,1,64),
            view_map=torch.tensor([[0,2]]), embodiment_ids=torch.tensor([[0]]))
        rollout = Rollout(ToyModel(),state,horizon_max=3)
        u = torch.tensor([[[.01,.02,.03],[.04,.05,.06],[.07,.08,.09]],
                          [[-.01,.01,0],[0,.02,0],[.03,0,0]]], requires_grad=True)
        z = rollout(u)
        self.assertEqual(tuple(z.shape),(2,3,6))
        torch.testing.assert_close(z[0,-1,:3],torch.tensor([.18,.24,.30]))
        grad = torch.autograd.grad(z[:,-1].sum(),u,create_graph=True)[0]
        torch.testing.assert_close(grad[0,:,0],torch.tensor([6.,4.,2.]))
        self.assertEqual(rollout.predictor_calls,3)
        self.assertEqual(rollout.predicted_samples,6)

    def test_plan_is_l2_clipped_and_nonfinite_rejected(self):
        context=PlanContext(None,None,torch.zeros(3),3,.3,1.,0)
        actual=bounded_plan(torch.tensor([[.3,.4,0.],[0,0,0.]]),context)
        torch.testing.assert_close(actual,torch.tensor([[.18,.24,0.],[0,0,0.]]))
        with self.assertRaises(ValueError): bounded_plan(torch.tensor([[float('nan'),0,0]]),context)
        with self.assertRaises(ValueError): bounded_plan(torch.zeros(4,3),context)

if __name__=='__main__': unittest.main()
