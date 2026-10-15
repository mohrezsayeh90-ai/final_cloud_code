import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from parcel_pinn.config import configuration
from parcel_pinn.model import CoupledPINN
from parcel_pinn.physics import (binary_impact, flow_terms, grid, interpolate, sample_impact_normal,
                                 soft_contact, spread, weights)
from parcel_pinn.training import fit, seed_all


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        seed_all(7)

    def test_transfer_conserves_force_work_and_constant_in_2d_and_3d(self):
        for D in [2, 3]:
            xyz,dv=grid((1.,)*D,(12,)*D)
            q=torch.tensor([[[.99,.01,.5][:D],[.1,.9,.3][:D]]],dtype=torch.float64)
            W,_=weights(xyz[None],q,.12,(1.,)*D,dv)
            u=torch.randn(1,len(xyz),D,dtype=torch.float64)
            U=interpolate(W,u,dv)
            F=torch.randn(1,2,D,dtype=torch.float64)
            f=spread(W,F)
            self.assertTrue(torch.allclose(f.sum(1)*dv,-F.sum(1),atol=1e-12))
            self.assertAlmostEqual(float((u*f).sum()*dv),float(-(U*F).sum()),places=12)
            const=interpolate(W,torch.ones_like(u)*3.7,dv)
            self.assertTrue(torch.allclose(const,torch.ones_like(const)*3.7,atol=1e-12))

    def test_periodic_seam_and_wall_normalization(self):
        xyz,dv=grid((1.,1.),(24,24))
        q=torch.tensor([[[.001,.01],[1.001,.01]]],dtype=torch.float64)
        W,_=weights(xyz[None],q,.1,(1.,1.),dv)
        self.assertTrue(torch.allclose(W[:,:,0],W[:,:,1],atol=1e-12))
        self.assertTrue(torch.allclose(W.sum(1)*dv,torch.ones(1,2,dtype=torch.float64)))

    def test_exact_taylor_green_satisfies_shared_NS_operator(self):
        xt=torch.rand(71,3,dtype=torch.float64,requires_grad=True)
        x,y,t=xt.unbind(1)
        out=torch.stack([x.sin()*y.cos()*torch.exp(-.02*t),
                         -x.cos()*y.sin()*torch.exp(-.02*t),
                         .25*((2*x).cos()+(2*y).cos()-2)*torch.exp(-.04*t)],1)
        r,d,_=flow_terms(out,xt,.01)
        self.assertLess(float(r.abs().max().detach()),1e-12)
        self.assertLess(float(d.abs().max().detach()),1e-12)

    def test_channel_hard_conditions_and_coupled_gradients(self):
        c=configuration('smoke');m=CoupledPINN(c,c.cases[0]);D=c.dimension
        a=torch.tensor([[.2,0.,.4],[.2,1.,.4]],dtype=torch.float64)
        self.assertLess(float(m.field(a)[:,:D].abs().max().detach()),1e-12)
        t=torch.zeros(1,1,dtype=torch.float64)
        q,v=m.particle_state(t)
        self.assertTrue(torch.allclose(q[0],m.q0));self.assertTrue(torch.allclose(v[0],m.v0))
        left=torch.tensor([[0.,.3,.4]],dtype=torch.float64)
        right=left.clone();right[:,0]=c.lengths[0]
        self.assertTrue(torch.allclose(m.field(left),m.field(right),atol=1e-12))
        loss,_=m.loss(torch.tensor([[.2],[.7]],dtype=torch.float64),torch.rand(2,13,D,dtype=torch.float64)*torch.tensor(c.lengths))
        loss.backward()
        for net in [m.fluid,m.parcels]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().max()>0 for p in net.parameters()))

    def test_per_depth_convention_and_fixed_loading(self):
        c=configuration('smoke');p=c.physical(c.cases[0])
        c.depth_ratio=2;p2=c.physical(c.cases[0])
        self.assertAlmostEqual(p2['physical_mass'],p['physical_mass']/2)
        self.assertAlmostEqual(p2['multiplicity'],p['multiplicity']*2)
        self.assertAlmostEqual(p2['loading'],p['loading'])

    def test_initial_pressure_is_free_but_gauge_is_fixed(self):
        c=configuration('smoke');m=CoupledPINN(c,c.cases[0])
        pts=torch.tensor([[.4,.3,0.],[1.1,.7,0.],[0.,0.,0.]],dtype=torch.float64)
        p=m.field(pts)[:,-1]
        self.assertGreater(float(p[:2].abs().max().detach()),1e-6)
        self.assertAlmostEqual(float(p[-1].detach()),0.,places=14)

    def test_binary_impact_no_reimpact_and_energy(self):
        a,b=binary_impact([1,.5],[-1,-.5],1,1,[-1,0],.8)
        aa,bb=binary_impact(a,b,1,1,[-1,0],.8)
        np.testing.assert_allclose(a,aa);np.testing.assert_allclose(b,bb)
        np.testing.assert_allclose(a+b,[0,0]);self.assertAlmostEqual((a@a+b@b)/2,.89)

    def test_normal_sampler_is_approaching_unit_length(self):
        rng=np.random.default_rng(55)
        normals=np.array([sample_impact_normal([1,2,3],rng) for _ in range(1000)])
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-14)
        self.assertTrue(np.all(normals@np.array([1,2,3])<0))

    def test_internal_contact_has_zero_total_force(self):
        q=torch.tensor([[[.4,.5],[.45,.5]]],dtype=torch.float64)
        v=torch.tensor([[[1.,0.],[-1.,0.]]],dtype=torch.float64)
        a=soft_contact(q,v,.1,1.,(3.,1.),100.,1.)
        self.assertTrue(torch.allclose(a.sum(1),torch.zeros(1,2,dtype=torch.float64)))
        self.assertLess(float(a[0,0,0]),0)

    def test_nonfinite_training_restores_best_and_saves_output(self):
        m=torch.nn.Linear(1,1).double();x=torch.ones(3,1,dtype=torch.float64)
        def objective(audit=False):
            loss=(m(x)-2).square().mean()
            return loss,{'residual':loss}
        with tempfile.TemporaryDirectory() as td:
            state,hist=fit(m,objective,3,.01,td,fail_at=2)
            self.assertIn('nonfinite',state['termination'])
            self.assertTrue(torch.isfinite(m(x)).all())
            self.assertTrue((Path(td)/'checkpoint.pt').is_file())
            self.assertTrue((Path(td)/'training_status.json').is_file())


if __name__=='__main__':unittest.main()
