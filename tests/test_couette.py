import unittest
import math
import numpy as np
import torch
from literature_validation.couette import (Settings,CylinderPINN,VelocityPINN,
                                           strain_traction,steady_operator,sample_points)
from parcel_pinn.training import seed_all


class CouetteTests(unittest.TestCase):
    def setUp(self):seed_all(11)

    def test_no_slip_for_both_formulations(self):
        for cls in [CylinderPINN,VelocityPINN]:
            m=cls(Settings(width=16))
            points,n=m.surface(127)
            u=m(points)[:,:2]
            target=torch.stack([-m.omega*(points[:,1]-.5),m.omega*(points[:,0]-1)],1)
            self.assertLess(float((u-target).abs().max().detach()),1e-12)
            wall=torch.tensor([[.2,0.],[1.4,1.]],dtype=torch.float64,requires_grad=True)
            target=torch.tensor([[-.5,0.],[.5,0.]],dtype=torch.float64)
            self.assertLess(float((m(wall)[:,:2]-target).abs().max().detach()),1e-12)

    def test_periodicity_and_centrosymmetry(self):
        for cls in [CylinderPINN,VelocityPINN]:
            m=cls(Settings(width=16))
            a=torch.tensor([[0.,.3],[.7,.25]],dtype=torch.float64,requires_grad=True)
            b=torch.tensor([[2.,.3],[1.3,.75]],dtype=torch.float64,requires_grad=True)
            fa,fb=m(a),m(b)
            self.assertTrue(torch.allclose(fa[0],fb[0],atol=1e-12))
            self.assertTrue(torch.allclose(fa[1,:2],-fb[1,:2],atol=1e-12))
            self.assertAlmostEqual(float(fa[1,2].detach()),float(fb[1,2].detach()),places=12)

    def test_rotating_cylinder_exact_NS_and_torque(self):
        radius,omega=.125,.37
        theta=(torch.arange(101,dtype=torch.float64)+.31)*2*math.pi/101
        n=torch.stack([theta.cos(),theta.sin()],1)
        xy=(radius*n).requires_grad_(True)
        x,y=xy.unbind(1);r2=x*x+y*y
        A=omega*radius**2
        field=torch.stack([-A*y/r2,A*x/r2,-.5*A*A/r2],1)
        residual,div=steady_operator(field,xy,.015625)
        self.assertLess(float(residual.abs().max().detach()),1e-11)
        self.assertLess(float(div.abs().max().detach()),1e-12)
        stress=strain_traction(field,xy,n)
        torque=.5*(-n[:,1]*stress[:,0]+n[:,0]*stress[:,1]).mean()
        self.assertAlmostEqual(float(torque.detach()),-omega,places=12)

    def test_sample_points_stay_in_fluid(self):
        xy=sample_points(1024,torch.Generator().manual_seed(129)).numpy()
        self.assertTrue(np.all((xy[:,0]-1)**2+(xy[:,1]-.5)**2>.125**2))
        self.assertTrue(np.all((xy[:,0]>=0)&(xy[:,0]<=2)&(xy[:,1]>=0)&(xy[:,1]<=1)))

    def test_rotation_parameter_receives_physics_gradients(self):
        m=VelocityPINN(Settings(width=16))
        loss,terms=m.loss(sample_points(32,torch.Generator().manual_seed(223)))
        loss.backward()
        self.assertTrue(torch.isfinite(m.omega.grad))
        self.assertGreater(abs(float(m.omega.grad)),1e-9)

    def test_multipole_features_keep_periodic_no_slip_constraints(self):
        m=VelocityPINN(Settings(width=16,feature_set='multipole'))
        points,n=m.surface(97)
        target=torch.stack([-m.omega*(points[:,1]-.5),m.omega*(points[:,0]-1)],1)
        self.assertLess(float((m(points)[:,:2]-target).abs().max().detach()),1e-12)
        a=torch.tensor([[0.,.3],[2.,.3]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.allclose(m(a)[0],m(a)[1],atol=1e-12))


if __name__=='__main__':unittest.main()
