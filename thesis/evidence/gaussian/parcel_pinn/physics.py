import itertools
import math
import numpy as np
import torch


def grid(lengths, shape, *, dtype=torch.float64):
    axes = [(torch.arange(n, dtype=dtype)+.5)*L/n for n, L in zip(shape, lengths)]
    xyz = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, len(shape))
    return xyz, math.prod(lengths)/math.prod(shape)


def gaussian_raw(x, q, width, lengths, periodic=(0, 2)):
    """Product of periodized 1D Gaussians; omitted prefactor cancels in normalization.

    x: [T,S,D], q: [T,N,D]. Sum images through >=8 widths beyond nearest image.
    Bounded directions are renormalized on the actual quadrature, never reflected.
    """
    out = torch.ones((*x.shape[:-1], q.shape[-2]), dtype=x.dtype, device=x.device)
    for d, L in enumerate(lengths):
        delta = x[..., d, None] - q[:, None, :, d]
        if d in periodic:
            delta = delta - L*torch.floor(delta/L+.5)
            images = max(1, math.ceil(8*float(width)/L))
            val = sum(torch.exp(-.5*((delta+k*L)/width)**2)
                      for k in range(-images, images+1))
        else:
            val = torch.exp(-.5*(delta/width)**2)
        out = out*val
    return out


def weights(x, q, width, lengths, dv, periodic=(0, 2)):
    g = gaussian_raw(x, q, width, lengths, periodic)
    z = g.sum(1, keepdim=True)*dv
    if torch.any(z <= torch.finfo(g.dtype).tiny):
        raise FloatingPointError("Gaussian support left the quadrature domain")
    return g/z, z


def interpolate(W, u, dv):
    return torch.einsum('tsn,tsd->tnd', W, u)*dv


def spread(W, force):
    return -torch.einsum('tsn,tnd->tsd', W, force)


def derivative(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]


def flow_terms(uvp, xt, nu):
    """Pointwise unsteady incompressible NS residual without external source."""
    dims = uvp.shape[1]-1
    gradients = [derivative(uvp[:, d], xt) for d in range(dims)]
    pressure_gradient = derivative(uvp[:, -1], xt)[:, :dims]
    lap = torch.stack([sum(derivative(gradients[d][:, j], xt)[:, j]
                           for j in range(dims)) for d in range(dims)], 1)
    advection = torch.stack([(uvp[:, :dims]*g[:, :dims]).sum(1) for g in gradients], 1)
    dt = torch.stack([g[:, -1] for g in gradients], 1)
    div = sum(gradients[d][:, d] for d in range(dims))
    return dt+advection+pressure_gradient-nu*lap, div, gradients


def binary_impact(v1, v2, m1, m2, normal, restitution):
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    a, b = np.asarray(v1, float).copy(), np.asarray(v2, float).copy()
    gn = np.dot(a-b, n)
    if gn < 0:
        J = -(1+restitution)*gn/(1/m1+1/m2)
        a += J*n/m1
        b -= J*n/m2
    return a, b


def wall_reflection(v, normal, restitution):
    v, n = np.asarray(v, float).copy(), np.asarray(normal, float)
    n = n/np.linalg.norm(n)
    return v-(1+restitution)*min(np.dot(v, n), 0)*n


def sample_impact_normal(relative_velocity, rng):
    e = np.asarray(relative_velocity, float)
    speed = np.linalg.norm(e)
    if speed == 0:
        raise ValueError('zero relative speed has no collision normal')
    e = e/speed
    base = np.eye(3)[np.argmin(np.abs(e))]
    a = np.cross(e, base); a /= np.linalg.norm(a)
    b = np.cross(e, a)
    r, phi = np.sqrt(rng.random()), 2*np.pi*rng.random()
    return -np.sqrt(1-r*r)*e+r*(np.cos(phi)*a+np.sin(phi)*b)


def group_event_count(rate, dt, rng):
    """Frozen-rate independent Poisson counter, NOT a state-changing collision step."""
    return rng.poisson(rate*dt)


def soft_contact(q, v, diameter, mass, lengths, stiffness, damping):
    """Physical spheres (multiplicity=1), frictionless clipped spring/dashpot.

    Contact and gravity are not deposited as hydrodynamic reaction.
    No lubrication or rotation. Contact times must be resolved by the PINN.
    """
    force = torch.zeros_like(q)
    for i, j in itertools.combinations(range(q.shape[1]), 2):
        delta = q[:, i]-q[:, j]
        delta = torch.stack([delta[:, d]-lengths[d]*torch.floor(delta[:, d]/lengths[d]+.5)
                             if d in (0, 2) else delta[:, d] for d in range(q.shape[-1])], 1)
        r = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
        n = delta/r[:, None]
        gap = diameter-r
        gn = ((v[:, i]-v[:, j])*n).sum(1)
        f = torch.where(gap > 0, torch.relu(stiffness*gap-damping*gn), torch.zeros_like(gap))
        ids = torch.zeros(q.shape[1], dtype=q.dtype, device=q.device)
        ids[i], ids[j] = 1, -1
        force = force+ids[None, :, None]*f[:, None, None]*n[:, None, :]
    for sign, wall in [(1., 0.), (-1., lengths[1])]:
        gap = diameter/2-sign*(q[..., 1]-wall)
        f = torch.where(gap > 0, torch.relu(stiffness*gap-damping*sign*v[..., 1]), torch.zeros_like(gap))
        unit = q.new_tensor(([0., sign, 0.])[:q.shape[-1]])
        force = force+f[..., None]*unit
    return force/mass
