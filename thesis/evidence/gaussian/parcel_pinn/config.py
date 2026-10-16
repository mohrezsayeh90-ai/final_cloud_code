from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class Config:
    preset: str = "demo"
    seed: int = 7
    seeds: list[int] = field(default_factory=lambda: [7])
    dimension: int = 2
    lengths: tuple = (3., 1.)
    depth_ratio: float = 1.  # B/H; 2D planar sphere-parcel force and mass per depth
    final_time: float = 1.
    reynolds: float = 20.
    density_ratio: float = 1000.
    nparcels: int = 6
    width: float = .14
    gravity: tuple = (0., -.1)  # effective buoyancy-corrected acceleration, U^2/H
    collision: str = "none"
    stiffness: float = 300.
    damping: float = .03
    hidden: int = 32
    depth: int = 3
    epochs: int = 400
    ode_epochs: int = 700
    tg_epochs: int = 1200
    lbfgs_steps: int = 80
    ode_lbfgs_steps: int = 120
    tg_lbfgs_steps: int = 200
    fourier_modes: int = 3
    relaxation_basis: bool = True
    lr: float = .002
    time_batch: int = 3
    space_batch: int = 40
    quadrature: tuple = (16, 16)
    frames: int = 41
    dpi: int = 180
    threads: int = 2
    residual_tolerance: float = .03
    relative_l2_tolerance: float = .03
    learning_rate_decay: float = .25
    cases: list[dict] = field(default_factory=lambda: [
        {"name": "low_inertia", "stokes": .08, "loading": .05},
        {"name": "baseline", "stokes": .2, "loading": .05},
        {"name": "high_loading", "stokes": .2, "loading": .15},
        {"name": "high_inertia", "stokes": .6, "loading": .05},
    ])

    def validate(self):
        if self.preset not in ("smoke", "demo", "paper"):
            raise ValueError("preset must be smoke, demo, or paper")
        if self.dimension not in (2, 3) or len(self.lengths) != self.dimension or any(x <= 0 for x in self.lengths):
            raise ValueError("lengths must match dimension (2 or 3)")
        if len(self.gravity) != self.dimension or len(self.quadrature) != self.dimension or self.depth_ratio <= 0:
            raise ValueError('gravity/quadrature must match dimension and B/H must be positive')
        if self.lengths[1] != 1:
            raise ValueError("H is the length scale: lengths[1] must equal 1")
        if min(self.reynolds, self.density_ratio, self.width, self.final_time, self.lr) <= 0:
            raise ValueError("physical and optimization scales must be positive")
        if self.width >= min(self.lengths[d] for d in range(self.dimension) if d != 1) / 3:
            raise ValueError("use width < one third of each periodic length")
        if self.collision not in ("none", "soft_sphere"):
            raise ValueError("supported collision closures: none, soft_sphere")
        if min(self.nparcels, self.hidden, self.depth, self.time_batch, self.space_batch, self.frames, self.threads) < 1:
            raise ValueError("network and sampling counts must be positive")
        if min(self.epochs, self.ode_epochs, self.tg_epochs, self.lbfgs_steps, self.ode_lbfgs_steps, self.tg_lbfgs_steps) < 0 or min(self.quadrature) < 2 or self.fourier_modes < 1:
            raise ValueError("epochs must be nonnegative and quadrature sizes >=2")
        if not self.seeds or not self.cases:
            raise ValueError("at least one seed and case are required")
        names = [c['name'] for c in self.cases]
        if len(set(names)) != len(names) or any(not n.replace('_', '').isalnum() for n in names):
            raise ValueError("case names must be unique safe identifiers")
        if any(c['stokes'] <= 0 or c['loading'] < 0 for c in self.cases):
            raise ValueError("Stokes must be positive and loading nonnegative")
        return self

    def physical(self, case):
        d = math.sqrt(18 * case['stokes'] / (self.density_ratio * self.reynolds))
        mp = self.density_ratio * math.pi * d**3 / 6 / (self.depth_ratio if self.dimension == 2 else 1.)
        volume = math.prod(self.lengths)
        N = case['loading'] * volume / (self.nparcels * mp)
        if self.collision == 'soft_sphere':
            N = 1.
        return dict(diameter=d, physical_mass=mp, multiplicity=N,
                    represented_mass=mp*N, stokes=case['stokes'],
                    loading=self.nparcels*mp*N/volume)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def configuration(preset="demo", path=None):
    c = Config(preset=preset)
    if preset == "smoke":
        c.epochs, c.ode_epochs, c.tg_epochs = 3, 5, 3
        c.lbfgs_steps, c.ode_lbfgs_steps, c.tg_lbfgs_steps = 0, 0, 0
        c.hidden, c.depth, c.quadrature = 16, 2, (8, 8)
        c.time_batch, c.space_batch, c.frames, c.dpi = 2, 12, 12, 100
        c.cases = c.cases[:2]
    if preset == "paper":
        c.epochs, c.ode_epochs, c.tg_epochs = 12000, 6000, 8000
        c.hidden, c.depth, c.quadrature = 64, 4, (64, 48)
        c.time_batch, c.space_batch, c.frames, c.dpi = 6, 192, 121, 300
        c.seeds = [7, 17, 29]
        c.lbfgs_steps, c.ode_lbfgs_steps, c.tg_lbfgs_steps = 800, 400, 400
    if path:
        values = json.loads(Path(path).read_text())
        for k, v in values.items():
            if not hasattr(c, k):
                raise ValueError(f"unknown configuration key: {k}")
            setattr(c, k, v)
    return c.validate()
