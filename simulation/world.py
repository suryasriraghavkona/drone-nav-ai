"""Lightweight drone simulator.

A top-down 2D world (a field with side walls) in which a drone flies from x=0 to
x=length. The drone carries a forward-facing monocular camera, simulated by
casting one ray per image column and drawing a first-person grayscale frame
(nearer surfaces are taller and brighter). Obstacles are circles; some are static
and some move sideways and bounce between the walls.

Conventions
-----------
* x: forward, y: left, heading h: angle from +x (counter-clockwise positive).
* steering command in [-1, 1]: +1 = full yaw to the left (increasing heading).
* speed command in [0, 1]: fraction of v_max.
* Camera columns run from the right edge of the view (index 0) to the left (last).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    length: float = 40.0        # distance to goal (m)
    width: float = 12.0         # corridor width (m)
    dt: float = 0.1             # simulation step (s)
    drone_radius: float = 0.3   # (m)
    v_max: float = 2.0          # (m/s)
    max_yaw_rate: float = 1.5   # (rad/s)
    max_heading: float = 1.2    # heading is clamped to +/- this (rad): no flying backwards
    fov: float = float(np.deg2rad(90))
    img_size: int = 64
    max_range: float = 15.0     # camera / range-sensor range (m)
    n_static: int = 7
    n_moving: int = 2
    max_steps: int = 500


class World:
    def __init__(self, seed=0, cfg=None):
        self.cfg = cfg or Config()
        self.seed = seed
        self._build(np.random.default_rng(seed))
        self.reset()

    # ------------------------------------------------------------------ layout
    def _build(self, rng):
        c = self.cfg
        n = c.n_static + c.n_moving
        xs, ys, rs = [], [], []
        tries = 0
        while len(xs) < n and tries < 5000:
            tries += 1
            x = rng.uniform(6.0, c.length - 4.0)
            y = rng.uniform(-c.width / 2 + 1.2, c.width / 2 - 1.2)
            r = rng.uniform(0.5, 0.9)
            # keep a gap wider than the drone between any two obstacles
            if all(np.hypot(x - px, y - py) > r + pr + 2.0 for px, py, pr in zip(xs, ys, rs)):
                xs.append(x); ys.append(y); rs.append(r)
        self.rad = np.array(rs)
        self.ox0 = np.array(xs)
        self.oy0 = np.array(ys)
        vy = np.zeros(len(xs))
        k = min(c.n_moving, len(xs))
        if k > 0:
            vy[-k:] = rng.choice([-1.0, 1.0], size=k) * rng.uniform(0.4, 0.9, size=k)
        self.vy0 = vy

    def reset(self):
        self.x, self.y, self.h = 0.0, 0.0, 0.0
        self.ox = self.ox0.copy()
        self.oy = self.oy0.copy()
        self.vy = self.vy0.copy()
        self.steps = 0
        self.path_len = 0.0
        self.last_speed = 0.0
        self.status = "running"
        return self.status

    # --------------------------------------------------------------- dynamics
    def step(self, steer, speed_frac):
        """Advance one step. Returns 'running', 'success' or 'collision'."""
        c = self.cfg
        steer = float(np.clip(steer, -1.0, 1.0))
        speed_frac = float(np.clip(speed_frac, 0.0, 1.0))
        self.h = float(np.clip(self.h + steer * c.max_yaw_rate * c.dt, -c.max_heading, c.max_heading))
        v = speed_frac * c.v_max
        self.x += v * np.cos(self.h) * c.dt
        self.y += v * np.sin(self.h) * c.dt
        self.path_len += v * c.dt
        self.last_speed = v
        # moving obstacles bounce between the side walls
        self.oy = self.oy + self.vy * c.dt
        lim = c.width / 2 - self.rad
        flip = np.abs(self.oy) > lim
        self.vy = np.where(flip, -self.vy, self.vy)
        self.oy = np.clip(self.oy, -lim, lim)
        self.steps += 1
        if self.collided():
            self.status = "collision"
        elif self.x >= c.length:
            self.status = "success"
        return self.status

    def collided(self):
        c = self.cfg
        if abs(self.y) > c.width / 2 - c.drone_radius:
            return True
        if len(self.rad) == 0:
            return False
        d = np.hypot(self.ox - self.x, self.oy - self.y)
        return bool(np.any(d < self.rad + c.drone_radius))

    # ---------------------------------------------------------------- sensing
    def cast_rays(self):
        """Distance (m) and hit type (0 wall, 1 obstacle) for each camera column."""
        c = self.cfg
        n = c.img_size
        ang = self.h + np.linspace(-c.fov / 2, c.fov / 2, n)
        dx, dy = np.cos(ang), np.sin(ang)
        half = c.width / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t_up = np.where(dy > 1e-9, (half - self.y) / dy, np.inf)
            t_dn = np.where(dy < -1e-9, (-half - self.y) / dy, np.inf)
        best = np.minimum(np.minimum(t_up, t_dn), c.max_range)
        best = np.maximum(best, 0.05)
        kind = np.zeros(n)
        for cx, cy, r in zip(self.ox, self.oy, self.rad):
            ocx, ocy = cx - self.x, cy - self.y
            tca = dx * ocx + dy * ocy
            d2 = ocx ** 2 + ocy ** 2 - tca ** 2
            hit = (tca > 0) & (d2 < r * r)
            t = tca - np.sqrt(np.maximum(r * r - d2, 0.0))
            t = np.where(hit & (t > 0), t, np.inf)
            closer = t < best
            best = np.where(closer, t, best)
            kind = np.where(closer, 1.0, kind)
        return best, kind

    def render(self, noise=0.0, rng=None):
        """First-person grayscale frame, uint8, shape (img_size, img_size)."""
        c = self.cfg
        n = c.img_size
        d, kind = self.cast_rays()
        rows = np.arange(n, dtype=np.float32)[:, None]
        horizon = n / 2.0
        bg = np.where(rows < horizon, 30 + 20 * rows / horizon, 60 + 40 * (rows - horizon) / horizon)
        bg = np.broadcast_to(bg, (n, n))
        half_h = (n * 1.5 / (d + 0.5)) / 2.0
        mask = np.abs(rows - horizon) < half_h[None, :]
        shade = (60 + 195 * (1 - d / c.max_range)) * np.where(kind > 0.5, 1.0, 0.6)
        img = np.where(mask, shade[None, :], bg).astype(np.float32)
        if noise > 0:
            rng = rng or np.random.default_rng()
            img = img + rng.normal(0, noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)
