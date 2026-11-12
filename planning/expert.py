"""Oracle potential-field planner.

Uses the true (privileged) positions of obstacles, including a short look-ahead
for moving ones. It is used ONLY to produce steering labels for the training
data and as an upper-bound reference in the evaluation, never as the 'baseline'.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def expert_steer(world, lookahead=1.0, k_att=1.0, k_rep=1.0, k_tan=1.0, R=4.0, R_wall=2.0, gain=0.4):
    c = world.cfg
    force = np.array([c.length - world.x, -world.y], dtype=float)
    force = k_att * force / (np.linalg.norm(force) + 1e-9)
    for ox, oy, vy, r in zip(world.ox, world.oy, world.vy, world.rad):
        if ox < world.x - 0.5:          # already behind the drone
            continue
        py = oy + vy * lookahead
        vec = np.array([world.x - ox, world.y - py])
        dist = np.linalg.norm(vec)
        d = max(dist - r - c.drone_radius, 0.05)
        if d < R:
            mag = (1.0 / d - 1.0 / R)
            force += k_rep * mag * vec / (dist + 1e-9)
            # tangential term: sidestep an obstacle that sits right on our line
            side = 1.0 if world.y - py >= 0 else -1.0
            force[1] += k_tan * mag * side
    dw = c.width / 2 - abs(world.y) - c.drone_radius
    if dw < R_wall:
        dw = max(dw, 0.05)
        force[1] += -np.sign(world.y) * k_rep * (1.0 / dw - 1.0 / R_wall)
    desired = np.arctan2(force[1], force[0])
    return float(np.clip(wrap(desired - world.h) / gain, -1.0, 1.0))


def expert_speed(world, slow_dist=3.0):
    """Oracle speed command: slow down when any obstacle is close ahead."""
    c = world.cfg
    for ox, oy, r in zip(world.ox, world.oy, world.rad):
        if 0 <= ox - world.x < slow_dist + r and abs(oy - world.y) < r + 1.5:
            return 0.5
    return 1.0
