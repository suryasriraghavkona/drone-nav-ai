"""Quick checks that need no PyTorch:  python tests/smoke_test.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from simulation.world import World
from planning.expert import expert_steer
from baseline.waypoint_nav import WaypointBaseline


def test_render():
    img = World(1).render()
    assert img.shape == (64, 64) and img.dtype == np.uint8


def test_collision_with_wall():
    w = World(1)
    for _ in range(200):
        if w.step(1.0, 1.0) == "collision":
            return
    raise AssertionError("expected a collision")


def test_reaches_goal_without_obstacles():
    w = World(1)
    w.rad = np.array([]); w.ox = w.oy = w.vy = np.array([])
    for _ in range(500):
        if w.step(expert_steer(w), 1.0) != "running":
            break
    assert w.status == "success"


def test_baseline_runs():
    w, b = World(2), WaypointBaseline()
    for _ in range(500):
        if w.step(*b.act(w)) != "running":
            break
    assert w.status in ("success", "collision", "running")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
