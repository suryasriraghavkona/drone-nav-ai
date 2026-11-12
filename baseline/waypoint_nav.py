"""Non-AI baseline: waypoint following with simple reactive range-sensor avoidance.

The drone steers towards the goal waypoint. If the range sensor sees something
closer than `front_thresh` straight ahead, it slows down and turns towards the
side with more free space, holding that turn until the way ahead is clear.
It uses only the range readings (no learning, no oracle information).
"""
import numpy as np
from planning.expert import wrap


class WaypointBaseline:
    def __init__(self, front_thresh=4.0, half_cone=10, gain=0.6):
        self.front_thresh = front_thresh
        self.half_cone = half_cone
        self.gain = gain
        self.turn_dir = 0.0

    def reset(self):
        self.turn_dir = 0.0

    def act(self, world):
        c = world.cfg
        d, _ = world.cast_rays()
        n = len(d)
        front = d[n // 2 - self.half_cone: n // 2 + self.half_cone].min()
        if front < self.front_thresh:
            if self.turn_dir == 0.0:
                left, right = d[n // 2:].mean(), d[: n // 2].mean()
                self.turn_dir = 1.0 if left >= right else -1.0
            return self.turn_dir, 0.5
        self.turn_dir = 0.0
        goal_heading = np.arctan2(-world.y, c.length - world.x)
        steer = float(np.clip(wrap(goal_heading - world.h) / self.gain, -1.0, 1.0))
        return steer, 1.0
