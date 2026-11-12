"""Command translation: CNN outputs -> (steering/yaw command, speed fraction).

* steering is smoothed with an exponential moving average to reduce jitter,
* forward speed is scaled by (1 - collision probability), with a floor,
* above `brake_thresh` the drone crawls and steers harder (corrective manoeuvre).
"""
import numpy as np


class CommandTranslator:
    def __init__(self, alpha=0.6, v_min=0.2, brake_thresh=0.8, boost=1.5):
        self.alpha, self.v_min, self.brake_thresh, self.boost = alpha, v_min, brake_thresh, boost
        self.s = 0.0

    def reset(self):
        self.s = 0.0

    def __call__(self, steer, p_coll):
        self.s = self.alpha * steer + (1 - self.alpha) * self.s
        cmd = self.s
        speed = max(self.v_min, 1.0 - p_coll)
        if p_coll > self.brake_thresh:
            speed = self.v_min
            cmd = cmd * self.boost
        return float(np.clip(cmd, -1, 1)), float(speed)
