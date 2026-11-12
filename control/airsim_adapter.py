"""OPTIONAL / FUTURE WORK (not used or tested in this project).

Sketch of how the same CommandTranslator output could drive a real AirSim drone.
Requires `pip install airsim` and a running AirSim (Unreal) environment.
"""


class AirSimAdapter:
    def __init__(self, v_max=2.0, max_yaw_rate_deg=85.0):
        import airsim  # imported lazily so the rest of the project does not need it
        self.airsim = airsim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.v_max, self.max_yaw_rate_deg = v_max, max_yaw_rate_deg

    def send(self, steer, speed_frac):
        self.client.moveByVelocityBodyFrameAsync(
            speed_frac * self.v_max, 0.0, 0.0, 0.1,
            yaw_mode=self.airsim.YawMode(True, steer * self.max_yaw_rate_deg))

    def get_frame(self):
        raise NotImplementedError("Grab a camera image with client.simGetImages(...) and resize to 64x64 grayscale.")
