import numpy as np
import torch
from perception.model import DroNetResNet8


class CNNPolicy:
    """Loads a trained checkpoint and predicts (steering, collision probability) from one frame."""

    def __init__(self, ckpt_path="models/checkpoints/best.pt", device="cpu"):
        ck = torch.load(ckpt_path, map_location=device)
        self.model = DroNetResNet8(img_size=ck.get("img_size", 64)).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, frame_uint8):
        x = torch.from_numpy(frame_uint8.astype(np.float32) / 255.0)[None, None].to(self.device)
        s, c = self.model(x)
        return float(s.item()), float(torch.sigmoid(c).item())
