import numpy as np
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """Frames (uint8 HxW) with steering and collision labels.
    Training augmentation: brightness jitter + horizontal mirror (steering sign flips)."""

    def __init__(self, frames, steer, coll, augment=False):
        self.frames, self.steer, self.coll, self.augment = frames, steer, coll, augment

    def __len__(self):
        return len(self.steer)

    def __getitem__(self, i):
        img = self.frames[i].astype(np.float32) / 255.0
        s = float(self.steer[i])
        if self.augment:
            img = np.clip(img * np.random.uniform(0.9, 1.1), 0, 1)
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                s = -s
        return torch.from_numpy(img).unsqueeze(0), torch.tensor(s, dtype=torch.float32), torch.tensor(float(self.coll[i]), dtype=torch.float32)
