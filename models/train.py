"""Train the two-headed CNN. Loss = MSE(steering) + beta * BCE(collision), beta ramps up (as in DroNet)."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from perception.model import DroNetResNet8
from perception.dataset import FrameDataset


def train(data_path="data/frames.npz", epochs=15, batch_size=128, lr=1e-3,
          ckpt_dir="models/checkpoints", log_dir="logs", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(data_path)
    frames, steer, coll, ep = d["frames"], d["steer"], d["coll"], d["episode"]
    eps = np.unique(ep)
    np.random.default_rng(seed).shuffle(eps)
    n_val = max(1, int(0.2 * len(eps)))
    is_val = np.isin(ep, eps[:n_val])          # split by episode: no leakage between train/val
    tr = FrameDataset(frames[~is_val], steer[~is_val], coll[~is_val], augment=True)
    va = FrameDataset(frames[is_val], steer[is_val], coll[is_val], augment=False)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    vl = DataLoader(va, batch_size=batch_size)
    pos = float(coll[~is_val].sum()); neg = float((~is_val).sum() - pos)
    pw = torch.tensor(float(np.clip(neg / max(pos, 1.0), 1.0, 20.0)), device=device)
    print(f"[train] device={device} train={len(tr)} val={len(va)} collision-positive={int(pos)} pos_weight={pw.item():.1f}")

    model = DroNetResNet8().to(device)
    print(f"[train] parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)
    best, hist, ckpt_rows = 1e9, [], []
    for e in range(epochs):
        beta = min(1.0, 0.1 + e / max(1.0, 0.5 * epochs))
        model.train(); tls = tlc = n = 0
        for img, s, c in tl:
            img, s, c = img.to(device), s.to(device), c.to(device)
            ps, pc = model(img)
            ls = F.mse_loss(ps, s)
            lc = F.binary_cross_entropy_with_logits(pc, c, pos_weight=pw)
            loss = ls + beta * lc
            opt.zero_grad(); loss.backward(); opt.step()
            tls += ls.item() * len(s); tlc += lc.item() * len(s); n += len(s)
        model.eval(); vs = vc = vmae = m = 0; tp = fp = fn = correct = 0
        with torch.no_grad():
            for img, s, c in vl:
                img, s, c = img.to(device), s.to(device), c.to(device)
                ps, pc = model(img)
                vs += F.mse_loss(ps, s).item() * len(s)
                vc += F.binary_cross_entropy_with_logits(pc, c, pos_weight=pw).item() * len(s)
                vmae += (ps - s).abs().sum().item(); m += len(s)
                pred = (torch.sigmoid(pc) > 0.5).float()
                tp += ((pred == 1) & (c == 1)).sum().item(); fp += ((pred == 1) & (c == 0)).sum().item()
                fn += ((pred == 0) & (c == 1)).sum().item(); correct += (pred == c).sum().item()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        row = dict(epoch=e + 1, beta=round(beta, 3), train_steer_loss=tls / n, train_coll_loss=tlc / n,
                   val_steer_loss=vs / m, val_coll_loss=vc / m, val_steer_mae=vmae / m,
                   val_coll_acc=correct / m, val_coll_f1=f1)
        hist.append(row)
        vloss = row["val_steer_loss"] + row["val_coll_loss"]
        saved = ""
        if vloss < best:
            best = vloss; saved = os.path.join(ckpt_dir, "best.pt")
            torch.save({"model": model.state_dict(), "img_size": 64, "epoch": e + 1}, saved)
        ckpt_rows.append(dict(id=e + 1, epoch=e + 1, architecture="DroNetResNet8",
                              steering_loss=row["val_steer_loss"], collision_loss=row["val_coll_loss"], saved_path=saved))
        print(f"[train] epoch {e+1}/{epochs}  val steer MAE={row['val_steer_mae']:.3f}  "
              f"collision acc={row['val_coll_acc']:.3f}  F1={f1:.3f}" + ("  *saved*" if saved else ""))
    h = pd.DataFrame(hist)
    h.to_csv(os.path.join(log_dir, "training_history.csv"), index=False)
    pd.DataFrame(ckpt_rows).to_csv(os.path.join(log_dir, "model_checkpoints.csv"), index=False)
    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h.epoch, h.train_steer_loss, label="train"); ax[0].plot(h.epoch, h.val_steer_loss, label="val")
    ax[0].set_title("Steering loss (MSE)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h.epoch, h.train_coll_loss, label="train"); ax[1].plot(h.epoch, h.val_coll_loss, label="val")
    ax[1].set_title("Collision loss (weighted BCE)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(log_dir, "figures", "training_curves.png"), dpi=150); plt.close(fig)
    return h


if __name__ == "__main__":
    train()
