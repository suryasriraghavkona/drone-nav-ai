"""Generate the training dataset from the simulator (no manual labelling).

Each episode is flown with an *exploration* policy (noisy oracle or random wander)
so that collisions actually happen. For every frame:
  * steering label  = what the oracle planner would steer at that moment,
  * collision label = 1 if the drone collides within the next `horizon` steps.
Output: data/frames.npz, data/labels.csv, data/samples/*.png
"""
import os
import numpy as np
import pandas as pd
from simulation.world import World
from planning.expert import expert_steer


def fly_episode(seed, rng, horizon=10, max_steps=250):
    w = World(seed)
    mode = rng.choice(["noisy_oracle", "wander", "reckless"])
    frames, steer, step_ids = [], [], []
    wander = 0.0
    for t in range(max_steps):
        frames.append(w.render(noise=3.0, rng=rng))
        label = expert_steer(w)
        steer.append(label)
        step_ids.append(t)
        if mode == "noisy_oracle":
            action = np.clip(label + rng.normal(0, 0.5), -1, 1)
        elif mode == "reckless":      # head for the goal and ignore obstacles -> many collisions
            action = np.clip(-1.5 * w.h - 0.3 * w.y + rng.normal(0, 0.2), -1, 1)
        else:
            wander = 0.8 * wander + rng.normal(0, 0.4)
            action = np.clip(wander + 0.3 * (-w.h), -1, 1)
        speed = 1.0 if mode == "reckless" else rng.uniform(0.5, 1.0)
        if w.step(action, speed) != "running":
            break
    n = len(frames)
    coll = np.zeros(n, dtype=np.float32)
    if w.status == "collision":
        coll[max(0, n - horizon):] = 1.0
    return np.stack(frames), np.array(steer, dtype=np.float32), coll, np.array(step_ids), w.status


def generate(n_episodes=200, out_dir="data", seed=0):
    rng = np.random.default_rng(seed)
    all_f, all_s, all_c, all_e, all_t = [], [], [], [], []
    outcomes = {"collision": 0, "success": 0, "running": 0}
    for ep in range(n_episodes):
        f, s, c, t, status = fly_episode(ep, rng)
        outcomes[status] += 1
        all_f.append(f); all_s.append(s); all_c.append(c)
        all_t.append(t); all_e.append(np.full(len(s), ep))
    frames = np.concatenate(all_f)
    steer = np.concatenate(all_s)
    coll = np.concatenate(all_c)
    episode = np.concatenate(all_e)
    step = np.concatenate(all_t)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "frames.npz"), frames=frames, steer=steer, coll=coll, episode=episode)
    df = pd.DataFrame({
        "id": np.arange(len(steer)), "episode": episode, "step": step,
        "timestamp": np.round(step * World().cfg.dt, 2),
        "image_path": [f"frames.npz[{i}]" for i in range(len(steer))],
        "steering_label": np.round(steer, 4), "collision_label": coll.astype(int)})
    df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    # a few sample images for the report / slide
    import cv2
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    for i in np.linspace(0, len(frames) - 1, 6).astype(int):
        big = cv2.resize(frames[i], (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, "samples", f"frame_{i}_steer{steer[i]:+.2f}_coll{int(coll[i])}.png"), big)
    print(f"[data] {len(steer)} frames from {n_episodes} episodes "
          f"(collision-positive frames: {int(coll.sum())}, episode outcomes: {outcomes})")
    return frames.shape[0]


if __name__ == "__main__":
    generate()
