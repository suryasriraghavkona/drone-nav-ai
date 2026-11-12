"""Run baseline, CNN (and an oracle reference) on identical layouts; save logs, tables and charts."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulation.world import World
from baseline.waypoint_nav import WaypointBaseline
from planning.expert import expert_steer, expert_speed
from control.command import CommandTranslator

COLORS = {"baseline": "tab:orange", "cnn": "tab:blue", "oracle": "tab:green"}


def run_episode(seed, method, policy=None):
    w = World(seed)
    base, tr = WaypointBaseline(), CommandTranslator()
    rows, path = [], []
    for t in range(w.cfg.max_steps):
        ps, pc = np.nan, np.nan
        if method == "baseline":
            steer, speed = base.act(w)
        elif method == "oracle":
            steer, speed = expert_steer(w), expert_speed(w)
        else:
            ps, pc = policy.predict(w.render())
            steer, speed = tr(ps, pc)
        status = w.step(steer, speed)
        rows.append(dict(run_id=f"{method}_{seed}", method=method, seed=seed, step=t,
                         timestamp=round((t + 1) * w.cfg.dt, 2), x=round(w.x, 3), y=round(w.y, 3),
                         heading=round(w.h, 3), velocity=round(w.last_speed, 3),
                         predicted_steering=ps, predicted_collision_prob=pc))
        path.append((w.x, w.y))
        if status != "running":
            break
    outcome = {"success": "success", "collision": "collision"}.get(w.status, "timeout")
    eff = float(np.hypot(w.x, w.y) / w.path_len) if outcome == "success" and w.path_len > 0 else np.nan
    res = dict(run_id=f"{method}_{seed}", method=method, seed=seed, outcome=outcome,
               success=int(outcome == "success"), collision_count=int(outcome == "collision"),
               path_efficiency=eff, steps=w.steps, time_s=round(w.steps * w.cfg.dt, 2))
    return res, rows, np.array(path)


def evaluate(n_runs=50, ckpt="models/checkpoints/best.pt", log_dir="logs", base_seed=10000):
    methods = ["baseline"]
    policy = None
    if os.path.exists(ckpt):
        from perception.inference import CNNPolicy
        policy = CNNPolicy(ckpt); methods.append("cnn")
    else:
        print(f"[eval] no checkpoint at {ckpt}: evaluating baseline and oracle only")
    methods.append("oracle")
    results, flight, paths = [], [], {}
    for i in range(n_runs):
        seed = base_seed + i
        for m in methods:
            r, rows, p = run_episode(seed, m, policy)
            results.append(r); flight += rows; paths[(m, seed)] = p
        if (i + 1) % 10 == 0:
            print(f"[eval] {i+1}/{n_runs} layouts done")
    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    res = pd.DataFrame(results); res.insert(0, "id", range(1, len(res) + 1))
    res.to_csv(os.path.join(log_dir, "evaluation_results.csv"), index=False)
    fl = pd.DataFrame(flight); fl.insert(0, "id", range(1, len(fl) + 1))
    fl.to_csv(os.path.join(log_dir, "flight_logs.csv"), index=False)
    summ = res.groupby("method").agg(
        runs=("seed", "count"), success_rate=("success", "mean"), collision_rate=("collision_count", "mean"),
        timeout_rate=("outcome", lambda s: (s == "timeout").mean()),
        mean_path_efficiency=("path_efficiency", "mean"), mean_time_s=("time_s", "mean")).reindex(methods).round(3)
    summ.to_csv(os.path.join(log_dir, "evaluation_summary.csv"))
    print("\n=== RESULTS (same layouts for every method) ===\n" + summ.to_string() + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, col, title in zip(ax, ["success_rate", "collision_rate", "mean_path_efficiency"],
                             ["Success rate", "Collision rate", "Path efficiency (successful runs)"]):
        a.bar(summ.index, summ[col].fillna(0), color=[COLORS[m] for m in summ.index]); a.set_title(title)
        a.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(log_dir, "figures", "comparison.png"), dpi=150); plt.close(fig)

    seed = base_seed
    w = World(seed)
    fig, a = plt.subplots(figsize=(10, 3.2))
    for x, y, r, v in zip(w.ox0, w.oy0, w.rad, w.vy0):
        a.add_patch(plt.Circle((x, y), r, color="gray", alpha=0.7 if v == 0 else 0.35, ls="--" if v else "-"))
    for m in methods:
        p = paths[(m, seed)]; a.plot(p[:, 0], p[:, 1], color=COLORS[m], label=m)
    a.axhline(w.cfg.width / 2, color="k"); a.axhline(-w.cfg.width / 2, color="k")
    a.set_xlim(0, w.cfg.length); a.set_ylim(-w.cfg.width / 2 - 0.5, w.cfg.width / 2 + 0.5); a.set_aspect("equal")
    a.set_title(f"Flight paths, layout seed {seed} (faded circles = moving obstacles at start)"); a.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(log_dir, "figures", "trajectories.png"), dpi=150); plt.close(fig)
    return summ


if __name__ == "__main__":
    evaluate()
