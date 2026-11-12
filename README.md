# AI-Based Autonomous Drone Navigation System

A DroNet-inspired CNN (ResNet-8, two heads) that looks at one camera frame from a simulated drone and predicts
**a steering command** and **a collision probability**. A command-translation module turns these into yaw and
speed commands so the drone avoids obstacles. The CNN is compared with a non-AI baseline
(waypoint following + reactive range-sensor avoidance) on identical obstacle layouts.

Everything runs on a normal laptop CPU. No GPU, Unreal Engine or AirSim is needed.

## Quick start

```bash
git clone <YOUR-REPO-URL>
cd drone-nav-ai
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --quick      # 1-2 minute check that everything works
python run_all.py              # full run (data -> training -> evaluation)
```

Results appear in `logs/` (CSV files) and `logs/figures/` (charts).

## What `run_all.py` does
1. **Data** (`data/generate_data.py`): flies the simulated drone with exploration policies and labels every frame
   automatically (steering = oracle planner's choice, collision = collides within the next 10 steps).
2. **Training** (`models/train.py`): trains the ResNet-8 with loss = MSE(steering) + beta * BCE(collision).
   Validation is split by episode. Best checkpoint -> `models/checkpoints/best.pt`.
3. **Evaluation** (`evaluation/evaluate.py`): runs baseline, CNN and an oracle reference on the same 50 unseen layouts.
   Metrics: success rate, collision rate, timeout rate, path efficiency, time to goal.

Run steps separately if you prefer: `python -m data.generate_data`, `python -m models.train`, `python -m evaluation.evaluate`.

## Folder structure
```
simulation/   lightweight simulator (world, obstacles, camera frame rendering)
data/         dataset generator; generated frames.npz, labels.csv, samples/
perception/   CNN model, dataset class, inference wrapper
models/       training script; checkpoints/
planning/     oracle potential-field planner (labels + reference)
baseline/     non-AI baseline navigation
control/      command translation; optional AirSim adapter (untested sketch)
evaluation/   evaluation, charts
logs/         flight_logs.csv, evaluation_results.csv, evaluation_summary.csv, training_history.csv, figures/
notebooks/    results_analysis.ipynb
tests/        smoke_test.py (no PyTorch needed)
run_all.py    one-command pipeline
```

## Output files
| File | Content |
|---|---|
| `logs/training_history.csv`, `logs/model_checkpoints.csv` | per-epoch losses, steering MAE, collision accuracy / F1 |
| `logs/flight_logs.csv` | position, velocity, predicted steering and collision probability per step |
| `logs/evaluation_results.csv` | one row per run: outcome, collisions, path efficiency |
| `logs/evaluation_summary.csv` | table comparing baseline, CNN and oracle |
| `logs/figures/` | training_curves.png, comparison.png, trajectories.png |

## Notes and limitations
* The simulator is a simplified 2D world; images are synthetic. Results show the method works *in this simulation*;
  transfer to real cameras is future work.
* The "oracle" uses true obstacle positions. It produces training labels and acts as an upper-bound reference only.
* Obstacle layouts are random; numbers vary a little between runs/seeds.
* `control/airsim_adapter.py` is an untested sketch for future AirSim use.

## Troubleshooting
* `ModuleNotFoundError: torch`: run `pip install torch` (CPU version is fine).
* Run commands from the repository root (the folder containing `run_all.py`).
* To run without PyTorch (baseline + oracle only): `python -m evaluation.evaluate` before training.
