"""One command to run the whole project: data -> training -> evaluation.

    python run_all.py            # full run
    python run_all.py --quick    # tiny smoke-test run (a minute or two)
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small, fast run to check the setup")
    ap.add_argument("--episodes", type=int, default=200, help="data-collection episodes")
    ap.add_argument("--epochs", type=int, default=15, help="training epochs")
    ap.add_argument("--eval-runs", type=int, default=50, help="evaluation layouts")
    a = ap.parse_args()
    if a.quick:
        a.episodes, a.epochs, a.eval_runs = 30, 3, 10
    t0 = time.time()
    from data.generate_data import generate
    from models.train import train
    from evaluation.evaluate import evaluate
    print("\n--- Step 1/3: generating dataset ---");  generate(n_episodes=a.episodes)
    print("\n--- Step 2/3: training CNN ---");        train(epochs=a.epochs)
    print("\n--- Step 3/3: evaluating ---");          evaluate(n_runs=a.eval_runs)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Results are in the logs/ folder (CSV files and figures/).")


if __name__ == "__main__":
    main()
