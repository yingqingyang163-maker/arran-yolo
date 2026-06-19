"""
Master pipeline: run after training completes.
1. Runs consensus inference (3 seeds per version → ≥2/3 voting)
2. Evaluates consensus results vs ground truth
3. Prints comparison report
"""
import json, sys
from pathlib import Path
from datetime import datetime

BASE = Path(r"D:\claude code\document\data\arran")
RUNS_DIR = BASE / "runs"

def check_training_complete():
    """Check if all 9 models finished training."""
    models = ["yolov8n", "yolo11n", "yolo12n"]
    seeds = [0, 1, 2]

    missing = []
    for m in models:
        for s in seeds:
            best_pt = RUNS_DIR / f"{m}_seed{s}" / "weights" / "best.pt"
            if not best_pt.exists():
                missing.append(f"{m}_seed{s}")

    if missing:
        print(f"Missing {len(missing)} models: {missing}")
        return False
    print(f"All 9 models found.")
    return True

def main():
    print("=" * 60)
    print("Arran YOLO Consensus Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 0: Check training
    if not check_training_complete():
        print("\nTraining not complete. Showing current progress:")
        summary = RUNS_DIR / "training_summary.json"
        if summary.exists():
            with open(summary) as f:
                data = json.load(f)
            for run, m in sorted(data.items()):
                print(f"  {run:<22} mAP50={m['mAP50']:.4f}")
        sys.exit(0)

    # Step 1: Consensus inference
    print("\n[Step 1] Running consensus inference...")
    import consensus_infer
    consensus_infer.main()

    # Step 2: Evaluate
    print("\n[Step 2] Evaluating consensus results...")
    import evaluate_consensus
    evaluate_consensus.main()

    print(f"\nPipeline complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
