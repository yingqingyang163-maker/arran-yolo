"""
Fine-tune YOLOv8, YOLOv11, YOLOv12 on Arran hillshade data.
Each version × 3 random seeds (0, 1, 2) = 9 models total.
Uses pretrained weights for transfer learning.
"""
import os, sys, json, time
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
BASE_DIR = Path(r"D:\claude code\document\data\arran")
DATA_YAML = BASE_DIR / "yolo" / "data.yaml"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 50
PATIENCE = 15
IMGSZ = 500
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_VERSIONS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo12n": "yolo12n.pt",
}
SEEDS = [0, 1, 2]

# ── Main ────────────────────────────────────────────────────
results_log = {}

print(f"Device: {DEVICE}")
print(f"Models: {list(MODEL_VERSIONS.keys())}")
print(f"Seeds: {SEEDS}")
print(f"Data: {DATA_YAML}")
print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}, ImgSz: {IMGSZ}, Batch: {BATCH}")
print("=" * 60)

for model_name, weights in MODEL_VERSIONS.items():
    for seed in SEEDS:
        run_name = f"{model_name}_seed{seed}"
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training {run_name}")
        print(f"{'='*60}")

        # Load pretrained model
        model = YOLO(weights)

        # Train
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            seed=seed,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            verbose=True,
            # Use deterministic ops for reproducibility
            deterministic=True,
        )

        # Log summary
        key = run_name
        results_log[key] = {
            "model": model_name,
            "seed": seed,
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            "best_epoch": int(getattr(results, "best_epoch", -1)),
        }
        print(f"  mAP50={results_log[key]['mAP50']:.4f}  mAP50-95={results_log[key]['mAP50-95']:.4f}")

# ── Save summary ────────────────────────────────────────────
summary_path = RUNS_DIR / "training_summary.json"
with open(summary_path, "w") as f:
    json.dump(results_log, f, indent=2)
print(f"\nSummary saved to {summary_path}")

# Print comparison table
print("\n" + "=" * 70)
print(f"{'Run':<20} {'mAP50':>8} {'mAP50-95':>8} {'Precision':>10} {'Recall':>8}")
print("-" * 70)
for run, m in results_log.items():
    print(f"{run:<20} {m['mAP50']:8.4f} {m['mAP50-95']:8.4f} {m['precision']:10.4f} {m['recall']:8.4f}")
