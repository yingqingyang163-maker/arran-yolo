"""
Run inference with all 3 seeds per model version, then apply consensus:
a box is kept only if >= 2/3 seeds agree (IoU >= 0.5 with same class).
"""
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
BASE_DIR = Path(r"D:\claude code\document\data\arran")
RUNS_DIR = BASE_DIR / "runs"
DATA_YAML = BASE_DIR / "yolo" / "data.yaml"
VAL_IMAGES = BASE_DIR / "yolo" / "images" / "val"

OUTPUT_DIR = BASE_DIR / "consensus_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["yolov8n", "yolo11n", "yolo12n"]
SEEDS = [0, 1, 2]
CONF_THRESH = 0.25
IOU_CONSENSUS = 0.5
MIN_VOTES = 2  # at least 2/3 seeds must agree


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def consensus_filter(all_detections, iou_thresh=0.5, min_votes=2):
    """
    Given N sets of detections (one per seed), keep only boxes
    that have >= min_votes agreement across seeds.

    all_detections: list of lists, each inner list is detections from one seed
                    each detection is [x1, y1, x2, y2, conf, cls]
    Returns: list of consensus detections
    """
    n_seeds = len(all_detections)
    if n_seeds < min_votes:
        return []

    # Group detections across seeds by class
    by_class = defaultdict(list)
    for seed_idx, dets in enumerate(all_detections):
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            by_class[int(cls)].append((seed_idx, det))

    consensus = []
    for cls_id, seed_dets in by_class.items():
        # Build adjacency: which detections overlap across seeds
        used = set()
        n = len(seed_dets)
        for i in range(n):
            if i in used:
                continue
            seed_i, det_i = seed_dets[i]
            box_i = det_i[:4]
            cluster = [(seed_i, det_i)]
            used.add(i)

            # Find all detections that match this one (across different seeds)
            for j in range(i + 1, n):
                if j in used:
                    continue
                seed_j, det_j = seed_dets[j]
                if seed_j == seed_i:
                    continue  # same seed, don't vote
                box_j = det_j[:4]
                if compute_iou(box_i, box_j) >= iou_thresh:
                    cluster.append((seed_j, det_j))
                    used.add(j)

            # Count unique seeds in this cluster
            unique_seeds = len(set(s for s, _ in cluster))
            if unique_seeds >= min_votes:
                # Average the boxes and confidences
                boxes_xyxy = np.array([d[1][:4] for d in cluster])
                confs = np.array([d[1][4] for d in cluster])
                avg_box = boxes_xyxy.mean(axis=0).tolist()
                avg_conf = confs.mean()
                consensus.append([*avg_box, avg_conf, cls_id])

    return consensus


def load_model(model_name, seed):
    """Load a trained model from its run directory."""
    # Find the best.pt
    model_path = RUNS_DIR / f"{model_name}_seed{seed}" / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def predict(model, image_path, conf=0.25):
    """Run inference on a single image, return [x1,y1,x2,y2,conf,cls] in pixel coords."""
    results = model(str(image_path), conf=conf, verbose=False)
    detections = []
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, classes):
                detections.append([*box.tolist(), float(conf), int(cls)])
    return detections


def main():
    # Collect validation images
    val_images = sorted(VAL_IMAGES.glob("*.png"))
    print(f"Validation images: {len(val_images)}")

    all_image_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Inference: {model_name}")
        print(f"{'='*60}")

        # Load all 3 seed models
        seed_models = []
        for seed in SEEDS:
            try:
                model = load_model(model_name, seed)
                seed_models.append((seed, model))
                print(f"  Loaded seed {seed}")
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")

        if len(seed_models) < MIN_VOTES:
            print(f"  Not enough models for {model_name}, skipping")
            continue

        for img_path in val_images:
            img_name = img_path.stem
            all_seed_dets = []

            for seed, model in seed_models:
                dets = predict(model, img_path, conf=CONF_THRESH)
                all_seed_dets.append(dets)

            # Apply consensus
            consensus_dets = consensus_filter(all_seed_dets, IOU_CONSENSUS, MIN_VOTES)

            if img_name not in all_image_results:
                all_image_results[img_name] = {}
            all_image_results[img_name][model_name] = {
                "consensus": consensus_dets,
                "per_seed": [len(d) for d in all_seed_dets],
            }

        # Quick stats
        total_consensus = sum(
            len(v[model_name]["consensus"]) for v in all_image_results.values()
        )
        total_raw = sum(
            sum(v[model_name]["per_seed"]) for v in all_image_results.values()
        )
        print(f"  Total raw detections (all seeds): {total_raw}")
        print(f"  Total consensus detections: {total_consensus}")

    # Save results
    # Convert to serializable format
    serializable = {}
    for img_name, model_results in all_image_results.items():
        serializable[img_name] = {}
        for model_name, data in model_results.items():
            serializable[img_name][model_name] = {
                "consensus": [
                    {"box": d[:4], "conf": d[4], "cls": d[5]}
                    for d in data["consensus"]
                ],
                "per_seed_counts": data["per_seed"],
            }

    output_path = OUTPUT_DIR / "consensus_predictions.json"
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
