"""
Evaluate consensus predictions against ground truth labels.
Compares: single-model, per-version consensus (≥2/3 seeds), and full ensemble.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_yolo_labels(label_dir):
    """Load all YOLO-format labels from a directory. Returns {stem: [[cls, cx, cy, w, h]]}"""
    labels = {}
    for txt_path in Path(label_dir).glob("*.txt"):
        stem = txt_path.stem
        boxes = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    boxes.append([cls_id, cx, cy, w, h])
        labels[stem] = boxes
    return labels


def yolo_to_xyxy(box, img_w=500, img_h=500):
    """Convert YOLO normalized [cls, cx, cy, w, h] to [x1, y1, x2, y2, cls] in pixels."""
    cls_id, cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2, cls_id]


def compute_iou(box1, box2):
    """IoU between [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0.0


def ap_per_class(tp, conf, pred_cls, target_cls):
    """Compute average precision per class (simplified COCO-style mAP)."""
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    ap = {}
    for c in unique_classes:
        mask = pred_cls == c
        n_gt = max(1, np.sum(target_cls == c))
        if np.sum(mask) == 0:
            ap[int(c)] = 0.0
            continue
        tp_c = np.cumsum(tp[mask])
        fp_c = np.cumsum(~tp[mask])
        rec = tp_c / n_gt
        prec = tp_c / np.maximum(tp_c + fp_c, np.finfo(np.float64).eps)
        # 11-point interpolation
        ap_c = 0
        for t in np.linspace(0, 1, 11):
            prec_at_t = prec[rec >= t]
            if len(prec_at_t) > 0:
                ap_c += np.max(prec_at_t) / 11
        ap[int(c)] = ap_c
    return ap


def evaluate_detections(gt_labels, pred_detections, iou_thresh=0.5):
    """
    Evaluate detections against ground truth.
    gt_labels: dict {stem: [[cls, cx, cy, w, h], ...]}
    pred_detections: dict {stem: [[x1, y1, x2, y2, conf, cls], ...]}
    """
    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_target_cls = []
    per_class = defaultdict(lambda: {"tp": [], "fp": [], "n_gt": 0})

    for stem, gt_boxes in gt_labels.items():
        gt_xyxy = [yolo_to_xyxy(b) for b in gt_boxes]
        preds = pred_detections.get(stem, [])

        # Match predictions to ground truth
        matched_gt = set()
        for pred in preds:
            px1, py1, px2, py2, pconf, pcls = pred
            best_iou = 0
            best_gt = -1
            for j, gt in enumerate(gt_xyxy):
                if j in matched_gt:
                    continue
                if gt[4] != pcls:
                    continue
                iou = compute_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j

            tp_flag = best_iou >= iou_thresh and best_gt >= 0
            if tp_flag:
                matched_gt.add(best_gt)

            all_tp.append(tp_flag)
            all_conf.append(pconf)
            all_pred_cls.append(pcls)
            all_target_cls.extend([b[4] for b in gt_xyxy])
            per_class[pcls]["tp"].append(tp_flag)
            per_class[pcls]["fp"].append(not tp_flag)

        for b in gt_xyxy:
            per_class[b[4]]["n_gt"] += 1

    if len(all_tp) == 0:
        return {"mAP50": 0.0, "per_class": {}}

    all_tp = np.array(all_tp)
    all_conf = np.array(all_conf)
    all_pred_cls = np.array(all_pred_cls)

    # Overall mAP
    ap_dict = ap_per_class(all_tp, all_conf, all_pred_cls, np.array(all_target_cls))
    mAP = np.mean(list(ap_dict.values())) if ap_dict else 0.0

    # Per-class precision/recall
    per_class_result = {}
    for cls_id, data in per_class.items():
        tp_sum = sum(data["tp"])
        fp_sum = sum(data["fp"])
        n_gt = data["n_gt"]
        prec = tp_sum / max(1, tp_sum + fp_sum)
        rec = tp_sum / max(1, n_gt)
        per_class_result[int(cls_id)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "n_gt": n_gt,
        }

    return {"mAP50": round(mAP, 4), "per_class": per_class_result}


def main():
    BASE = Path(r"D:\claude code\document\data\arran")

    # Load consensus predictions
    consensus_file = BASE / "consensus_results" / "consensus_predictions.json"
    if not consensus_file.exists():
        print("No consensus predictions found. Run consensus_infer.py first.")
        return

    with open(consensus_file) as f:
        consensus_data = json.load(f)

    # Load ground truth
    val_label_dir = BASE / "yolo" / "labels" / "val"
    gt_labels = load_yolo_labels(val_label_dir)
    print(f"Ground truth images: {len(gt_labels)}")

    CLASS_NAMES = {0: "roundhouse", 1: "shieling", 2: "smallcairn"}

    for model_name in ["yolov8n", "yolo11n", "yolo12n"]:
        # Build predictions dict from consensus data
        pred_dict = {}
        per_seed_counts = []
        for img_name, model_results in consensus_data.items():
            if model_name in model_results:
                consensus = model_results[model_name]["consensus"]
                dets = []
                for d in consensus:
                    dets.append([*d["box"], d["conf"], d["cls"]])
                pred_dict[img_name] = dets
                per_seed_counts.append(model_results[model_name].get("per_seed_counts", []))

        if not pred_dict:
            print(f"\n{model_name}: No predictions")
            continue

        # Evaluate
        result = evaluate_detections(gt_labels, pred_dict)

        print(f"\n{'='*50}")
        print(f"  {model_name} - Consensus (>=2/3 seeds)")
        print(f"{'='*50}")
        print(f"  mAP50: {result['mAP50']:.4f}")
        for cls_id, cls_name in CLASS_NAMES.items():
            if cls_id in result["per_class"]:
                c = result["per_class"][cls_id]
                print(f"    {cls_name}: P={c['precision']:.3f} R={c['recall']:.3f} GT={c['n_gt']}")

        # Average per-seed raw detections
        if per_seed_counts:
            avg_raw = [sum(x) / len(x) for x in zip(*per_seed_counts)]
            print(f"  Avg raw dets per seed: {[round(x, 1) for x in avg_raw]}")

    # Also evaluate single best seed per model
    print(f"\n{'='*50}")
    print(f"  Single-model baselines (best.pt from training)")
    print(f"{'='*50}")

    # Find training summary
    summary_path = BASE / "runs" / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            training_results = json.load(f)
        print(f"\n  Training mAP50 scores:")
        print(f"  {'Run':<22} {'mAP50':>8} {'mAP50-95':>8}")
        print(f"  {'-'*40}")
        for run, metrics in sorted(training_results.items()):
            print(f"  {run:<22} {metrics['mAP50']:8.4f} {metrics['mAP50-95']:8.4f}")


if __name__ == "__main__":
    main()
