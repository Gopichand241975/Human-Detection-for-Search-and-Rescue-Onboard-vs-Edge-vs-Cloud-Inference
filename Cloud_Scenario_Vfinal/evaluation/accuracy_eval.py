import json
import os
import glob
import re
import csv


PREDICTIONS_DIR = "data/predictions"
LABELS_DIR = "data/dataset/labels"
OUTPUT_DIR = "evaluation/results/accuracy"

# Client resizes to this before sending — model runs on this resolution
INFER_WIDTH = 640
INFER_HEIGHT = 480

IOU_THRESHOLD = 0.3

# All classes the model detects (mirrored from server/config/config.py CLASS_MAP)
# Must match CLASS_MAP exactly — extra class IDs here would inflate false-negative counts.
MODEL_CLASSES = {0, 1}


# ---------- extract run id ----------
def extract_run_id(path):
    match = re.search(r"run_(\d{8}_\d{6})", path)
    return match.group(1) if match else None


# ---------- extract image index from image_id ----------
def extract_image_index(image_id):
    # e.g. "run_20260425_171143_img_0001" -> 1
    match = re.search(r"img_(\d+)$", image_id)
    if not match:
        return None
    return int(match.group(1))


# ---------- load label file ----------
def load_labels(image_index, class_ids=None):
    """
    Load YOLO labels for one image.
    Returns boxes in normalized cx,cy,w,h format.
    If class_ids is given, only return boxes matching those classes.
    """
    label_file = os.path.join(LABELS_DIR, f"{image_index:03d}.txt")
    if not os.path.exists(label_file):
        return []

    boxes = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            if class_ids is not None and class_id not in class_ids:
                continue
            cx = float(parts[1])
            cy = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])
            boxes.append({"class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


# ---------- convert prediction bbox to normalized cx,cy,w,h ----------
def pred_bbox_to_normalized(bbox):
    """
    Predictions store absolute top-left (x, y, width, height) in the
    inference resolution (INFER_WIDTH x INFER_HEIGHT).
    Convert to normalized center cx,cy,w,h to match YOLO label format.
    """
    cx = (bbox["x"] + bbox["width"]  / 2) / INFER_WIDTH
    cy = (bbox["y"] + bbox["height"] / 2) / INFER_HEIGHT
    w  = bbox["width"]  / INFER_WIDTH
    h  = bbox["height"] / INFER_HEIGHT
    return cx, cy, w, h


# ---------- compute IoU between two boxes in cx,cy,w,h format ----------
def compute_iou(box_a, box_b):
    ax1 = box_a[0] - box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay2 = box_a[1] + box_a[3] / 2

    bx1 = box_b[0] - box_b[2] / 2
    by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2
    by2 = box_b[1] + box_b[3] / 2

    inter_area = (
        max(0.0, min(ax2, bx2) - max(ax1, bx1)) *
        max(0.0, min(ay2, by2) - max(ay1, by1))
    )
    union_area = (
        (ax2 - ax1) * (ay2 - ay1) +
        (bx2 - bx1) * (by2 - by1) -
        inter_area
    )

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ---------- match predictions to ground truth ----------
def match_predictions(preds, gts, iou_threshold):
    tp = 0
    matched_gt = set()

    for pred in preds:
        pred_box = pred_bbox_to_normalized(pred["bbox"])
        best_iou    = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue
            if gt["class_id"] != pred["class_id"]:
                continue
            gt_box = (gt["cx"], gt["cy"], gt["w"], gt["h"])
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou    = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


# ---------- load predictions for a run ----------
def load_run_predictions(run_id):
    pred_dir = os.path.join(PREDICTIONS_DIR, f"run_{run_id}")
    if not os.path.isdir(pred_dir):
        return []

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.json")))
    predictions = []
    for f in pred_files:
        with open(f, "r") as fh:
            data = json.load(fh)
        predictions.append(data)
    return predictions


# ---------- evaluate a single run ----------
def evaluate_run(run_id):
    predictions = load_run_predictions(run_id)

    if not predictions:
        print(f"[SKIP] run_{run_id}: no predictions found")
        return

    rows = []
    total_tp = total_fp = total_fn = 0

    for pred_data in predictions:
        image_id = pred_data.get("image_id", "")
        image_index = extract_image_index(image_id)

        if image_index is None:
            print(f"  [WARN] cannot parse image index from '{image_id}'")
            continue

        preds = pred_data.get("detections", [])

        # Always evaluate against the full set of model classes,
        # so missed detections (FNs) are counted even when preds=[]
        gts = load_labels(image_index, class_ids=MODEL_CLASSES)

        tp, fp, fn = match_predictions(preds, gts, IOU_THRESHOLD)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append({
            "image_id":   image_id,
            "gt_count":   len(gts),
            "pred_count": len(preds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # aggregate row
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (
        2 * agg_precision * agg_recall / (agg_precision + agg_recall)
        if (agg_precision + agg_recall) > 0
        else 0.0
    )

    rows.append({
        "image_id":   "__aggregate__",
        "gt_count":   sum(r["gt_count"]   for r in rows),
        "pred_count": sum(r["pred_count"] for r in rows),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(agg_precision, 4),
        "recall":    round(agg_recall,    4),
        "f1":        round(agg_f1,        4),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_file = os.path.join(OUTPUT_DIR, f"run_{run_id}_accuracy.csv")
    fieldnames = [
        "image_id", "gt_count", "pred_count",
        "tp", "fp", "fn", "precision", "recall", "f1"
    ]

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Saved {out_file}  (images={len(rows)-1}, F1={agg_f1:.4f})")


# ---------- get all runs ----------
def get_all_runs():
    dirs = glob.glob(os.path.join(PREDICTIONS_DIR, "run_*/"))
    runs = []
    for d in dirs:
        run_id = extract_run_id(os.path.basename(os.path.normpath(d)))
        if run_id:
            runs.append(run_id)
    return sorted(set(runs))


# ---------- main ----------
def main():
    runs = get_all_runs()

    if not runs:
        print("[ERROR] No runs found in", PREDICTIONS_DIR)
        return

    for run_id in runs:
        evaluate_run(run_id)


if __name__ == "__main__":
    main()
