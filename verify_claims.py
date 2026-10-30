"""Reproduce the numerical claims made in the report (Sections 6.1 and 6.2).

1. Image-level misdiagnosis rate on the 90 test images, per model.
2. Breakdown of those errors (wrong plant / same plant, other disease / ...).
3. That mAP is identical with the original and the corrected class-name list.

Usage:  python verify_claims.py path/to/Plantdoc.v2i.yolov8
Writes results/claims_check.txt and results/test_errors.csv.
"""
import collections
import csv
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

from detector import CORRECTED_NAMES, HEALTHY_CLASSES, MODELS, ORIGINAL_NAMES, ROOT, detect, load_class_names

DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent / "dataset" / "Plantdoc.v2i.yolov8"
OUT = ROOT / "results"
names = load_class_names(CORRECTED_NAMES)
species = lambda name: name.split()[0].lower()
lines, error_rows = [], []


def log(text=""):
    print(text)
    lines.append(text)


def error_type(gt, pred):
    if not pred:
        return "no detection"
    same = [g for g in gt if species(g) == species(pred)]
    if not same:
        return "wrong plant species"
    if pred in HEALTHY_CLASSES:
        return "same plant: diseased leaf predicted healthy"
    if all(g in HEALTHY_CLASSES for g in same):
        return "same plant: healthy leaf predicted diseased"
    return "same plant: confused with another disease"


# 1 + 2. Image-level errors: the top prediction (highest-confidence box, conf >= 0.25)
# is correct if its class is one of the classes annotated in that image.
images = sorted((DATA / "test" / "images").glob("*.jpg"))
for model_key in MODELS:
    errors = collections.Counter()
    for img in images:
        label = DATA / "test" / "labels" / (img.stem + ".txt")
        gt = {names[int(l.split()[0])] for l in label.read_text().splitlines() if l.strip()}
        _, dets, _ = detect(str(img), model_key)
        pred = dets[0].label if dets else ""
        if pred not in gt:
            kind = error_type(gt, pred)
            errors[kind] += 1
            error_rows.append([model_key, img.name, " / ".join(sorted(gt)), pred,
                               f"{dets[0].confidence:.3f}" if dets else "", kind])
    total = sum(errors.values())
    log(f"{model_key}: {total}/{len(images)} test images misdiagnosed ({total / len(images):.1%})")
    for kind, n in errors.most_common():
        log(f"    {kind:45s} {n:3d}  ({n / total:.0%} of errors)")

# 3. mAP with the original vs the corrected names (names are metadata only).
log()
for tag, path in [("original names ", ORIGINAL_NAMES), ("corrected names", CORRECTED_NAMES)]:
    cfg = OUT / f"_data_{tag.split()[0]}.yaml"
    yaml.safe_dump({"path": str(DATA), "train": "train/images", "val": "valid/images", "test": "test/images",
                    "nc": 28, "names": list(load_class_names(path).values())}, open(cfg, "w"), sort_keys=False)
    for model_key, weights in MODELS.items():
        m = YOLO(str(weights)).val(data=str(cfg), split="test", imgsz=640, batch=16, conf=0.001, iou=0.7,
                                   verbose=False, plots=False, project=str(OUT / "_val"), name=tag.split()[0], exist_ok=True)
        log(f"{tag} {model_key.split()[0]}: mAP@50={m.box.map50:.4f}  mAP@50-95={m.box.map:.4f}")
    cfg.unlink()
shutil.rmtree(OUT / "_val", ignore_errors=True)

(OUT / "claims_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(OUT / "test_errors.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "image", "ground_truth_classes", "top_prediction", "confidence", "error_type"])
    writer.writerows(error_rows)
print(f"\nSaved {OUT / 'claims_check.txt'} and {OUT / 'test_errors.csv'}")
