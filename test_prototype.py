"""Batch-test the prototype pipeline and record results.

Runs every image in a folder through each model and writes
results/test_log.csv plus annotated images in results/<model>/.

Usage:  python test_prototype.py [image_folder]
"""
import csv
import sys
import time
from pathlib import Path

from PIL import Image

from detector import DEFAULT_CONF, MODELS, ROOT, detect, get_model

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_images"
images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)

rows = []
for model_key in MODELS:
    get_model(model_key)
    detect(images[0], model_key)  # warm-up so the first timing is fair
    sub = out_dir / model_key.split()[0]
    sub.mkdir(exist_ok=True)
    print(f"\n== {model_key}")
    for img in images:
        start = time.perf_counter()
        annotated, dets, infer_ms = detect(str(img), model_key, DEFAULT_CONF)
        Image.fromarray(annotated).save(sub / img.name)
        total_ms = (time.perf_counter() - start) * 1000
        top = dets[0] if dets else None
        print(f"{img.name[:45]:45s} {total_ms:6.0f} ms  "
              + (f"{top.label} ({top.confidence:.2f}), {len(dets)} det(s)" if top else "no detection"))
        rows.append({
            "model": model_key.split()[0],
            "image": img.name,
            "num_detections": len(dets),
            "top_class": top.label if top else "",
            "top_confidence": f"{top.confidence:.3f}" if top else "",
            "all_detections": "; ".join(f"{d.label} {d.confidence:.2f} {list(d.box)}" for d in dets),
            "inference_ms": f"{infer_ms:.1f}",
            "end_to_end_ms": f"{total_ms:.1f}",
            "expected_class": "",   # fill in manually
            "correct": "",          # fill in manually (Y/N)
        })

with open(out_dir / "test_log.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {out_dir / 'test_log.csv'} and annotated images in {out_dir}")
