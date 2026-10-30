# Plant Disease Detection — M4 Prototype

A web app that takes a plant leaf photo, runs the trained YOLOv8 model from M3 on it and shows
the **disease class**, the **bounding box** and the **confidence score**.

## Files

| File | Purpose |
|---|---|
| `detector.py` | Inference pipeline: loads the model, checks the class mapping, runs detection |
| `advisor.py` | Plant Q&A chat with Qwen 3.8 27B (vision) on Groq: the user asks questions, the model sees the photo and detections |
| `streamlit_app.py` | Streamlit web UI: upload, camera or example image, pick a model and threshold, see the results |
| `requirements.txt`, `packages.txt`, `.streamlit/config.toml` | Python packages, system package (libGL for OpenCV) and theme for Streamlit Cloud |
| `test_prototype.py` | Batch test: runs every image through both models and writes `results/test_log.csv` and the annotated images |
| `verify_claims.py` | Reproduces the numbers in the report: misdiagnosis rate and error breakdown on the test set, and mAP with the original vs corrected class names. Writes `results/claims_check.txt` and `results/test_errors.csv` |
| `config/class_names_corrected.txt` | Verified class names (see the class-name fix section) |
| `demo_images/` | 12 test-set images that both models get right, for the live demo |
| `results/` | Test logs, annotated images and a UI screenshot |
| `models/`, `config/class_names.txt`, `sample_images/`, `evaluation_reference/` | Files handed over by M3 (unchanged) |

## How to run

```bash
pip install -r requirements.txt      # CPU PyTorch, ultralytics, streamlit (no GPU needed)
streamlit run streamlit_app.py        # web UI, open http://localhost:8501
python test_prototype.py              # batch test on sample_images/
python test_prototype.py path/to/img_folder   # batch test on your own images
python verify_claims.py path/to/Plantdoc.v2i.yolov8   # re-check the report's numbers (needs the dataset)
```

**Online demo (Streamlit Cloud):** deploy this repo with main file `streamlit_app.py` and
Python 3.12. `packages.txt` installs libGL, which OpenCV needs on the Streamlit Cloud image.

**AI plant Q&A (optional):** get a free API key at https://console.groq.com/keys and
put it in `.streamlit/secrets.toml` (git-ignored), or in the app's **Settings → Secrets** on
Streamlit Cloud:

```toml
GROQ_API_KEY = "your-key"
# GROQ_MODEL = "qwen/qwen3.8-27b"   # optional; must be a Groq model that accepts images
```

Without a key the app still works; only the "Ask about this plant" section shows a setup hint.

## Workflow

Input image (upload/camera/example) → `detector.detect()` → YOLOv8 (imgsz 640, conf 0.25, IoU 0.7)
→ annotated image + summary (Healthy / Disease detected) + table (class, status, confidence, box) + timing

- Settings come from `config/inference_settings.txt`. You can change the confidence threshold with the slider in the UI.
- When a model is first loaded, the app checks that each model is the expected one (its built-in names match `class_names.txt`), then shows the names from `class_names_corrected.txt`.
- The 10 classes without a disease in their name (e.g. `Apple leaf`, `Tomato leaf`) are shown as **Healthy**.

## ⚠️ Class-name fix (important)

The name list in the Roboflow `data.yaml` does **not** match the label IDs in the label files.
The models were trained with that yaml, so their built-in names are wrong too. Examples:
ID 10 is a strawberry leaf, but the yaml calls it "Peach leaf"; ID 24 is a cherry leaf, but the yaml calls it "Tomato mold leaf".

We checked it two ways:
1. Keywords in the original image filenames for each label ID (e.g. 70 of the 94 unique source images labelled ID 10
   contain "strawberry"). Across all 28 IDs, the corrected names match 1,268 filenames and the original names only 249.
2. Looking at cropped label boxes for every ID (e.g. ID 20 images carry a "Mite Damage" watermark).

With the corrected names, the dataset labels of all 5 sample images match what the photos show.

- `config/class_names.txt`: original (wrong) list, kept so we can check the model is the expected one
- `config/class_names_corrected.txt`: verified mapping, **used by the prototype**

The label IDs were always consistent, so the models learned correctly and **no retraining is needed**.
The overall metrics (mAP etc.) are unchanged: `verify_claims.py` evaluates the test set with both name lists and gets
identical results (mAP@50 0.5305 for YOLOv8s, 0.4915 for YOLOv8n), because YOLO matches predictions by class ID. Only the per-class names in M3's reports
(confusion matrix, per-class AP) need to be relabelled with the corrected list.

## Test results (corrected names)

Full test set, 90 images, CPU. This reproduces M3's metrics exactly:

| Model | Precision | Recall | mAP@50 | mAP@50-95 | Image-level: main disease correct | Image-level: plant correct |
|---|---|---|---|---|---|---|
| YOLOv8s | 0.488 | 0.490 | 0.531 | 0.355 | 52/90 (57.8%) | 67/90 (74.4%) |
| YOLOv8n | 0.518 | 0.498 | 0.492 | 0.335 | 59/90 (65.6%) | 74/90 (82.2%) |

"Image-level correct" means the highest-confidence box (conf ≥ 0.25) has one of the image's ground-truth classes.
Per-image results are in `results/test_set_results.csv`.

**Misdiagnosed test images** (`results/claims_check.txt`, details in `results/test_errors.csv`):

| Model | Misdiagnosed | Wrong plant | Same plant, other disease | Diseased predicted healthy | Healthy predicted diseased | No detection |
|---|---|---|---|---|---|---|
| YOLOv8n | 31/90 (34.4%) | 15 | 10 | 5 | 0 | 1 |
| YOLOv8s | 38/90 (42.2%) | 23 | 11 | 2 | 2 | 0 |

- **Strong classes:** Corn rust (7/7), Blueberry, Apple rust, Cherry, Squash powdery mildew
- **Weak classes:** Potato late blight (1/8 for v8s), Tomato early blight, Peach, Bell_pepper leaf spot

Sample images (`results/test_log.csv`): strawberry ✓, apple rust ✓, peach → Cherry ✗,
tomato yellow virus → Tomato mosaic ✗, bell pepper leaf spot → Tomato leaf (v8s) / Bell_pepper leaf (v8n) ✗.

**Runtime (CPU, after warm-up):** about 120–250 ms per image for YOLOv8s and 75–115 ms for YOLOv8n.

**Model choice:** v8s has the higher mAP, but v8n gets more images right at the default
threshold and is about 2× faster on CPU. v8n is a good choice for the live demo.

**Demo images:** `demo_images/` has 12 test-set images from 12 different classes. Both models
classify them correctly with confidence ≥ 0.77. They are shown as examples in the UI.

**Issues found and fixed**
- The class names were wrong (see above). Fixed with `class_names_corrected.txt`.
- The first UI (Gradio) passed images as RGB numpy arrays, but Ultralytics reads numpy arrays as BGR.
  That gave wrong confidences. Fixed by passing PIL images (the Streamlit app does the same).
- The first prediction after loading took several seconds. Fixed by caching each model and
  warming it up on a sample image when it is first loaded.
- Streamlit Cloud could not import OpenCV (`cv2`) because libGL was missing. Fixed with `packages.txt`.
