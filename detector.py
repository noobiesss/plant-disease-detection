"""Inference pipeline for the plant disease detection prototype.

Shared by the Gradio UI (app.py) and the batch test script (test_prototype.py),
so both use exactly the same model loading, settings and post-processing.
"""
import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODELS = {
    "YOLOv8s (higher mAP)": ROOT / "models" / "yolov8s_best.pt",
    "YOLOv8n (faster)": ROOT / "models" / "yolov8n_best.pt",
}
DEFAULT_MODEL = "YOLOv8s (higher mAP)"
IMG_SIZE = 640       # from config/inference_settings.txt
DEFAULT_CONF = 0.25  # from config/inference_settings.txt
IOU = 0.7

# Classes that describe a healthy leaf (no disease in the name).
HEALTHY_CLASSES = {
    "Apple leaf", "Bell_pepper leaf", "Blueberry leaf", "Cherry leaf",
    "Peach leaf", "Raspberry leaf", "Soyabean leaf", "Strawberry leaf",
    "Tomato leaf", "grape leaf",
}


# The models were trained with the Roboflow data.yaml, whose name list does not
# match the label IDs (e.g. ID 10 is strawberry, not "Peach leaf"). The models'
# built-in names are that wrong list (class_names.txt), so we display the
# verified mapping from class_names_corrected.txt instead.
ORIGINAL_NAMES = ROOT / "config" / "class_names.txt"
CORRECTED_NAMES = ROOT / "config" / "class_names_corrected.txt"


def load_class_names(path=CORRECTED_NAMES):
    names = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" in line:
            idx, name = line.split(":", 1)
            names[int(idx)] = name.strip()
    return names


_cache = {}


def get_model(model_key=DEFAULT_MODEL):
    """Load a model once, check it is the expected model and apply the corrected names."""
    if model_key not in _cache:
        model = YOLO(str(MODELS[model_key]))
        expected = load_class_names(ORIGINAL_NAMES)
        if dict(model.names) != expected:
            diff = {k: (model.names.get(k), expected.get(k))
                    for k in set(model.names) | set(expected)
                    if model.names.get(k) != expected.get(k)}
            raise ValueError(f"Class mapping mismatch for {model_key}: {diff}")
        model.model.names = load_class_names()  # also used by result.plot()
        _cache[model_key] = model
    return _cache[model_key]


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in original image pixels

    @property
    def healthy(self):
        return self.label in HEALTHY_CLASSES


def detect(image, model_key=DEFAULT_MODEL, conf=DEFAULT_CONF):
    """Run detection on a file path or PIL image (numpy arrays must be BGR).

    Returns (annotated RGB numpy image, list[Detection], inference time in ms).
    """
    model = get_model(model_key)
    start = time.perf_counter()
    result = model.predict(image, imgsz=IMG_SIZE, conf=conf, iou=IOU, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = [
        Detection(
            class_id=int(b.cls),
            label=model.names[int(b.cls)],
            confidence=float(b.conf),
            box=tuple(round(v) for v in b.xyxy[0].tolist()),
        )
        for b in result.boxes
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    annotated = result.plot()[:, :, ::-1]  # plot() returns BGR
    return annotated, detections, elapsed_ms
