"""Plant Disease Detection prototype — Streamlit web UI.

Run locally:  streamlit run streamlit_app.py
Also the entry point on Streamlit Cloud. Detection itself lives in detector.py.
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from detector import DEFAULT_CONF, DEFAULT_MODEL, MODELS, ROOT, detect, get_model

DEMO = sorted((ROOT / "demo_images").glob("*.jpg"))
SAMPLES = sorted((ROOT / "sample_images").glob("*.jpg"))

st.set_page_config(page_title="Plant disease detection", page_icon="🌿", layout="wide")


@st.cache_resource
def load_model(model_key):
    model = get_model(model_key)
    detect(str(SAMPLES[0]), model_key)  # warm-up, so the first real prediction is fast
    return model


@st.cache_data(max_entries=32, show_spinner=False)
def run_detection(image_bytes, model_key, conf):
    load_model(model_key)
    # PIL keeps RGB order; a numpy array would be read as BGR by Ultralytics
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return detect(image, model_key, conf)


def demo_label(path):
    return Path(path).name.split("_jpg")[0].replace("-", " ")[:40]


st.title("🌿 Plant disease detection")
st.caption("Upload or take a leaf photo. The model finds each leaf, names the disease "
           "and shows how confident it is.")
with st.container(horizontal=True):
    st.badge("YOLOv8", icon=":material/memory:", color="green")
    st.badge("28 classes", icon=":material/category:", color="green")
    st.badge("13 plant species", icon=":material/eco:", color="green")
    st.badge("Runs on CPU", icon=":material/speed:", color="gray")

left, right = st.columns([5, 6], gap="large")

with left:
    with st.container(border=True):
        source = st.segmented_control(
            "Image source",
            [":material/upload: Upload", ":material/photo_camera: Camera", ":material/image: Example"],
            default=":material/upload: Upload", required=True)

        image_bytes = None
        if source.endswith("Upload"):
            file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])
            if file:
                image_bytes = file.getvalue()
        elif source.endswith("Camera"):
            shot = st.camera_input("Take a photo of a leaf")
            if shot:
                image_bytes = shot.getvalue()
        else:
            choice = st.selectbox("Example image", DEMO, format_func=demo_label)
            image_bytes = Path(choice).read_bytes()

        if image_bytes:
            st.image(image_bytes, caption="Input image", alt="Uploaded leaf photo")

    with st.expander("Settings", icon=":material/tune:"):
        model_key = st.segmented_control("Model", list(MODELS), default=DEFAULT_MODEL,
                                         required=True)
        conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)

with right:
    if not image_bytes:
        st.info("Upload a photo, take one with the camera, or pick an example "
                "to see the diagnosis here.", icon=":material/arrow_back:")
    else:
        with st.spinner("Detecting..."):
            annotated, detections, infer_ms = run_detection(image_bytes, model_key, conf)

        st.image(annotated, caption="Detection result",
                 alt="Leaf photo with detected leaves outlined and labelled")

        if not detections:
            st.warning("Try a clearer, closer photo of a single leaf, or lower the "
                       "confidence threshold in Settings.", title="No leaf detected")
        else:
            diseased = [d for d in detections if not d.healthy]
            if diseased:
                st.error(f"Top finding: **{diseased[0].label}** "
                         f"({diseased[0].confidence:.0%} confidence)",
                         title="Disease detected", icon=":material/warning:")
            else:
                st.success(f"**{detections[0].label}** "
                           f"({detections[0].confidence:.0%} confidence)",
                           title="Healthy leaf", icon=":material/check_circle:")

            with st.container(border=True):
                for d in detections[:5]:
                    status = "Healthy" if d.healthy else "Diseased"
                    st.progress(d.confidence,
                                text=f"{d.label} · {status} · {d.confidence:.0%}")
                if len(detections) > 5:
                    st.caption(f"+ {len(detections) - 5} more in the table below")

            with st.expander("All detections (class, confidence, box)",
                             icon=":material/table:"):
                st.dataframe(
                    pd.DataFrame([{
                        "Class": d.label,
                        "Status": "Healthy" if d.healthy else "Diseased",
                        "Confidence": d.confidence,
                        "Box [x1, y1, x2, y2]": str(list(d.box)),
                    } for d in detections]),
                    hide_index=True,
                    column_config={"Confidence": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="percent")},
                    alt="Table of all detections with class, status, confidence and box")

        st.caption(f"{model_key} · threshold {conf:.2f} · {len(detections)} object(s) · "
                   f"inference {infer_ms:.0f} ms")

st.caption("Prototype for INT4203 · Results are model predictions, not an expert diagnosis.")
