"""Plant Disease Detection prototype — Streamlit web UI.

Run locally:  streamlit run streamlit_app.py
Also the entry point on Streamlit Cloud. Detection itself lives in detector.py.
"""
import hashlib
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import groq
from PIL import Image

from advisor import DEFAULT_GROQ_MODEL, make_client, stream_advice, summarize
from detector import DEFAULT_CONF, DEFAULT_MODEL, MODELS, ROOT, detect, get_model

DEMO = sorted((ROOT / "demo_images").glob("*.jpg"))
SAMPLES = sorted((ROOT / "sample_images").glob("*.jpg"))

st.set_page_config(page_title="Plant disease detection", page_icon=":material/eco:",
                   layout="wide")


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


def get_setting(name, default=None):
    """Read a value from .streamlit/secrets.toml (or Cloud secrets), then the environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:  # no secrets file at all
        pass
    return os.environ.get(name, default)


AVATARS = {"user": ":material/person:", "assistant": ":material/potted_plant:"}


@st.cache_resource
def llm_client(api_key):
    return make_client(api_key)


def care_advice(image_bytes, detections):
    """Chat box where the user asks the AI about the current photo and findings."""
    st.subheader("Ask about this plant", icon=":material/forum:")
    api_key = get_setting("GROQ_API_KEY")
    if not api_key:
        st.caption(":material/key: Add `GROQ_API_KEY` to `.streamlit/secrets.toml` (or the "
                   "app's Secrets on Streamlit Cloud) to chat with the AI about this plant.")
        return
    client = llm_client(api_key)
    model = get_setting("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    # Start a fresh conversation whenever the photo or findings change
    key = hashlib.sha1(image_bytes + summarize(detections).encode()).hexdigest()
    if st.session_state.get("advice_key") != key:
        st.session_state.advice_key = key
        st.session_state.advice = []
    history = st.session_state.advice

    # Messages go in a container above the input, so new ones appear above it too
    chat = st.container()
    question = st.chat_input("e.g. How do I treat this? How often should I water it?",
                             submit_mode="disable")
    with chat:
        for msg in history:
            with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
                st.markdown(msg["content"])
        if not question:
            if not history:
                st.caption("The AI sees your photo and the diagnosis. Ask about treatment, "
                           "watering, light, or whether it can spread.")
            return
        history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            try:
                reply = st.write_stream(stream_advice(client, model, image_bytes,
                                                      detections, history))
            except groq.APIError as e:
                history.pop()  # let the user retry the same question
                st.error(f"AI request failed: {e.message or e}", icon=":material/error:")
                return
        history.append({"role": "assistant", "content": reply})
        st.caption("AI-generated answer. Check with a local plant expert before treating.")


with st.sidebar:
    st.subheader("Settings", icon=":material/tune:")
    model_key = st.segmented_control(
        "Detection model", list(MODELS), default=DEFAULT_MODEL, required=True,
        format_func=lambda k: k.split()[0],
        help="YOLOv8s is more accurate; YOLOv8n is faster.")
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
                     help="Hide detections the model is less sure about than this.")
    st.space("medium")
    st.subheader("About", icon=":material/info:")
    st.markdown(":green-badge[YOLOv8] :green-badge[28 classes] "
                ":green-badge[13 species] :gray-badge[CPU]")
    st.caption("Prototype for INT4203. Results are model predictions, "
               "not an expert diagnosis.")

st.title("Plant disease detection", icon=":material/eco:")
st.caption("Photograph a leaf. The model finds it, names the disease, "
           "and you can ask the AI how to care for the plant.")
st.space("small")

left, right = st.columns([5, 6], gap="large")

with left:
    st.subheader("Your photo", icon=":material/add_a_photo:")
    with st.container(border=True):
        source = st.segmented_control(
            "Image source",
            [":material/upload: Upload", ":material/photo_camera: Camera",
             ":material/image: Example"],
            default=":material/upload: Upload", required=True,
            label_visibility="collapsed")

        image_bytes = None
        if source.endswith("Upload"):
            file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
            if file:
                image_bytes = file.getvalue()
        elif source.endswith("Camera"):
            shot = st.camera_input("Take a photo of a leaf", label_visibility="collapsed")
            if shot:
                image_bytes = shot.getvalue()
        else:
            choice = st.selectbox("Example image", DEMO, format_func=demo_label,
                                  label_visibility="collapsed")
            image_bytes = Path(choice).read_bytes()

        detections = []
        if image_bytes:
            with st.spinner("Examining the leaf..."):
                annotated, detections, infer_ms = run_detection(image_bytes, model_key, conf)
            if st.toggle("Show original photo"):
                st.image(image_bytes, alt="Your leaf photo")
            else:
                st.image(annotated, alt="Leaf photo with detected leaves outlined and labelled")
            st.caption(f"{model_key.split()[0]} · threshold {conf:.2f} · "
                       f"{infer_ms:.0f} ms")

with right:
    st.subheader("Diagnosis", icon=":material/stethoscope:")
    if not image_bytes:
        with st.container(border=True):
            st.markdown("**How it works**")
            st.markdown("1. Upload a leaf photo, take one, or pick an example.\n"
                        "2. The model outlines each leaf and names what it sees.\n"
                        "3. Ask the AI how to treat or care for the plant.")
            st.caption("Works best with one leaf, filling most of the frame, in daylight.")
    elif not detections:
        st.warning("Try a clearer, closer photo of a single leaf, or lower the "
                   "confidence threshold in the sidebar.", title="No leaf detected",
                   icon=":material/search_off:")
    else:
        diseased = [d for d in detections if not d.healthy]
        top = diseased[0] if diseased else detections[0]
        with st.container(border=True):
            if diseased:
                st.badge("Disease detected", icon=":material/coronavirus:", color="red")
            else:
                st.badge("Healthy", icon=":material/check_circle:", color="green")
            st.header(top.label, anchor=False)
            with st.container(horizontal=True):
                st.metric("Confidence", f"{top.confidence:.0%}")
                st.metric("Leaves found", len(detections))
                st.metric("Diseased", len(diseased))

        with st.expander(f"All detections ({len(detections)})", icon=":material/table:"):
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

        st.space("small")
        with st.container(border=True):
            care_advice(image_bytes, detections)
