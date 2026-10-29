"""Plant care advice from an LLM on Groq, based on the detector's findings.

Used by the web UI (streamlit_app.py). The vision model gets the leaf photo plus the
detections, so it can explain the disease and suggest treatment and care.
"""
import base64
import io

from groq import Groq
from PIL import Image

# Qwen 3.8 27B is the vision model on Groq (preview), so it can look at the photo.
# Override with GROQ_MODEL in .streamlit/secrets.toml to use another model.
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """You are a friendly plant doctor helping home gardeners and small farmers.
You receive a leaf photo and the findings of a YOLOv8 plant disease detector.
The detector can be wrong: if the photo clearly disagrees with it, say so politely.

Answer the user's questions about this plant: the disease, treatment, watering, light,
soil, pruning, prevention and so on. For treatment, give organic options first and
chemical options only if needed. Suggest a local agricultural extension office or plant
expert when the problem is serious.

Keep answers short (under 200 words), in plain language with Markdown lists where they
help, and stay on plants and gardening."""


def summarize(detections):
    """Describe the detections as plain text for the prompt."""
    if not detections:
        return "The detector found no leaf in this photo."
    lines = [f"- {d.label} ({'healthy' if d.healthy else 'diseased'}, "
             f"{d.confidence:.0%} confidence)" for d in detections[:5]]
    return "The detector found:\n" + "\n".join(lines)


def make_client(api_key):
    return Groq(api_key=api_key)


def stream_advice(client, model, image_bytes, detections, history):
    """Yield the model's reply text chunk by chunk.

    history is the chat so far, starting with the user's first question:
    [{"role": "user"|"assistant", "content": str}]. The photo and detections are
    attached to that first question.
    """
    mime = Image.MIME[Image.open(io.BytesIO(image_bytes)).format]
    image_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": summarize(detections)
                + "\n\nMy question: " + history[0]["content"]},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]},
        *history[1:],
    ]
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True,
        reasoning_format="hidden")  # keep Qwen's thinking out of the answer
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
