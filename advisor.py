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

For the first answer, use these short sections in Markdown:
### What it is
One or two sentences on the disease (or confirm the leaf looks healthy).
### What to do now
3-5 concrete treatment steps, organic options first, chemical options only if needed.
### Ongoing care
3-5 tips on watering, light, soil, pruning and prevention for this plant.
### When to get help
When to contact a local agricultural extension office or plant expert.

Keep it under 250 words, plain language, no jargon without a short explanation.
For follow-up questions, answer briefly and stay on plants and gardening."""


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

    history is the visible chat so far: [{"role": "user"|"assistant", "content": str}].
    An empty history asks for the first care advice.
    """
    mime = Image.MIME[Image.open(io.BytesIO(image_bytes)).format]
    image_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": summarize(detections)
                + "\n\nPlease give care advice for this plant."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]},
        *history,
    ]
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True,
        reasoning_format="hidden")  # keep Qwen's thinking out of the answer
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
