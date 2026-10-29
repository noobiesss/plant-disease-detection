"""Plant care advice from Google Gemini, based on the detector's findings.

Used by the web UI (streamlit_app.py). Gemini gets the leaf photo plus the
detections, so it can explain the disease and suggest treatment and care.
"""
import io

from google import genai
from google.genai import types
from PIL import Image

# An alias that always points at Google's current Flash model.
# Override with GEMINI_MODEL in .streamlit/secrets.toml to pin a version.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

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
    return genai.Client(api_key=api_key)


def stream_advice(client, model, image_bytes, detections, history):
    """Yield Gemini's reply text chunk by chunk.

    history is the visible chat so far: [{"role": "user"|"assistant", "content": str}].
    An empty history asks for the first care advice.
    """
    mime = Image.MIME[Image.open(io.BytesIO(image_bytes)).format]
    first_turn = types.Content(role="user", parts=[
        types.Part.from_bytes(data=image_bytes, mime_type=mime),
        types.Part.from_text(text=summarize(detections)
                             + "\n\nPlease give care advice for this plant."),
    ])
    contents = [first_turn] + [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part.from_text(text=m["content"])])
        for m in history
    ]
    stream = client.models.generate_content_stream(
        model=model, contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT))
    for chunk in stream:
        if chunk.text:
            yield chunk.text
