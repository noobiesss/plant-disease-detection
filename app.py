"""Plant Disease Detection prototype — Gradio web UI.

Run:  python app.py   then open http://127.0.0.1:7860
"""
import html
import time

import gradio as gr

from detector import DEFAULT_CONF, DEFAULT_MODEL, MODELS, ROOT, detect, get_model

# demo_images/: test-set images both models classify correctly (good for the live demo)
DEMO = sorted(str(p) for p in (ROOT / "demo_images").glob("*.jpg"))
SAMPLES = sorted(str(p) for p in (ROOT / "sample_images").glob("*.jpg"))

THEME = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#hero {
    background: linear-gradient(135deg, #065f46 0%, #059669 55%, #34d399 100%);
    border-radius: 20px; padding: 28px 32px; color: white; margin-bottom: 8px;
}
#hero h1 { margin: 0 0 6px; font-size: 30px; font-weight: 800; color: white; }
#hero p { margin: 0; opacity: .9; font-size: 15px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.chip {
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
    padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600;
}
.card {
    border: 1px solid var(--border-color-primary); border-radius: 16px;
    padding: 18px 20px; background: var(--background-fill-primary);
}
.status { display: flex; align-items: center; gap: 14px; }
.status .icon {
    width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center;
    font-size: 22px; flex-shrink: 0;
}
.status.ok .icon { background: #d1fae5; }
.status.bad .icon { background: #fee2e2; }
.status.none .icon { background: #e2e8f0; }
.status .label { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; opacity: .7; }
.status .title { font-size: 20px; font-weight: 700; }
.status.ok .title { color: #059669; }
.status.bad .title { color: #dc2626; }
.bars { margin-top: 16px; display: flex; flex-direction: column; gap: 10px; }
.bar-row { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; font-size: 14px; }
.bar-row .tag { font-size: 11px; font-weight: 700; padding: 1px 8px; border-radius: 999px; margin-left: 6px; }
.tag.ok { background: #d1fae5; color: #065f46; }
.tag.bad { background: #fee2e2; color: #991b1b; }
.track { grid-column: 1 / -1; height: 8px; border-radius: 999px; background: var(--neutral-200); overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.fill.ok { background: linear-gradient(90deg, #34d399, #059669); }
.fill.bad { background: linear-gradient(90deg, #f87171, #dc2626); }
.meta { margin-top: 14px; font-size: 12px; opacity: .65; }
.placeholder { text-align: center; padding: 26px 10px; opacity: .6; }
#detect-btn { font-size: 16px; font-weight: 700; min-height: 48px; }
#disclaimer { text-align: center; font-size: 12px; opacity: .6; margin-top: 8px; }
"""

HERO = f"""
<div id="hero">
  <h1>🌿 Plant Disease Detection</h1>
  <p>Upload or snap a leaf photo. The model finds each leaf, names the disease and shows how confident it is.</p>
  <div class="chips">
    <span class="chip">YOLOv8</span><span class="chip">28 classes</span>
    <span class="chip">13 plant species</span><span class="chip">Runs on CPU</span>
  </div>
</div>
"""

PLACEHOLDER = ('<div class="card placeholder">Upload an image and press '
               '<b>Detect</b> to see the diagnosis here.</div>')


def result_card(detections, model_key, conf, infer_ms, total_ms):
    if not detections:
        icon, cls, label, title = "🔍", "none", "No result", "No leaf detected"
        hint = ("<div class='meta'>Try a clearer, closer photo of a single leaf, "
                "or lower the confidence threshold.</div>")
        bars = ""
    else:
        diseased = [d for d in detections if not d.healthy]
        if diseased:
            icon, cls, label, title = "⚠️", "bad", "Disease detected", diseased[0].label
        else:
            icon, cls, label, title = "✅", "ok", "Healthy leaf", detections[0].label
        hint = ""
        rows = []
        for d in detections[:5]:
            k = "ok" if d.healthy else "bad"
            rows.append(
                f"<div class='bar-row'><span>{html.escape(d.label)}"
                f"<span class='tag {k}'>{'Healthy' if d.healthy else 'Diseased'}</span></span>"
                f"<b>{d.confidence:.0%}</b>"
                f"<div class='track'><div class='fill {k}' style='width:{d.confidence:.0%}'></div></div></div>")
        more = (f"<div class='meta'>+ {len(detections) - 5} more in the table below</div>"
                if len(detections) > 5 else "")
        bars = f"<div class='bars'>{''.join(rows)}</div>{more}"
    meta = (f"<div class='meta'>{html.escape(model_key)} · threshold {conf:.2f} · "
            f"{len(detections)} object(s) · inference {infer_ms:.0f} ms · total {total_ms:.0f} ms</div>")
    return (f"<div class='card'><div class='status {cls}'><div class='icon'>{icon}</div>"
            f"<div><div class='label'>{label}</div><div class='title'>{html.escape(title)}</div></div></div>"
            f"{bars}{hint}{meta}</div>")


def run(image, model_key, conf):
    if image is None:
        raise gr.Error("Please upload an image first.")
    start = time.perf_counter()
    annotated, detections, infer_ms = detect(image, model_key, conf)
    total_ms = (time.perf_counter() - start) * 1000
    rows = [[i + 1, d.label, "Healthy" if d.healthy else "Diseased",
             f"{d.confidence:.3f}", str(list(d.box))]
            for i, d in enumerate(detections)]
    return annotated, result_card(detections, model_key, conf, infer_ms, total_ms), rows


def reset():
    return None, None, PLACEHOLDER, None


with gr.Blocks(title="Plant Disease Detection") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            # PIL keeps RGB order; a numpy array would be read as BGR by Ultralytics
            inp = gr.Image(type="pil", label="Leaf photo", height=380,
                           sources=["upload", "webcam", "clipboard"])
            with gr.Accordion("⚙️ Settings", open=False):
                model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
                conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05,
                                 label="Confidence threshold")
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                btn = gr.Button("🔍 Detect", variant="primary", scale=2, elem_id="detect-btn")
            # Example images stay hidden until the presenter opens this section
            with gr.Accordion("🖼️ Example images", open=False):
                gr.Examples(DEMO, inputs=inp, label=None, examples_per_page=12)
        with gr.Column(scale=6):
            out_img = gr.Image(label="Detection result", type="numpy", height=380,
                               interactive=False)
            summary = gr.HTML(PLACEHOLDER)
            with gr.Accordion("📋 All detections (class, confidence, box)", open=False):
                table = gr.Dataframe(
                    headers=["#", "Class", "Status", "Confidence", "Box [x1, y1, x2, y2]"],
                    interactive=False)
    gr.HTML("<div id='disclaimer'>Prototype for INT4203 · Results are model predictions, "
            "not an expert diagnosis.</div>")

    btn.click(run, [inp, model_dd, conf], [out_img, summary, table])
    clear_btn.click(reset, None, [inp, out_img, summary, table])

if __name__ == "__main__":
    for key in MODELS:  # load, validate class mapping and warm up before serving
        get_model(key)
        detect(SAMPLES[0], key)
    demo.launch(theme=THEME, css=CSS, footer_links=[])
