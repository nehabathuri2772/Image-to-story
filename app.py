import os, re, io, tempfile
from datetime import datetime

import gradio as gr
from PIL import Image
from huggingface_hub import InferenceClient
import requests

# ====== Config (Space → Settings → Variables & secrets) ======
HF_TOKEN = os.getenv("HF_TOKEN")  # strongly recommended
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "nlpconnect/vit-gpt2-image-captioning")
LLM_MODEL     = os.getenv("LLM_MODEL",     "Qwen/Qwen2.5-3B-Instruct")

# Warn if running without token (lower rate limits)
if not HF_TOKEN:
    gr.Warning("Running without HF_TOKEN → public rate limits; you may see 429/401 errors.")

# ====== Inference API clients ======
cap_client = InferenceClient(CAPTION_MODEL, token=HF_TOKEN, timeout=120)
llm_client = InferenceClient(LLM_MODEL, token=HF_TOKEN, timeout=120)

# ----------------- Helpers -----------------
def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def smart_trim_to_max_words(text: str, max_words: int) -> str:
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= max_words:
        return text.strip()
    clipped = " ".join(tokens[:max_words]).strip()
    last_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    if last_end >= int(max_words * 0.6):
        clipped = clipped[: last_end + 1]
    return clipped.strip()

def parse_title_and_story(text: str):
    t = text.replace("\r\n", "\n").strip()
    m = re.match(r'^\s*Title:\s*(.+?)\n\n(.*)$', t, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", t

# ----------------- API wrappers w/ fallback -----------------
def _hf_headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def caption_api(img: Image.Image) -> str:
    """Try client.image_to_text; if it fails, fallback to raw REST."""
    try:
        return cap_client.image_to_text(img).strip()
    except Exception as e1:
        # Fallback: REST POST with image bytes
        try:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            r = requests.post(
                f"https://api-inference.huggingface.co/models/{CAPTION_MODEL}",
                headers=_hf_headers(),
                data=buf.getvalue(),
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            # Common HF response formats:
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            # Some models return {"generated_text": "..."} directly
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            # Unknown shape → return stringified payload
            return str(data)
        except Exception as e2:
            raise RuntimeError(f"caption_api failed: {e1} | fallback: {e2}")

def build_user_prompt(image_desc: str, audience: str, story_type: str,
                      min_words: int, max_words: int, gen_title: bool) -> str:
    if gen_title:
        title_part = (
            "Start with one line exactly like: Title: <concise, original title>\n"
            "Then a blank line, then the story.\n"
        )
    else:
        title_part = "Not generating the title. Start directly with the story.\n"

    return (
        f"Write a **{story_type.lower()}** short story for a **{audience.lower()}** audience, "
        f"based ONLY on this image description:\n"
        f"{image_desc}\n\n"
        "Constraints:\n"
        f"- Length: between {min_words} and {max_words} words.\n"
        f"- Tone/genre: {story_type}.\n"
        "- No meta commentary or warnings.\n"
        "- Keep it self-contained, vivid, and consistent with the image description.\n\n"
        "Formatting:\n"
        f"{title_part}"
        "If you include a title, put nothing else on the title line except the title itself.\n"
    ).strip()

def generate_story_api(prompt_text: str, temperature: float, top_p: float, max_words: int) -> str:
    approx_tokens = int(max_words * 1.7)
    max_new_tokens = max(180, min(1600, approx_tokens + 80))
    try:
        return llm_client.text_generation(
            prompt_text,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        ).strip()
    except Exception as e1:
        # Fallback via REST
        try:
            payload = {
                "inputs": prompt_text,
                "parameters": {
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
                "options": {"wait_for_model": True},  # queue if model is cold
            }
            r = requests.post(
                f"https://api-inference.huggingface.co/models/{LLM_MODEL}",
                headers={**_hf_headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            r.raise_for_status()
            data = r.json()
            # Common format for text-generation: list of dicts with "generated_text"
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            return str(data)
        except Exception as e2:
            raise RuntimeError(f"generate_story_api failed: {e1} | fallback: {e2}")

# ----------------- Inference wiring -----------------
def infer(image_input, audience, story_type, min_words, max_words, gen_title, temperature, top_p):
    try:
        min_words = int(min_words); max_words = int(max_words)
        if min_words < 200: min_words = 200
        if max_words > 1000: max_words = 1000
        if min_words > max_words:
            min_words, max_words = max_words, min_words
            gr.Warning("Swapped min/max to keep a valid range (200–1000).")

        if image_input is None:
            gr.Warning("Please upload an image.")
            return "", ""

        gr.Info("Captioning the image (remote API)…")
        image_desc = caption_api(image_input)

        user_prompt = build_user_prompt(
            image_desc, audience, story_type, min_words, max_words, gen_title
        )

        gr.Info("Generating your story (remote API)…")
        raw = generate_story_api(user_prompt, temperature, top_p, max_words)

        title, story = parse_title_and_story(raw)
        story = "\n\n".join(p for p in story.split("\n") if p.strip())

        wc = word_count(story)
        if wc > max_words:
            story = smart_trim_to_max_words(story, max_words)
            wc = word_count(story)
            gr.Info(f"Trimmed to {wc} words to respect the {max_words}-word limit.")
        elif wc < min_words:
            gr.Warning(f"Generated {wc} words (target {min_words}-{max_words}). "
                       "Try lowering min or raising temperature.")
        return title, story

    except Exception as e:
        # Show the real error in the UI so you can diagnose quickly
        err = f"Error: {type(e).__name__}: {e}"
        gr.Warning(err)
        return "Error", err

def download_story(title: str, story: str):
    if not story.strip():
        gr.Warning("Nothing to download yet — generate a story first.")
        return None
    safe_title = re.sub(r"[^\w\-\s]", "", title or "Image_to_Story")[:60].strip() or "Image_to_Story"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{ts}.txt"
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        if title.strip():
            f.write(title.strip() + "\n\n")
        f.write(story.strip() + "\n")
    return path

# ---------------- UI ----------------
CSS = (
  "body{background:#f6f7fb}"
  "#col{max-width:1200px;margin:0 auto;padding:12px}"
  ".chip{display:inline-block;padding:8px 12px;border-radius:12px;"
  "background:#ede9fe;color:#4c1d95;font-weight:600;font-size:14px}"
  ".card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;"
  "padding:16px;box-shadow:0 2px 10px rgba(0,0,0,.03)}"
  ".fullw button{width:100%}"
  "#story textarea{font-size:1.06em;line-height:1.65em}"
  ".muted{color:#6b7280;font-size:14px;margin-top:4px}"
  ".tips{background:#f9fafb;border:1px dashed #e5e7eb;border-radius:12px;padding:12px}"
)

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(css=CSS, theme=THEME, title="Image to Story (API)") as demo:
    with gr.Column(elem_id="col"):
        gr.Markdown(
            "<div style='text-align:center'>"
            "<h1>Image to Story (API)</h1>"
            "<p class='muted' style='margin-top:-4px'>Upload an image and let the app call remote models to craft a story.</p>"
            "</div>"
        )

        with gr.Row():
            # Left: inputs
            with gr.Column(scale=1):
                gr.Markdown("<span class='chip'>Upload Image</span>")
                with gr.Column(elem_classes=["card"]):
                    # IMPORTANT: use type='pil' so we pass a Pillow image to API wrapper
                    image_in = gr.Image(label=None, type="pil", height=320)
                    gr.Markdown("<div class='muted'>Supported: JPG/PNG.</div>")

                with gr.Row():
                    with gr.Column(elem_classes=["card"]):
                        story_type = gr.Dropdown(
                            label="Genre",
                            choices=["Adventure", "Comedy", "Drama", "Fantasy", "Romance", "Horror"],
                            value="Adventure",
                        )
                    with gr.Column(elem_classes=["card"]):
                        audience = gr.Dropdown(
                            label="Target Audience",
                            choices=["Children", "Adult"],
                            value="Children",
                        )

                with gr.Row():
                    with gr.Column(elem_classes=["card"]):
                        min_words = gr.Number(value=400, label="Minimum Words", precision=0)
                        gr.Markdown("<div class='muted'>200</div>")
                    with gr.Column(elem_classes=["card"]):
                        max_words = gr.Number(value=800, label="Maximum Words", precision=0)
                        gr.Markdown("<div class='muted'>1000</div>")

                with gr.Column(elem_classes=["card"]):
                    gen_title = gr.Checkbox(value=True, label="Generate Title")
                    temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Imagination Level")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Vocabulary breadth")

                with gr.Row():
                    submit_btn = gr.Button("Generate Story", variant="primary", elem_classes=["fullw"])

            # Right: outputs
            with gr.Column(scale=1):
                gr.Markdown("<span class='chip'>Generated Story</span>")
                with gr.Column(elem_classes=["card"]):
                    title_out = gr.Textbox(
                        label="Title",
                        interactive=False,
                        placeholder="(Title appears here)",
                        show_copy_button=True,
                    )
                    story_out = gr.Textbox(
                        label="Story",
                        elem_id="story",
                        lines=22,
                        show_copy_button=True,
                    )
                with gr.Column(elem_classes=["card"]):
                    download_btn = gr.DownloadButton("Download story")

        submit_btn.click(
            fn=infer,
            inputs=[image_in, audience, story_type, min_words, max_words, gen_title, temperature, top_p],
            outputs=[title_out, story_out],
        )
        download_btn.click(fn=download_story, inputs=[title_out, story_out], outputs=download_btn)

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=True)
