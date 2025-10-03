# app.py — Image → Story (local CPU) with API endpoints for Hugging Face Spaces

import os, re, tempfile
import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

# ---------- CPU sanity ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(2)
except Exception:
    pass

# ---------- Models (small & local) ----------
# 1) Image captioner
CAPTION_ID = "nlpconnect/vit-gpt2-image-captioning"
cap_model = VisionEncoderDecoderModel.from_pretrained(CAPTION_ID)
cap_proc  = ViTImageProcessor.from_pretrained(CAPTION_ID)
cap_tok   = AutoTokenizer.from_pretrained(CAPTION_ID)

# 2) Small instruction LLM (override with env SMALL_LLM_ID if needed)
STORY_ID = (os.getenv("SMALL_LLM_ID") or "Qwen/Qwen2.5-0.5B-Instruct").strip()
story_tok = AutoTokenizer.from_pretrained(STORY_ID, use_fast=True)
story_model = AutoModelForCausalLM.from_pretrained(
    STORY_ID,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# ---------- Helpers ----------
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

def apply_chat_prompt(system_text: str, user_text: str) -> str:
    # Use chat template if available; otherwise simple fallback
    if hasattr(story_tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return story_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"[System]\n{system_text}\n\n[User]\n{user_text}\n\n[Assistant]\n"

# ---------- Captioning (PIL → text) ----------
def _shrink(img: Image.Image, max_side: int = 384) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)

def caption_image(pil_img: Image.Image) -> str:
    img = _shrink(pil_img.convert("RGB"))
    pixel_values = cap_proc(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        out_ids = cap_model.generate(
            pixel_values,
            max_length=24,
            num_beams=3,
            no_repeat_ngram_size=2,
        )
    return cap_tok.decode(out_ids[0], skip_special_tokens=True).strip()

# ---------- Prompt & story ----------
def build_user_prompt(image_desc: str, audience: str, genre: str,
                      min_words: int, max_words: int, gen_title: bool) -> str:
    title_part = (
        "Start with one line exactly like: Title: <concise, original title>\n"
        "Then a blank line, then the story.\n"
        if gen_title else
        "Do NOT include any title line. Start directly with the story.\n"
    )
    return (
        f"Write a **{genre.lower()}** short story for a **{audience.lower()}** audience. "
        f"Base it ONLY on this image description:\n{image_desc}\n\n"
        "Constraints:\n"
        f"- Length: between {min_words} and {max_words} words.\n"
        f"- Tone/genre: {genre}.\n"
        "- No meta commentary or warnings.\n"
        "- Keep it self-contained, vivid, and easy to follow.\n\n"
        "Formatting:\n"
        f"{title_part}"
        "If you include a title, put nothing else on the title line except the title itself.\n"
    ).strip()

def generate_story(user_text: str, temperature: float, top_p: float, max_words: int) -> str:
    prompt = apply_chat_prompt("You are a helpful, creative storyteller.", user_text)
    inputs = story_tok(prompt, return_tensors="pt")
    start = inputs["input_ids"].shape[1]
    approx_tokens = int(max_words * 1.7)
    max_new = max(160, min(1000, approx_tokens + 80))
    with torch.no_grad():
        outputs = story_model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.05,
            max_new_tokens=int(max_new),
            eos_token_id=story_tok.eos_token_id,
            pad_token_id=story_tok.eos_token_id,
        )
    gen_ids = outputs[0][start:]
    return story_tok.decode(gen_ids, skip_special_tokens=True).strip()

# ---------- Pipeline (image → caption → story) ----------
def infer(image_input, audience, genre, min_words, max_words, gen_title, temperature, top_p):
    if image_input is None:
        gr.Warning("Please upload an image.")
        return "", ""

    min_words = int(min_words); max_words = int(max_words)
    if min_words < 150: min_words = 150
    if max_words > 900: max_words = 900
    if min_words > max_words:
        min_words, max_words = max_words, min_words
        gr.Warning("Swapped min/max to keep a valid range (150–900).")

    gr.Info("Captioning the image (ViT-GPT2)…")
    image_desc = caption_image(image_input)

    user_prompt = build_user_prompt(image_desc, audience, genre, min_words, max_words, gen_title)

    gr.Info("Writing your story (small LLM on CPU)…")
    raw = generate_story(user_prompt, temperature, top_p, max_words)

    title, story = parse_title_and_story(raw)
    story = "\n\n".join(p for p in story.split("\n") if p.strip())

    wc = word_count(story)
    if wc > max_words:
        story = smart_trim_to_max_words(story, max_words)
        gr.Info(f"Trimmed to {word_count(story)} words to respect the limit.")
    elif wc < min_words:
        gr.Warning(f"Generated {wc} words (target {min_words}-{max_words}). "
                   "Try lowering min or raising temperature.")
    return title, story

def download_story(title: str, story: str):
    if not story.strip():
        gr.Warning("Nothing to download yet — generate a story first.")
        return None
    safe = re.sub(r"[^\w\-\s]", "", title or "Story")[:60].strip() or "Story"
    path = tempfile.mktemp(prefix=safe + "_", suffix=".txt")
    with open(path, "w", encoding="utf-8") as f:
        if title.strip():
            f.write(title.strip() + "\n\n")
        f.write(story.strip() + "\n")
    return path

# ---------- UI ----------
CSS = (
  "body{background:#f6f7fb}"
  "#col{max-width:900px;margin:0 auto;padding:12px}"
  ".chip{display:inline-block;padding:8px 12px;border-radius:12px;"
  "background:#ede9fe;color:#4c1d95;font-weight:600;font-size:14px}"
  ".card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;"
  "padding:16px;box-shadow:0 2px 10px rgba(0,0,0,.03)}"
  ".fullw button{width:100%}"
  "#story textarea{font-size:1.06em;line-height:1.65em}"
  ".muted{color:#6b7280;font-size:14px;margin-top:4px}"
)

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(css=CSS, theme=THEME, title="Image → Story") as demo:
    with gr.Column(elem_id="col"):
        gr.Markdown(
            "<div style='text-align:center'>"
            "<h1>Lightweight Image → Story</h1>"
            "<p class='muted' style='margin-top:-4px'>Generate your story with the image tailored to your preferences.</p>"
            "</div>"
        )

        with gr.Row():
            with gr.Column(elem_classes=["card"]):
                image_in = gr.Image(label="Upload Image", type="pil", height=320)

        with gr.Row():
            with gr.Column(elem_classes=["card"]):
                genre = gr.Dropdown(
                    label="Genre",
                    choices=["Adventure", "Comedy", "Drama", "Fantasy", "Romance", "Horror", "Mystery", "Sci-Fi"],
                    value="Adventure",
                )
            with gr.Column(elem_classes=["card"]):
                audience = gr.Dropdown(
                    label="Target Audience",
                    choices=["Children", "Teen", "Adult"],
                    value="Children",
                )

        with gr.Row():
            with gr.Column(elem_classes=["card"]):
                min_words = gr.Number(value=300, label="Minimum Words", precision=0)
                gr.Markdown("<div class='muted'>150 – 900</div>")
            with gr.Column(elem_classes=["card"]):
                max_words = gr.Number(value=600, label="Maximum Words", precision=0)
                gr.Markdown("<div class='muted'>150 – 900</div>")

        with gr.Column(elem_classes=["card"]):
            gen_title = gr.Checkbox(value=True, label="Generate Title")
            temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Creativity (temperature)")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")

        with gr.Row():
            submit_btn = gr.Button("Generate Story", variant="primary", elem_classes=["fullw"])

        with gr.Column(elem_classes=["card"]):
            title_out = gr.Textbox(label="Title", interactive=False, show_copy_button=True)
            story_out = gr.Textbox(label="Story", elem_id="story", lines=18, show_copy_button=True)

        with gr.Column(elem_classes=["card"]):
            download_btn = gr.DownloadButton("Download .txt")

        # ---------- API endpoints (this is what makes it API-based) ----------
        submit_btn.click(
            fn=infer,
            inputs=[image_in, audience, genre, min_words, max_words, gen_title, temperature, top_p],
            outputs=[title_out, story_out],
            api_name="generate"  # exposes /api/predict with api_name="/generate"
        )
        download_btn.click(
            fn=download_story,
            inputs=[title_out, story_out],
            outputs=download_btn,
            api_name="download"  # exposes /api/predict with api_name="/download"
        )

if __name__ == "__main__":
    demo.queue().launch()
