# Image -> Story (CPU-only). Plain UI + better grounding.
import os, re, tempfile
from datetime import datetime

import gradio as gr
import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)

# --- make CPU Spaces stabler/faster ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(2)
except Exception:
    pass

# ---------------- Models ----------------
# Older captioner kept for fallback
_CAPTION_VIT_ID = "nlpconnect/vit-gpt2-image-captioning"
cap_model = VisionEncoderDecoderModel.from_pretrained(_CAPTION_VIT_ID)
cap_processor = ViTImageProcessor.from_pretrained(_CAPTION_VIT_ID)
cap_tokenizer = AutoTokenizer.from_pretrained(_CAPTION_VIT_ID)

# Better captioner (BLIP) - CPU OK
BLIP_ID = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(BLIP_ID)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_ID)

# Story LLM (CPU friendly)
STORY_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
story_tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_ID, use_fast=True)
story_model = AutoModelForCausalLM.from_pretrained(
    STORY_MODEL_ID,
    device_map="cpu",
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Lightweight CLIP for grounding keywords (very fast vs LLM)
CLIP_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

# Label pool for CLIP grounding (generic scene/object words)
CLIP_LABELS = [
    "wigs","wig shop","hair","salon","mannequin","mannequin heads","shelf","store",
    "glasses","woman","person","face","indoor","counter","books","classroom",
    "shoes","bags","hats",
]

# --------------- Helpers ---------------
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

def caption_blip(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=32)
    text = blip_processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()

def caption_vit(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    pixel_values = cap_processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = cap_model.generate(
            pixel_values,
            max_length=24,
            num_beams=2,
            no_repeat_ngram_size=2,
        )
    caption = cap_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def auto_caption(image_path: str) -> str:
    try:
        text = caption_blip(image_path)
        # sanity: if BLIP returns an ultra-short or empty caption, fallback
        if len(text.split()) < 2:
            return caption_vit(image_path)
        return text
    except Exception:
        return caption_vit(image_path)

def clip_top_labels(image_path: str, k: int = 4):
    img = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=CLIP_LABELS, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = clip_model(**inputs)
        logits = out.logits_per_image[0]  # (num_labels,)
        vals, idx = torch.topk(logits, k=min(k, len(CLIP_LABELS)))
    return [CLIP_LABELS[i] for i in idx.tolist()]

def build_user_prompt(
    image_desc: str,
    audience: str,
    story_type: str,
    min_words: int,
    max_words: int,
    gen_title: bool,
    force_words=None,
) -> str:
    if gen_title:
        title_part = (
            "Start with one line exactly like: Title: <concise, original title>\n"
            "Then a blank line, then the story.\n"
        )
    else:
        title_part = "Do NOT include any title line. Start directly with the story.\n"

    grounding = ""
    if force_words:
        fw = ", ".join(force_words)
        grounding = (
            "\nIn the FIRST paragraph, explicitly include at least 2 of these words: "
            + fw + ". Do not invent unrelated objects."
        )

    prompt = (
        f"Write a **{story_type.lower()}** short story for a **{audience.lower()}** audience "
        f"based ONLY on this image description:\n"
        f"{image_desc}\n\n"
        "Constraints:\n"
        f"- Length: between {min_words} and {max_words} words.\n"
        f"- Tone/genre: {story_type}.\n"
        "- No meta commentary or warnings.\n"
        "- Keep it self-contained and vivid."
        f"{grounding}\n\n"
        "Formatting:\n"
        f"{title_part}"
        "If you include a title, put nothing else on the title line except the title itself.\n"
    )
    return prompt.strip()

def qwen_chat_prompt(user_text: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful, creative storyteller."},
        {"role": "user", "content": user_text},
    ]
    return story_tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

def generate_story_with_qwen(user_text: str, temperature: float, top_p: float, max_words: int) -> str:
    prompt_text = qwen_chat_prompt(user_text)
    inputs = story_tokenizer(prompt_text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    max_new_tokens = max(120, min(500, int(max_words * 1.2)))  # CPU friendly
    with torch.no_grad():
        outputs = story_model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
            eos_token_id=story_tokenizer.eos_token_id,
            pad_token_id=story_tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][input_len:]
    text = story_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text

# --------------- Core callback ---------------
def infer(image_input, manual_desc, audience, story_type, min_words, max_words, gen_title, temperature, top_p):
    min_words = int(min_words); max_words = int(max_words)
    if min_words < 200: min_words = 200
    if max_words > 1000: max_words = 1000
    if min_words > max_words:
        min_words, max_words = max_words, min_words
        gr.Warning("Swapped min/max to keep a valid range (200–1000).")
    if image_input is None:
        gr.Warning("Please upload an image.")
        return "", ""

    if manual_desc and manual_desc.strip():
        image_desc = manual_desc.strip()
    else:
        gr.Info("Making a caption (BLIP)…")
        image_desc = auto_caption(image_input)

    # CLIP grounding words (fast) to avoid random objects
    try:
        force_words = clip_top_labels(image_input, k=4)
    except Exception:
        force_words = None

    user_prompt = build_user_prompt(
        image_desc, audience, story_type, min_words, max_words, gen_title, force_words
    )

    gr.Info("Writing your story with Qwen 2.5 (CPU)…")
    raw = generate_story_with_qwen(user_prompt, temperature, top_p, max_words)

    title, story = parse_title_and_story(raw)
    story = "\n\n".join(p for p in story.split("\n") if p.strip())

    wc = word_count(story)
    if wc > max_words:
        story = smart_trim_to_max_words(story, max_words)
        wc = word_count(story)
        gr.Info(f"Trimmed to {wc} words to respect the {max_words}-word limit.")
    elif wc < min_words:
        gr.Warning(f"Generated {wc} words (target {min_words}-{max_words}). Try lowering min or raising temperature.")
    return title, story

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

def reset_all():
    return None, "", "Children", "Adventure", 400, 700, True, 0.9, 0.95, "", ""

# ---------------- UI (plain & neutral) ----------------
CSS = (
  "/* Plain, neutral look */"
  ":root{--card:transparent;}"
  "body{background:#ffffff;}"
  "#col-container{max-width:1100px;margin-left:auto;margin-right:auto;padding:8px}"
  ".glass{background:transparent;backdrop-filter:none;-webkit-backdrop-filter:none;"
  "border:1px solid #e5e7eb;box-shadow:none;border-radius:16px;padding:16px}"
  "#story textarea{font-size:1.04em;line-height:1.6em;color:#111}"
  ".gradio-container .prose h1{color:#111;background:none}"
  "button{border-radius:9999px !important}"
  "[data-testid=\\\"block-label\\\"],[data-testid=\\\"block-label\\\"] *{"
  "background:transparent !important;color:#111 !important;box-shadow:none !important;border:none !important}"
  ".badge,.tag,.token,.label,.label-wrap{background:transparent !important;color:#111 !important;"
  "box-shadow:none !important;border:none !important}"
)

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(css=CSS, theme=THEME, title="Image -> Story • Qwen 2.5 (CPU)") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            "<div style='text-align:center'>"
            "<h1>Image → Story</h1>"
            "<p style='opacity:.9'>Upload an image, pick a genre, set word limits, and (optionally) generate a title.</p>"
            "<div style='font-size:14px;opacity:.8'>"
            "Captioner: BLIP base (fallback: ViT-GPT2) · "
            "Story LLM: <code>Qwen/Qwen2.5-1.5B-Instruct</code>"
            "</div></div>"
        )

        with gr.Row():
            with gr.Column(elem_classes=["glass"]):
                image_in = gr.Image(label="Drop image here", type="filepath", height=320)
                manual_desc = gr.Textbox(label="(Optional) Describe the image to override the caption", placeholder="e.g., A woman in a wig shop with mannequin heads on shelves")
                audience = gr.Radio(label="Target Audience", choices=["Children", "Adult"], value="Children")
                story_type = gr.Dropdown(
                    label="Story Type",
                    choices=["Adventure", "Comedy", "Drama", "Fantasy", "Romance"],
                    value="Adventure",
                )
                with gr.Row():
                    min_words = gr.Slider(200, 1000, value=400, step=50, label="Min words (200–1000)")
                    max_words = gr.Slider(200, 1000, value=700, step=50, label="Max words (200–1000)")
                gen_title = gr.Checkbox(value=True, label="Generate Title")
                with gr.Row():
                    temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Creativity (temperature)")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                with gr.Row():
                    submit_btn = gr.Button("✨ Tell me a story", variant="primary")
                    reset_btn  = gr.Button("↺ Reset")

            with gr.Column(elem_classes=["glass"]):
                title_out = gr.Textbox(label="Title", interactive=False, placeholder="(Title appears here)", show_copy_button=True)
                story_out = gr.Textbox(label="Story", elem_id="story", lines=20, show_copy_button=True)
                download_btn = gr.DownloadButton("📥 Download .txt")

        submit_btn.click(
            fn=infer,
            inputs=[image_in, manual_desc, audience, story_type, min_words, max_words, gen_title, temperature, top_p],
            outputs=[title_out, story_out],
        )
        download_btn.click(fn=download_story, inputs=[title_out, story_out], outputs=download_btn)
        reset_btn.click(
            fn=reset_all,
            inputs=None,
            outputs=[image_in, manual_desc, audience, story_type, min_words, max_words, gen_title, temperature, top_p, title_out, story_out],
        )

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
