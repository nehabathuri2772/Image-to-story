
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

# -------- CPU stability / speed hints --------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(2)
except Exception:
    pass

# ---------------- Models ----------------
# Fallback captioner 
_VIT_CAP_ID = "nlpconnect/vit-gpt2-image-captioning"
vit_cap_model = VisionEncoderDecoderModel.from_pretrained(_VIT_CAP_ID)
vit_cap_proc = ViTImageProcessor.from_pretrained(_VIT_CAP_ID)
vit_cap_tok = AutoTokenizer.from_pretrained(_VIT_CAP_ID)

# Primary captioner (better on CPU)
BLIP_ID = "Salesforce/blip-image-captioning-base"
blip_proc = BlipProcessor.from_pretrained(BLIP_ID)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_ID)

# Story LLM 
STORY_ID = "Qwen/Qwen2.5-1.5B-Instruct"
story_tok = AutoTokenizer.from_pretrained(STORY_ID, use_fast=True)
story_model = AutoModelForCausalLM.from_pretrained(
    STORY_ID,
    device_map="cpu",
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Lightweight CLIP for quick grounding keywords
CLIP_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_ID)
clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)

# Label pool for CLIP grounding (generic scene/object words)
CLIP_LABELS = [
    "wigs","wig shop","hair","salon","mannequin","mannequin heads","shelf","store",
    "glasses","woman","person","face","indoor","counter","books","classroom",
    "shoes","bags","hats",
]

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))
# trimming the story to not exceed max words

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
#generating caption with blip
def caption_blip(path: str) -> str:
    img = Image.open(path).convert("RGB")
    inputs = blip_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=32)
    text = blip_proc.tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()

def caption_vit(path: str) -> str:
    img = Image.open(path).convert("RGB")
    pixels = vit_cap_proc(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = vit_cap_model.generate(
            pixels,
            max_length=24,
            num_beams=2,
            no_repeat_ngram_size=2,
        )
    return vit_cap_tok.decode(ids[0], skip_special_tokens=True).strip()

def auto_caption(path: str) -> str:
    try:
        txt = caption_blip(path)
        if len(txt.split()) < 2:
            return caption_vit(path)
        return txt
    except Exception:
        return caption_vit(path)

def clip_top_labels(path: str, k: int = 4):
    img = Image.open(path).convert("RGB")
    inputs = clip_proc(text=CLIP_LABELS, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = clip_model(**inputs)
        logits = out.logits_per_image[0]
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
        title_part = "Not generating the title. Starting directly with the story.\n"

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
    return story_tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
def generate_story(user_text: str, temperature: float, top_p: float, max_words: int) -> str:
    prompt_text = qwen_chat_prompt(user_text)
    inputs = story_tok(prompt_text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # Token headroom for full endings
    approx_tokens = int(max_words * 1.7)            
    max_new_tokens = max(180, min(1600, approx_tokens + 80))

    with torch.no_grad():
        outputs = story_model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
            eos_token_id=story_tok.eos_token_id,
            pad_token_id=story_tok.eos_token_id,
        )

    gen_ids = outputs[0][input_len:]
    return story_tok.decode(gen_ids, skip_special_tokens=True).strip()

# takes all the input and generates a caption and based on the caption it writes the story
def infer(image_input, audience, story_type, min_words, max_words, gen_title, temperature, top_p):
    min_words = int(min_words); max_words = int(max_words)
    if min_words < 200: min_words = 200
    if max_words > 1000: max_words = 1000
    if min_words > max_words:
        min_words, max_words = max_words, min_words
        gr.Warning("Swapped min/max to keep a valid range (200–1000).")
    if image_input is None:
        gr.Warning("Please upload an image.")
        return "", ""

    gr.Info("Making a caption")
    image_desc = auto_caption(image_input)

    try:
        force_words = clip_top_labels(image_input, k=4)
    except Exception:
        force_words = None

    user_prompt = build_user_prompt(
        image_desc, audience, story_type, min_words, max_words, gen_title, force_words
    )

    gr.Info("Writing your story")
    raw = generate_story(user_prompt, temperature, top_p, max_words)

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
#this function is helps in downloading the story
def download_story(title: str, story: str):
    if not story.strip():
        gr.Warning("Nothing to download yet generate a story first.")
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

# ---------------- UI (design of the user interface starts here) ----------------
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

with gr.Blocks(css=CSS, theme=THEME, title="Image to Story Generator") as demo:
    with gr.Column(elem_id="col"):
        gr.Markdown(
            "<div style='text-align:center'>"
            "<h1>Image to Story</h1>"
            "<p class='muted' style='margin-top:-4px'>Upload an image and let AI create a captivating story based on what it sees!</p>"
            "<div class='muted' style='font-size:13px'>"
            "</div></div>"
        )

        with gr.Row():
            # -------- Left column is all inpur prompting --------
            with gr.Column(scale=1):
                gr.Markdown("<span class='chip'>Upload Image</span>")
                with gr.Column(elem_classes=["card"]):
                    image_in = gr.Image(label=None, type="filepath", height=320)
                    gr.Markdown("<div class='muted'>Supported: JPG/PNG. Clear scenes with people/objects work best.</div>")

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
                        gr.Markdown("<div class='muted'> 1000</div>")

                with gr.Column(elem_classes=["card"]):
                    gen_title = gr.Checkbox(value=True, label="Generate Title")
                    temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Imagination Level")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Vocabulary breadth")

                with gr.Row():
                    submit_btn = gr.Button("Generate Story", variant="primary", elem_classes=["fullw"])

                gr.Markdown(
                    "<div class='tips' style='margin-top:10px'>"
                    "<strong>Tips for better stories:</strong>"
                    "<ul style='margin:6px 0 0 18px'>"
                    "<li>Use clear, high-quality images</li>"
                    "<li>People, objects, or scenes work best</li>"
                    "<li>Try different genres for varied styles</li>"
                    "<li>Adjust word counts to your preferred length</li>"
                    "</ul></div>"
                )

            # -------- Right column is outputs where you get the story--------
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
    demo.queue().launch(ssr_mode=False)
