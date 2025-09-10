# app.py — Image → Caption → Story (TinyLlama, CPU)
# Radios styled as segmented buttons (accessible & clickable)

import torch
import gradio as gr
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# --------- Models ---------
CAPTION_ID = "nlpconnect/vit-gpt2-image-captioning"
cap_model = VisionEncoderDecoderModel.from_pretrained(CAPTION_ID)
cap_proc  = ViTImageProcessor.from_pretrained(CAPTION_ID)
cap_tok   = AutoTokenizer.from_pretrained(CAPTION_ID)
cap_model.eval()

STORY_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
story_tok = AutoTokenizer.from_pretrained(STORY_ID, use_fast=True)
story_llm = AutoModelForCausalLM.from_pretrained(
    STORY_ID, device_map="cpu", torch_dtype=torch.float32, low_cpu_mem_usage=True
)
if story_llm.config.pad_token_id is None and story_tok.pad_token_id is None:
    story_tok.pad_token = story_tok.eos_token
    story_llm.config.pad_token_id = story_tok.eos_token_id
story_llm.eval()

# --------- Helpers ---------
@torch.inference_mode()
def caption_image(img: Image.Image, max_len: int = 20) -> str:
    pix = cap_proc(images=img, return_tensors="pt").pixel_values
    out = cap_model.generate(
        pix, max_new_tokens=max_len, num_beams=4, do_sample=False, eos_token_id=cap_tok.eos_token_id
    )
    return cap_tok.decode(out[0], skip_special_tokens=True).strip()

def build_prompt(caption: str, genre: str, audience: str, style: str, words: int) -> str:
    return (
        f"You are a kind storyteller for {audience.lower()}.\n"
        f"Write a {genre.lower()} story of about {words} words.\n"
        f"Style: {style}.\n"
        f"Base the story on this image description: {caption}\n"
        f"Keep it positive, safe, and easy to follow."
    )

@torch.inference_mode()
def generate_story(image, genre, audience, style, temperature, top_p, max_new_tokens, show_caption, seed):
    if image is None:
        return "", "Please upload an image."
    torch.manual_seed(int(seed))

    cap = caption_image(image, max_len=20)
    prompt = build_prompt(cap, genre, audience, style, words=max(80, min(max_new_tokens, 300)))

    inputs = story_tok(prompt, return_tensors="pt")
    gen = story_llm.generate(
        **inputs,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        eos_token_id=story_tok.eos_token_id,
        pad_token_id=story_llm.config.pad_token_id,
        repetition_penalty=1.05,
    )
    story = story_tok.decode(gen[0], skip_special_tokens=True).strip()
    return (cap if show_caption else ""), story

# --------- UI (fixed, clickable segmented radios) ---------
SEG_CSS = """
/* Make Radio behave like segmented buttons without breaking clicks */
.seg .wrap { display:flex; flex-wrap:wrap; gap:8px; }
.seg .item { position: relative; }

.seg input[type="radio"]{
  position:absolute; inset:0; width:100%; height:100%;
  margin:0; opacity:0; cursor:pointer;  /* keep it clickable */
}

.seg label{
  display:inline-block;
  padding:10px 14px;
  border:1px solid #d1d5db; border-radius:10px;
  background:#ffffff; color:#111827; user-select:none;
}

/* Selected state – supports both input+label and label>input DOM orders */
.seg input[type="radio"]:focus + label { outline:2px solid #fb923c66; }
.seg input[type="radio"]:checked + label { background:#fb923c; border-color:#fb923c; color:#fff; }
.seg label:has(input[type="radio"]:checked) { background:#fb923c; border-color:#fb923c; color:#fff; }
"""

with gr.Blocks(css=SEG_CSS) as demo:
    gr.Markdown("## Image → Story")

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Upload image")

            audience = gr.Radio(
                ["Children", "Adult"],
                value="Children",
                label="Target Audience",
                elem_classes=["seg"],
            )

            genre = gr.Radio(
                ["Adventure", "Comedy", "Drama", "Fantasy", "Romance", "Mystery", "Sci-Fi", "Slice of Life"],
                value="Fantasy",
                label="Story Genre",
                elem_classes=["seg"],
            )

            style = gr.Textbox(value="calm", label="Style & tone (optional)")

            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Creativity (temperature)")
                top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Top-p")

            with gr.Row():
                max_tokens = gr.Slider(80, 320, value=220, step=10, label="Story length (max new tokens)")
                seed = gr.Number(value=42, precision=0, label="Seed")

            show_caption = gr.Checkbox(value=True, label="Show the auto-caption")
            go = gr.Button("Generate Story", variant="primary")

        with gr.Column():
            cap_out = gr.Textbox(label="Image Caption", interactive=False)
            story_out = gr.Textbox(lines=18, label="Generated Story", interactive=False)

    go.click(
        generate_story,
        inputs=[image, genre, audience, style, temperature, top_p, max_tokens, show_caption, seed],
        outputs=[cap_out, story_out],
    )

if __name__ == "__main__":
    demo.launch()
