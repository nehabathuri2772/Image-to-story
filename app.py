# app.py — Image ➜ Caption ➜ Story
# Simple, clean version:
# - Captioner: nlpconnect/vit-gpt2-image-captioning (CPU-friendly)
# - Story LLM:  TinyLlama/TinyLlama-1.1B-Chat-v1.0 (single model, CPU)
# - Genre & Audience as nice segmented buttons (not dropdowns)

import torch
import gradio as gr
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ------------------ Models ------------------

# 1) Image captioner (lightweight)
CAPTION_ID = "nlpconnect/vit-gpt2-image-captioning"
cap_model = VisionEncoderDecoderModel.from_pretrained(CAPTION_ID)
cap_proc  = ViTImageProcessor.from_pretrained(CAPTION_ID)
cap_tok   = AutoTokenizer.from_pretrained(CAPTION_ID)
cap_model.eval()

# 2) Story writer (TinyLlama, single choice)
STORY_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
story_tok = AutoTokenizer.from_pretrained(STORY_ID, use_fast=True)
story_llm = AutoModelForCausalLM.from_pretrained(
    STORY_ID,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
# some chat models lack a pad token; fall back to EOS
if story_llm.config.pad_token_id is None and story_tok.pad_token_id is None:
    story_tok.pad_token = story_tok.eos_token
    story_llm.config.pad_token_id = story_tok.eos_token_id
story_llm.eval()

# ------------------ Helpers ------------------

@torch.inference_mode()
def caption_image(img: Image.Image, max_len: int = 20) -> str:
    """Generate a short caption for the image."""
    pixel_values = cap_proc(images=img, return_tensors="pt").pixel_values
    out = cap_model.generate(
        pixel_values,
        max_new_tokens=max_len,
        num_beams=4,
        do_sample=False,
        eos_token_id=cap_tok.eos_token_id,
    )
    return cap_tok.decode(out[0], skip_special_tokens=True).strip()

def make_prompt(caption: str, genre: str, audience: str, style: str, words: int) -> str:
    """Simple, model-agnostic instruction prompt."""
    return (
        f"You are a kind storyteller for {audience.lower()}.\n"
        f"Write a {genre.lower()} story of about {words} words.\n"
        f"Style: {style}.\n"
        f"Base the story on this image description: {caption}\n"
        f"Keep it positive, safe, and easy to follow."
    )

@torch.inference_mode()
def run_pipeline(
    image: Image.Image,
    genre: str,
    audience: str,
    style: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    show_caption: bool,
    seed: int,
):
    if image is None:
        return "", "Please upload an image first."

    torch.manual_seed(int(seed))

    # 1) Caption
    cap = caption_image(image, max_len=20)

    # 2) Story
    prompt = make_prompt(cap, genre, audience, style, words=max(80, min(max_new_tokens, 300)))
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

# ------------------ UI ------------------

with gr.Blocks(css="""
/* Make segmented buttons look clean and roomy */
.sbtn .wrap {gap: 8px;}
.sbtn button {padding: 10px 14px; border-radius: 10px;}
""") as demo:
    gr.Markdown("## Image → Story")

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Upload image")

            audience = gr.SegmentedButton(
                ["Children", "Adult"],
                value="Children",
                label="Target Audience",
                elem_classes=["sbtn"],
            )

            genre = gr.SegmentedButton(
                ["Adventure", "Comedy", "Drama", "Fantasy", "Romance", "Mystery", "Sci-Fi", "Slice of Life"],
                value="Fantasy",
                label="Story Genre",
                elem_classes=["sbtn"],
            )

            style = gr.Textbox(
                value="calm, peaceful, heartwarming, descriptive",
                label="Style & tone (optional)",
            )

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
        run_pipeline,
        inputs=[image, genre, audience, style, temperature, top_p, max_tokens, show_caption, seed],
        outputs=[cap_out, story_out],
    )

if __name__ == "__main__":
    demo.launch()

