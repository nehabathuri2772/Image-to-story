# This is a image to story
# given a image the application tells you a story based on it.
# we have title generation, story generation and download story option for this.


import os
import re
import tempfile
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
    CLIPProcessor,
    CLIPModel,
)

# ------------- MODELS -------------
# Image captioner (small, CPU OK)
CAPTION_MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
cap_model = VisionEncoderDecoderModel.from_pretrained(CAPTION_MODEL_ID)
cap_processor = ViTImageProcessor.from_pretrained(CAPTION_MODEL_ID)
cap_tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL_ID)

# Story model: Qwen 2.5 (1.5B Instruct) - CPU friendly
STORY_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
story_tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_ID, use_fast=True)
story_model = AutoModelForCausalLM.from_pretrained(
    STORY_MODEL_ID,
    device_map="cpu",
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# CLIP for image-text grounding score (reranking)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

# ------------- HELPERS -------------

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


def caption_image_local(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    pixel_values = cap_processor(images=img, return_tensors="pt").pixel_values
    output_ids = cap_model.generate(
        pixel_values,
        max_length=24,
        num_beams=4,
        no_repeat_ngram_size=2,
    )
    caption = cap_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def build_user_prompt(
    image_desc: str,
    audience: str,
    story_type: str,
    min_words: int,
    max_words: int,
    gen_title: bool,
    keywords=None,
) -> str:
    title_part = (
        "Start with one line exactly like: Title: <concise, original title>\n"
        "Then a blank line, then the story.\n"
    ) if gen_title else (
        "Do NOT include any title line. Start directly with the story.\n"
    )

    kw_line = ""
    if keywords:
        kw_str = ", ".join(keywords)
        kw_line = (
            "
Grounding: In the FIRST paragraph, explicitly mention at least 3 of these words verbatim: "
            + kw_str + "."
        )

    return f"""
Write a **{story_type.lower()}** short story for a **{audience.lower()}** audience based ONLY on this image description:
{image_desc}

Constraints:
- Length: between {min_words} and {max_words} words.
- Tone/genre: {story_type}.
- No meta commentary or warnings.
- Keep it self-contained and vivid.{kw_line}

Formatting:
{title_part}
If you include a title, put nothing else on the title line except the title itself.
""".strip()


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
    max_new_tokens = max(128, min(1200, int(max_words * 1.5)))
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


def extract_keywords(text, k: int = 8):
    stop = {
        "a","an","the","with","of","and","in","on","at","for","to","from","by","black","white","gray","colour","color"
    }
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]*", text)]
    keep = []
    for w in words:
        if w not in stop and w not in keep:
            keep.append(w)
        if len(keep) >= k:
            break
    return keep


def clip_score(image_path: str, text: str) -> float:
    img = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text], images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = clip_model(**inputs)
        score = out.logits_per_image[0].item()
    return float(score)

# ------------- CORE CALLBACKS -------------

def infer(image_input, audience, story_type, min_words, max_words, gen_title, temperature, top_p):
    min_words = int(min_words)
    max_words = int(max_words)

    # Clamp + sanity
    if min_words < 200:
        min_words = 200
    if max_words > 1000:
        max_words = 1000
    if min_words > max_words:
        min_words, max_words = max_words, min_words
        gr.Warning("Swapped min/max to keep a valid range (200–1000).")

    if image_input is None:
        gr.Warning("Please upload an image.")
        return "", ""

    gr.Info("Making a caption (free, local)…")
    image_desc = caption_image_local(image_input)

    # Extract keywords from caption to ground the story
    keywords = extract_keywords(image_desc)

    user_prompt = build_user_prompt(
        image_desc, audience, story_type, min_words, max_words, gen_title, keywords
    )

    gr.Info("Writing your story with Qwen 2.5 (CPU)…")

    # Generate several candidates and pick the one most aligned to the image using CLIP
    candidates = []
    for t in [temperature, min(temperature + 0.1, 1.3), max(temperature - 0.1, 0.3)]:
        raw = generate_story_with_qwen(user_prompt, t, top_p, max_words)
        ti, st = parse_title_and_story(raw)
        st = "

".join(p for p in st.split("
") if p.strip())
        candidates.append((ti, st))

    scores = [clip_score(image_input, (ti + "

" + st).strip()) for ti, st in candidates]
    best_idx = int(np.argmax(scores))
    title, story = candidates[best_idx]

    wc = word_count(story)
    if wc > max_words:
        story = smart_trim_to_max_words(story, max_words)
        wc = word_count(story)
        gr.Info(f"Trimmed to {wc} words to respect the {max_words}-word limit.")
    elif wc < min_words:
        gr.Warning(
            f"Generated {wc} words (target {min_words}-{max_words}). Try lowering min or raising temperature."
        )

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
    return None, "Children", "Adventure", 400, 700, True, 0.9, 0.95, "", ""


# ------------- UI -------------

CSS = """

:root { --card: transparent; }
body { background: #ffffff; }
#col-container { max-width: 1100px; margin-left:auto; margin-right:auto; padding: 8px; }

.glass { background: transparent; backdrop-filter: none; -webkit-backdrop-filter: none; border: 1px solid #e5e7eb; box-shadow: none; border-radius: 16px; padding: 16px; }

#story textarea { font-size: 1.04em; line-height: 1.6em; color: #111; }

.gradio-container .prose h1 { color: #111; background: none; }

button { border-radius: 9999px !important; }

[data-testid="block-label"], [data-testid="block-label"] * { background: transparent !important; color: #111 !important; box-shadow: none !important; border: none !important; }
.badge, .tag, .token, .label, .label-wrap { background: transparent !important; color: #111 !important; box-shadow: none !important; border: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
)

with gr.Blocks(css=CSS, theme=THEME, title="Image to Story") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <div style="text-align:center">
              <h1>Image to Story</h1>
              <p style="opacity:.9">Upload an image, pick a genre and get a story.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(elem_classes=["glass"]):
                image_in = gr.Image(label="Drop image here", type="filepath", height=320)
                audience = gr.Radio(
                    label="Target Audience", choices=["Children", "Adult"], value="Children"
                )
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
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top‑p")

                with gr.Row():
                    submit_btn = gr.Button(' Generate story', variant="primary")
                    reset_btn = gr.Button('↺ Reset')

                # Example images (optional)
                gr.Examples(
                    examples=[["examples/hopper.jpeg"], ["examples/crabby.png"]],
                    inputs=[image_in],
                    label="Try an example image",
                )

            with gr.Column(elem_classes=["glass"]):
                title_out = gr.Textbox(
                    label="Title", interactive=False, placeholder="(Title will appear here)", show_copy_button=True
                )
                story_out = gr.Textbox(
                    label="Story", elem_id="story", lines=20, show_copy_button=True
                )
                download_btn = gr.DownloadButton("Download Story")

        # Wire up events
        submit_btn.click(
            fn=infer,
            inputs=[image_in, audience, story_type, min_words, max_words, gen_title, temperature, top_p],
            outputs=[title_out, story_out],
        )

        download_btn.click(fn=download_story, inputs=[title_out, story_out], outputs=download_btn)
        reset_btn.click(
            fn=reset_all,
            inputs=None,
            outputs=[image_in, audience, story_type, min_words, max_words, gen_title, temperature, top_p, title_out, story_out],
        )

        gr.Markdown(
            """
            ---
            **Credits & Citations**  
            • Base idea: fffiloni/Image-to-Story (Hugging Face Space)  
            • Models: `nlpconnect/vit-gpt2-image-captioning` (captioning) + `Qwen/Qwen2.5-1.5B-Instruct` (story)  
            • Interface enhancements by nehabathuri 
            • LLM help acknowledged (ChatGPT)
            """
        )

if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
