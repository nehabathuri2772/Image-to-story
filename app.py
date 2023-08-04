import gradio as gr
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

def infer(image_input):
    #img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(image_input, stream=True).raw).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    print(caption)

    return caption

css="""
#col-container {max-width: 910px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            # Image to Story
            Upload an image, get a story ! 
            <br/>
            <br/>
            [![Duplicate this Space](https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm.svg)](https://huggingface.co/spaces/fffiloni/SplitTrack2MusicGen?duplicate=true) for longer audio, more control and no queue.</p>
            """
        )
        image_in = gr.Image(label="Image input")
        submit_btn = gr.Button('Sumbit')
        story = gr.Textbox(label="Generated Story")
    submit_btn.click(fn=infer, inputs=[image_in], outputs=[story])

demo.queue().launch()
