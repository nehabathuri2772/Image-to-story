import gradio as gr
import torch

from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

import os 
hf_token = os.environ.get('HF_TOKEN')
from gradio_client import Client
client = Client("https://fffiloni-test-llama-api.hf.space/", hf_token=hf_token)

def infer(image_input):
    #img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(image_input).convert('RGB')

    prompt = "Can you please describe what's happening in the image, and give information about the characters and the place ?"
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)

   

    llama_q = f"""
    I'll give you a simple image caption, from i want you to provide a story that would fit well with the image:
    '{generated_text}'
    
    """
    
    result = client.predict(
    				llama_q,	# str in 'Message' Textbox component
    				api_name="/predict"
    )

    
    
    
    print(f"Llama2 result: {result}")

    return generated_text, result

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
        image_in = gr.Image(label="Image input", type="filepath")
        submit_btn = gr.Button('Sumbit')
        caption = gr.Textbox(label="Generated Caption")
        story = gr.Textbox(label="generated Story")
    submit_btn.click(fn=infer, inputs=[image_in], outputs=[caption, story])

demo.queue().launch()
