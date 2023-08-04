import gradio as gr

import os 
hf_token = os.environ.get('HF_TOKEN')
from gradio_client import Client
client = Client("https://fffiloni-test-llama-api.hf.space/", hf_token=hf_token)

clipi_client = Client("https://fffiloni-clip-interrogator-2.hf.space/")


def infer(image_input):
    
    clipi_result = clipi_client.predict(
    				image_input,	# str (filepath or URL to image) in 'parameter_3' Image component
    				"best",	# str in 'Select mode' Radio component
    				6,	# int | float (numeric value between 2 and 24) in 'best mode max flavors' Slider component
    				api_name="/clipi2"
    )
    print(clipi_result)
   

    llama_q = f"""
    I'll give you a simple image caption, from i want you to provide a story that would fit well with the image:
    '{clipi_result}'
    
    """
    
    result = client.predict(
    				llama_q,	# str in 'Message' Textbox component
    				api_name="/predict"
    )

    print(f"Llama2 result: {result}")

    return clipi_result, result

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
            """
        )
        image_in = gr.Image(label="Image input", type="filepath")
        submit_btn = gr.Button('Sumbit')
        caption = gr.Textbox(label="Generated Caption")
        story = gr.Textbox(label="generated Story")
    submit_btn.click(fn=infer, inputs=[image_in], outputs=[caption, story])

demo.queue().launch()
