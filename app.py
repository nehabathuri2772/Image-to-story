import gradio as gr

import os 
hf_token = os.environ.get('HF_TOKEN')
from gradio_client import Client
client = Client("https://fffiloni-test-llama-api.hf.space/", hf_token=hf_token)

clipi_client = Client("https://fffiloni-clip-interrogator-2.hf.space/")

def get_text_after_colon(input_text):
    # Find the first occurrence of ":"
    colon_index = input_text.find(":")
    
    # Check if ":" exists in the input_text
    if colon_index != -1:
        # Extract the text after the colon
        result_text = input_text[colon_index + 1:].strip()
        return result_text
    else:
        # Return the original text if ":" is not found
        return input_text

def infer(image_input):
    
    clipi_result = clipi_client.predict(
    				image_input,	# str (filepath or URL to image) in 'parameter_3' Image component
    				"best",	# str in 'Select mode' Radio component
    				4,	# int | float (numeric value between 2 and 24) in 'best mode max flavors' Slider component
    				api_name="/clipi2"
    )
    print(clipi_result)
   

    llama_q = f"""
    I'll give you a simple image caption, from i want you to provide a story that would fit well with the image:
    '{clipi_result[0]}'
    
    """
    
    result = client.predict(
    				llama_q,	# str in 'Message' Textbox component
    				api_name="/predict"
    )

    print(f"Llama2 result: {result}")

    result = get_text_after_colon(result)

    return result

css="""
#col-container {max-width: 910px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center">Image to Story</h1>
            <p style="text-align: center">Upload an image, get a story made by Llama2 !</p>
            """
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image input", type="filepath")
                submit_btn = gr.Button('Tell me a story')
            with gr.Column():
                #caption = gr.Textbox(label="Generated Caption")
                story = gr.Textbox(label="generated Story")
        gr.Examples(examples=[["./examples/crabby.png"],["./examples/hopper.jpeg"]],
                    fn=infer,
                    inputs=[image_in],
                    outputs=[story],
                    cache_examples=True
                   )
    submit_btn.click(fn=infer, inputs=[image_in], outputs=[story])

demo.queue(max_size=12).launch()
