import gradio as gr
from share_btn import community_icon_html, loading_icon_html, share_js
import re
import os 
hf_token = os.environ.get('HF_TOKEN')
from gradio_client import Client
client = Client("https://fffiloni-test-llama-api-debug.hf.space/", hf_token=hf_token)

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

def infer(image_input, audience):
    gr.Info('Calling CLIP Interrogator ...')
    clipi_result = clipi_client.predict(
    				image_input,	# str (filepath or URL to image) in 'parameter_3' Image component
    				"best",	# str in 'Select mode' Radio component
    				4,	# int | float (numeric value between 2 and 24) in 'best mode max flavors' Slider component
    				api_name="/clipi2"
    )
    print(clipi_result)
   

    llama_q = f"""
    I'll give you a simple image caption, please provide a fictional story for a {audience} audience that would fit well with the image. Please be creative, do not worry and only generate a cool fictional story. 
    Here's the image description: 
    '{clipi_result[0]}'
    
    """
    gr.Info('Calling Llama2 ...')
    result = client.predict(
    				llama_q,	# str in 'Message' Textbox component
                    "I2S",
    				api_name="/predict"
    )

    print(f"Llama2 result: {result}")

    result = get_text_after_colon(result)

    # Split the text into paragraphs based on actual line breaks
    paragraphs = result.split('\n')
    
    # Join the paragraphs back with an extra empty line between each paragraph
    formatted_text = '\n\n'.join(paragraphs)


    return formatted_text, gr.Group.update(visible=True)

css="""
#col-container {max-width: 910px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from {
      transform: rotate(0deg);
  }
  to {
      transform: rotate(360deg);
  }
}
#share-btn-container {
  display: flex; 
  padding-left: 0.5rem !important; 
  padding-right: 0.5rem !important; 
  background-color: #000000; 
  justify-content: center; 
  align-items: center; 
  border-radius: 9999px !important; 
  max-width: 13rem;
}
div#share-btn-container > div {
    flex-direction: row;
    background: black;
    align-items: center;
}
#share-btn-container:hover {
  background-color: #060606;
}
#share-btn {
  all: initial; 
  color: #ffffff;
  font-weight: 600; 
  cursor:pointer; 
  font-family: 'IBM Plex Sans', sans-serif; 
  margin-left: 0.5rem !important; 
  padding-top: 0.5rem !important; 
  padding-bottom: 0.5rem !important;
  right:0;
}
#share-btn * {
  all: unset;
}
#share-btn-container div:nth-child(-n+2){
  width: auto !important;
  min-height: 0px !important;
}
#share-btn-container .wrap {
  display: none !important;
}
#share-btn-container.hidden {
  display: none!important;
}
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
                image_in = gr.Image(label="Image input", type="filepath", elem_id="image-in", height=420)
                audience = gr.Radio(label="Target Audience", choices=["Children", "Adult"], value="Children")
                submit_btn = gr.Button('Tell me a story')
            with gr.Column():
                #caption = gr.Textbox(label="Generated Caption")
                story = gr.Textbox(label="generated Story", elem_id="story")

                with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                    community_icon = gr.HTML(community_icon_html)
                    loading_icon = gr.HTML(loading_icon_html)
                    share_button = gr.Button("Share to community", elem_id="share-btn")
        
        gr.Examples(examples=[["./examples/crabby.png", "Children"],["./examples/hopper.jpeg", "Adult"]],
                    fn=infer,
                    inputs=[image_in, audience],
                    outputs=[story, share_group],
                    cache_examples=True
                   )
        
    submit_btn.click(fn=infer, inputs=[image_in, audience], outputs=[story, share_group])
    share_button.click(None, [], [], _js=share_js)

demo.queue(max_size=12).launch()
