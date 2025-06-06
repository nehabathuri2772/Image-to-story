import gradio as gr
import re
import os 
hf_token = os.environ.get('HF_TOKEN')
from gradio_client import Client

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=hf_token).half().cuda()

#client = Client("https://fffiloni-test-llama-api-debug.hf.space/", hf_token=hf_token)

clipi_client = Client("https://fffiloni-clip-interrogator-2.hf.space/")

def llama_gen_story(prompt):

    instruction = """[INST] <<SYS>>\nYou are a storyteller. You'll be given an image description and some keyword about the image. 
            For that given you'll be asked to generate a story that you think could fit very well with the image provided.
            Always answer with a cool story, while being safe as possible.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

    
    prompt = instruction.format(prompt)
    
    generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096)
    output_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    #print(generate_ids)
    #print(output_text)
    pattern = r'\[INST\].*?\[/INST\]'
    cleaned_text = re.sub(pattern, '', output_text, flags=re.DOTALL)
    return cleaned_text

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
    result = llama_gen_story(llama_q)

    print(f"Llama2 result: {result}")

    result = get_text_after_colon(result)

    # Split the text into paragraphs based on actual line breaks
    paragraphs = result.split('\n')
    
    # Join the paragraphs back with an extra empty line between each paragraph
    formatted_text = '\n\n'.join(paragraphs)


    return formatted_text

css="""
#col-container {max-width: 910px; margin-left: auto; margin-right: auto;}


div#story textarea {
    font-size: 1.5em;
    line-height: 1.4em;
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
                story = gr.Textbox(label="generated Story", elem_id="story", height=420)
        
        gr.Examples(examples=[["./examples/crabby.png", "Children"],["./examples/hopper.jpeg", "Adult"]],
                    fn=infer,
                    inputs=[image_in, audience],
                    outputs=[story],
                    cache_examples=True
                   )
        
    submit_btn.click(fn=infer, inputs=[image_in, audience], outputs=[story])

demo.queue(max_size=12).launch()
