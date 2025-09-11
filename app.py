import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import re

# Initialize models (CPU-compatible, free models)
print("Loading models...")

# Image captioning model - BLIP
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Text generation model - GPT-2 (free and works on CPU)
story_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
story_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# Set pad token
story_tokenizer.pad_token = story_tokenizer.eos_token

print("Models loaded successfully!")

def generate_image_caption(image):
    """Generate a caption from the input image"""
    try:
        inputs = caption_processor(image, return_tensors="pt")
        out = caption_model.generate(**inputs, max_new_tokens=50, num_beams=5)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def create_story_prompt(caption, genre, target_audience):
    """Create a story prompt based on caption, genre, and target audience"""
    
    genre_styles = {
        "Fantasy": "In a magical realm where anything is possible",
        "Mystery": "Something strange and mysterious was happening",
        "Adventure": "An exciting journey was about to begin",
        "Romance": "Love was in the air as",
        "Horror": "In the darkness, something sinister lurked",
        "Comedy": "In a hilarious turn of events",
        "Drama": "Life took an unexpected turn when",
        "Sci-Fi": "In a world of advanced technology and space exploration"
    }
    
    audience_styles = {
        "Children": "Once upon a time, ",
        "Young Adult": "It was a day like any other until ",
        "Adult": "The scene before them revealed ",
        "Family": "Everyone gathered around as "
    }
    
    genre_start = genre_styles.get(genre, "")
    audience_start = audience_styles.get(target_audience, "")
    
    prompt = f"{audience_start}{genre_start.lower()} {caption}. "
    return prompt

def generate_story(image, genre, target_audience, min_words, max_words):
    """Main function to generate story from image"""
    
    if image is None:
        return "Please upload an image to generate a story."
    
    # Validate word count inputs
    if min_words <= 0 or max_words <= 0:
        return "Please enter valid word counts (greater than 0)."
    
    if min_words > max_words:
        return "Minimum words cannot be greater than maximum words."
    
    try:
        # Step 1: Generate image caption
        caption = generate_image_caption(image)
        if caption.startswith("Error"):
            return caption
        
        # Step 2: Create story prompt
        story_prompt = create_story_prompt(caption, genre, target_audience)
        
        # Step 3: Generate story
        inputs = story_tokenizer.encode(story_prompt, return_tensors="pt")
        
        # Calculate approximate token count (rough estimate: 1 word ≈ 1.3 tokens)
        min_tokens = int(min_words * 1.3)
        max_tokens = int(max_words * 1.3)
        
        # Generate story with appropriate parameters
        with torch.no_grad():
            outputs = story_model.generate(
                inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=story_tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_p=0.9
            )
        
        # Decode the generated story
        generated_text = story_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt to get just the generated story
        story = generated_text[len(story_prompt):].strip()
        
        # Clean up the story
        story = re.sub(r'\n+', '\n\n', story)  # Clean up line breaks
        story = story.replace(story_tokenizer.eos_token, '')  # Remove end tokens
        
        # Ensure the story ends with proper punctuation
        if story and story[-1] not in '.!?':
            # Find the last complete sentence
            sentences = re.split(r'[.!?]+', story)
            if len(sentences) > 1:
                story = '.'.join(sentences[:-1]) + '.'
        
        # Word count check
        word_count = len(story.split())
        
        final_story = f"**Generated Story** (Word count: {word_count})\n\n{story_prompt}{story}"
        
        return final_story
        
    except Exception as e:
        return f"Error generating story: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Image to Story Generator", theme=gr.themes.Soft()) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1> Image to Story Generator</h1>
            <p>Upload an image and let AI create a captivating story based on what it sees!</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=300
                )
                
                with gr.Row():
                    genre_input = gr.Dropdown(
                        choices=["Fantasy", "Mystery", "Adventure", "Romance", "Horror", "Comedy", "Drama", "Sci-Fi"],
                        label="Genre",
                        value="Adventure"
                    )
                    
                    audience_input = gr.Dropdown(
                        choices=["Children", "Young Adult", "Adult", "Family"],
                        label="Target Audience",
                        value="Family"
                    )
                
                with gr.Row():
                    min_words_input = gr.Number(
                        label="Minimum Words",
                        value=50,
                        minimum=10,
                        maximum=500
                    )
                    
                    max_words_input = gr.Number(
                        label="Maximum Words",
                        value=200,
                        minimum=20,
                        maximum=1000
                    )
                
                generate_btn = gr.Button("Generate Story", variant="primary", size="lg")
                
                gr.HTML("""
                <div style="margin-top: 20px; padding: 10px; background-color: #f0f8ff; border-radius: 10px;">
                    <h4>💡 Tips for better stories:</h4>
                    <ul>
                        <li>Use clear, high-quality images</li>
                        <li>Images with people, objects, or scenes work best</li>
                        <li>Try different genres for varied storytelling styles</li>
                        <li>Adjust word count based on your preferred story length</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=1):
                # Output component
                story_output = gr.Textbox(
                    label="Generated Story",
                    lines=20,
                    max_lines=30,
                    placeholder="Your generated story will appear here...",
                    show_copy_button=True
                )
        
        # Event handler
        generate_btn.click(
            fn=generate_story,
            inputs=[image_input, genre_input, audience_input, min_words_input, max_words_input],
            outputs=story_output,
            show_progress=True
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
            <h3>How it works:</h3>
            <p><strong>1.</strong> AI analyzes your image and generates a descriptive caption</p>
            <p><strong>2.</strong> The caption is combined with your chosen genre and audience preferences</p>
            <p><strong>3.</strong> A creative story is generated within your specified word count</p>
            <br>
            <p><em>Note: This app uses free, CPU-based models for accessibility. Generation may take 30-60 seconds.</em></p>
        </div>
        """)
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )