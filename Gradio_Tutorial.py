
### MoodArt Gradio Tutorial ###
# This tutorial demonstrates a Gradio app with sliders, text input, images, rows/columns,
# and integration with Hugging Face Transformers pipelines.

# -------------------------------------------------
# Step 1: Install Dependencies 
# -------------------------------------------------
# pip install gradio transformers pillow

# -------------------------------------------------
# Step 2: Import Libraries
# -------------------------------------------------
import gradio as gr
from transformers import pipeline
from PIL import Image, ImageDraw

# -------------------------------------------------
# Step 3: Load Hugging Face Pipeline
# -------------------------------------------------
# Text-classification pipeline for emotion detection
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base")

# -------------------------------------------------
# Step 4: Placeholder Image Generator (due to the limit of the cpu machine,
# here we just generate a visible image to demenstrate the idea)
# -------------------------------------------------
def generate_image(prompt, brightness, style):
    img = Image.new("RGB", (400, 400), (brightness, style, 150))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), prompt, fill=(255, 255, 255))
    return img

# Brightness slider: red channel
# Style slider: green channel
# (In a GPU environment, can replace with a real text-to-image model)

# -------------------------------------------------
# Step 5: App Logic
# -------------------------------------------------
def mood_app(prompt, brightness, style):
    image = generate_image(prompt, brightness, style)
    emotion = emotion_classifier(prompt)[0]
    return image, emotion["label"]

# -------------------------------------------------
# Step 6: Build Gradio Interface
# -------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# MoodArt Demo")

    # Text input
    prompt = gr.Textbox(label="Describe a mood or scene")

    # Sliders for image properties
    with gr.Row():
        brightness = gr.Slider(0, 255, value=120, label="Brightness")
        style = gr.Slider(0, 255, value=180, label="Style")

    # Generate button
    run = gr.Button("Generate")

    # Outputs: image and emotion
    with gr.Row():
        output_image = gr.Image(label="Generated Image")
        with gr.Column():
            output_emotion = gr.Textbox(label="Detected Emotion")

    # Connect button to function
    run.click(
        mood_app,
        inputs=[prompt, brightness, style],
        outputs=[output_image, output_emotion])

# -------------------------------------------------
# Step 7: Launch the App
# -------------------------------------------------
demo.launch()

# -----------------------------------------------------------------------------
# Step 8: Open the URL displayed in the terminal (e.g., http://127.0.0.1:7860)
# -----------------------------------------------------------------------------
