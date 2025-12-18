# Gradio-Tutorial-
Overview:
This tutorial demonstrates how to build a simple, interactive Gradio app called “MoodArt”, which allows users to input a descriptive mood or scene, adjust sliders for image brightness and style, and instantly view a generated placeholder image along with a detected emotion. The app demonstrates the use of Gradio layouts, including rows, columns, sliders, text input fields, and image outputs, providing a hands-on introduction to creating user-friendly interfaces for AI applications.
Hugging Face Integration:
The app integrates the Hugging Face Transformers library through a text-classification pipeline using the model j-hartmann/emotion-english-distilroberta-base. This pipeline analyzes the user-provided prompt and predicts the most likely emotion label. While the image generation in this tutorial uses a CPU-friendly placeholder for fast prototyping, the structure of the app allows easy replacement with GPU-based models for real AI-generated images in more advanced setups.

