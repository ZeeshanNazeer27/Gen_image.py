import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image

@st.cache_resource
def load_pipeline():
    # Load the stable diffusion pipeline
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
    return pipeline.to("cuda" if torch.cuda.is_available() else "cpu")  # Move the model to GPU if available

pipe = load_pipeline()

# Streamlit app
st.title("Text-to-Image Generator")

# Text prompt input
prompt = st.text_input("Enter a text prompt for image generation:", "a cricket stadium with match going on in it")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Generate the image
        image = pipe(prompt).images[0]
        
        # Save the image
        image.save("generated_image.png")
        
        # Display the image
        st.image(image, caption="Generated Image", use_column_width=True)
