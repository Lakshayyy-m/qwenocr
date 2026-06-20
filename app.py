import streamlit as st
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
import ast
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
from openai import OpenAI

st.set_page_config(page_title="Qwen 2.5 OCR", layout="wide")

# Sidebar for image upload
st.sidebar.title("Qwen 2.5 OCR")
uploaded_file = st.sidebar.file_uploader("Upload an iamge", type = ["jpg", "jpeg", "png"])

# Display the uploaded image in sidebar
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    temp_image_path = "temp_uploaded_image.jpg"

    # Convert RGBA images to RGB before saving as JPEG
    if image.mode == "RGBA":
        image = image.convert('RGB')
    
    image.save(temp_image_path)

# Helper functions
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:]) # Only keeps stuff after this 
            json_output = json_output.split("```")[0] # Remove everything after the closing ```
            break # exit the loop once ```json is found
    return json_output

def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):
    