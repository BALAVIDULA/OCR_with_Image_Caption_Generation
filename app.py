import streamlit as st
import cv2
import pytesseract
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import numpy as np

# Path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the processor and model from disk
processor = BlipProcessor.from_pretrained(r"C:\Users\balav\OneDrive\Desktop\BALU\PROJECTS\OCR_image_context\saved docs")
model = BlipForConditionalGeneration.from_pretrained(r"C:\Users\balav\OneDrive\Desktop\BALU\PROJECTS\OCR_image_context\saved docs")

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def combine_ocr_and_caption(image_path):
    extracted_text = extract_text_from_image(image_path)
    image_caption = generate_caption(image_path)
    combined_description = f"Caption: {image_caption} \n \nExtracted Text: {extracted_text}"
    return combined_description

# Custom CSS to add a background image and style text areas
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://smartengines.com/wp-content/uploads/2021/05/mobile_ocr.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .text-container {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white background */
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("OCR with Image Caption")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("")

    # Process the image and display results
    if st.button("Process Image"):
        result = combine_ocr_and_caption("temp_image.jpg")
        # Use the text-container class to style the result text
        st.markdown(f'<div class="text-container">{result}</div>', unsafe_allow_html=True)
