import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
MODEL_PATH = "crop_disease_model.h5"
DRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  
import gdown
from tensorflow.keras.models import load_model

# Google Drive file ID (replace with your model's ID)
file_id = "YOUR_FILE_ID_HERE"  
output = "crop_disease_model.h5"

# Download model from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Load the model
model = load_model(output)
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... this may take a minute ⏳"):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
model = load_model()
st.title("🌱 Crop Disease Detection App")
st.write("Upload a leaf image, and the model will predict if it's **healthy or diseased**.")
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize((224, 224))  # match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("✅ Prediction Result")
    st.write(f"**Class:** {class_index}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")