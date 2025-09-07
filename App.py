import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load pretrained model
model = load_model("crop_disease_model.h5")

# Class labels
class_labels = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Corn Gray Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
]

st.title("🌱 Crop Disease Detector")
st.write("Upload a leaf image to check if it's healthy or diseased.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    # Resize to model input
    img = img.resize((256,256))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    if class_idx < len(class_labels):
        st.success(f"Prediction: {class_labels[class_idx]}")
    else:
        st.error("Prediction index out of range. Please check labels.")
