# ===============================================================
# app.py - Oral Cancer Detection Web UI (Streamlit)
# ===============================================================

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# -------------------------------
# Load the trained model
# -------------------------------
MODEL_PATH = os.path.join("..", "model", "oral_cancer_efficientnet.keras")
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
else:
    model = load_model(MODEL_PATH)

# -------------------------------
# Define class labels
# -------------------------------
class_labels = ["NON CANCER", "CANCER"]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Oral Cancer Detection", layout="centered")
st.title("🦷 Oral Cancer Detection")
st.write(
    """
    Upload an image of the oral cavity, and the model will predict 
    whether it indicates cancer or not.
    """
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    # Display result
    if confidence > 0.5:
        st.error(f"Prediction: **CANCER** ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: **NON CANCER** ({(1-confidence)*100:.2f}% confidence)")

    # Optional: Show probability bar
    st.write("Prediction Confidence:")
    st.progress(confidence if confidence > 0.5 else 1-confidence)
