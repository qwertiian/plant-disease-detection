# Plant Disease Detection with Explainable AI (Grad-CAM)
# Production-ready Deep Learning System for Plant Pathology Diagnosis

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from gradcam import generate_gradcam, overlay_gradcam

# Set page title and layout
st.set_page_config(
    page_title="Plant Disease Detection with Explainable AI",
    page_icon="🌿",
    layout="wide"
)

# Set Keras cache to workspace folder
os.environ["KERAS_HOME"] = os.path.abspath(".keras")

# Load disease class names
@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

# Load trained models with caching
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

class_names = load_class_names()

# Sidebar: Model Architecture Selection
st.sidebar.title("🌿 Architecture Configuration")
st.sidebar.markdown(
    """
    Select the deep learning architecture for disease diagnosis:
    """
)

model_choice = st.sidebar.radio(
    "Select Model Architecture:",
    ("MobileNetV2 (Transfer Learning)", "Custom CNN (From Scratch)")
)

if model_choice == "Custom CNN (From Scratch)":
    model_path = "custom_cnn_model.keras"
    target_layer_name = "last_conv"
    concept_summary = "End-to-end Convolutional Neural Network featuring sequential Conv2D, MaxPooling2D, and Dense classification layers."
else:
    model_path = "mobilenet_model.keras"
    target_layer_name = "Conv_1"
    concept_summary = "Pre-trained MobileNetV2 architecture with depthwise separable convolutions optimized for edge inference."

# Display Header
st.title("🌿 Plant Disease Detection with Explainable AI")
st.markdown(
    f"""
    **Active Architecture:** `{model_choice}`  
    *{concept_summary}*
    """
)

# Sidebar: Input mode selection
input_option = st.sidebar.selectbox("Choose Image Source:", ["Upload Image", "Sample Leaves"])

image_to_predict = None

if input_option == "Sample Leaves":
    sample_dir = "PlantLeave Dataset/plantvillage dataset/color"
    sample_options = [
        "Tomato___Early_blight",
        "Apple___Cedar_apple_rust",
        "Corn_(maize)___Common_rust_",
        "Potato___Late_blight",
        "Tomato___healthy"
    ]
    chosen_sample_folder = st.sidebar.selectbox("Select Sample Category:", sample_options)
    full_folder_path = os.path.join(sample_dir, chosen_sample_folder)
    
    if os.path.exists(full_folder_path):
        files = [f for f in os.listdir(full_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            sample_img_path = os.path.join(full_folder_path, files[0])
            image_to_predict = Image.open(sample_img_path)
            st.sidebar.info(f"Loaded: {files[0]}")
    else:
        st.sidebar.warning("Sample dataset directory not found locally. Please use 'Upload Image' to test any leaf image.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)

# Load selected model
model = load_model(model_path)

if model is None:
    st.warning(f"⚠️ Model file `{model_path}` not found. Please train the model first by running `python {('train_custom_cnn.py' if 'Custom' in model_choice else 'train_mobilenet.py')}`.")
elif image_to_predict is not None:
    # Prepare image for inference (128x128 matching training dimension)
    resized_img = image_to_predict.resize((128, 128))
    img_array = np.array(resized_img, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)

    # Compute prediction and Grad-CAM heatmap
    heatmap, pred_idx, confidence = generate_gradcam(img_tensor, model, target_layer_name)
    predicted_label = class_names[pred_idx].replace("___", " - ").replace("_", " ")

    # Superimpose heatmap onto leaf image
    overlay_img, raw_heatmap = overlay_gradcam(resized_img, heatmap)

    # Display prediction results
    st.subheader("📊 Diagnostic Results")
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric(label="Predicted Condition", value=predicted_label)
    with col_metric2:
        st.metric(label="Prediction Confidence", value=f"{confidence * 100:.2f}%")

    st.progress(min(float(confidence), 1.0))

    # Display Explainable AI Visualizations
    st.subheader("🔍 Explainable AI: Grad-CAM Heatmap Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(resized_img, caption="1. Input Leaf Image", width="stretch")

    with col2:
        st.image(raw_heatmap, caption=f"2. Grad-CAM Activation ({target_layer_name})", width="stretch")

    with col3:
        st.image(overlay_img, caption="3. Explainable Heatmap Overlay", width="stretch")

    # Scientific Interpretation
    st.info(
        f"**Explainability Breakdown:** The red and yellow highlighted regions indicate the leaf areas "
        f"that exerted the strongest gradient activation on the final convolutional layer (`{target_layer_name}`). "
        f"This visualizes precisely where lesions, discoloration, or rust spots led the model to diagnose **{predicted_label}**."
    )
else:
    st.info("👈 Please upload a leaf image or select a sample leaf from the sidebar to view prediction and Grad-CAM explainability.")
