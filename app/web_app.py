import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_image
from src.explain import make_gradcam_heatmap, overlay_heatmap

# Constants
MODEL_PATH = 'models/best_model.h5'
IMG_SIZE = (224, 224)
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

st.set_page_config(page_title="Diabetic Retinopathy Screening", layout="wide")

@st.cache_resource
def load_screening_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

st.title("👁️ Diabetic Retinopathy Screening System")
st.markdown("""
This system uses Deep Learning (ResNet50) to detect Diabetic Retinopathy stages from retinal fundus images.
It provides an **Explainable AI (Grad-CAM)** visualization to highlight suspicious areas.
""")

model = load_screening_model()

if model is None:
    st.warning(f"Model file not found at `{MODEL_PATH}`. Please train the model first.")
else:
    st.sidebar.header("Upload Retinal Scan")
    uploaded_file = st.sidebar.file_uploader("Choose a fundus image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Scan', use_column_width=True)

        # Save uploaded file temporarily for processing
        temp_path = "temp_upload.png"
        image.save(temp_path)

        if st.sidebar.button("Predict & Analyze"):
            with st.spinner("Analyzing retina..."):
                # Preprocess
                processed_img = preprocess_image(temp_path)
                img_array = np.expand_dims(processed_img, axis=0)
                
                # Predict
                preds = model.predict(img_array)
                pred_class_idx = np.argmax(preds[0])
                confidence = np.max(preds[0])
                
                pred_label = CLASSES[pred_class_idx]
                
                # Grad-CAM
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out")
                
                # Denormalize image for overlay
                display_img = np.uint8(processed_img * 255)
                overlay = overlay_heatmap(display_img, heatmap)
            
            # Display Results
            st.sidebar.success("Analysis Complete")
            st.sidebar.metric("Predicted Stage", pred_label)
            st.sidebar.metric("Confidence", f"{confidence*100:.2f}%")
            
            with col2:
                st.subheader("Explainable AI (Grad-CAM)")
                st.image(overlay, caption=f'Lesion Heatmap ({pred_label})', use_column_width=True)
                
            st.info("The heatmap highlights areas (red/yellow) that most influenced the model's decision.")
            
            # Risk Level
            risk_level = "Low" if pred_class_idx < 2 else "Moderate" if pred_class_idx == 2 else "High"
            risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Moderate" else "red"
            st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
