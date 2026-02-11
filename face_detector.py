# face_detector.py - PROJECT #5 COMPUTER VISION
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Custom CSS
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)}
.stButton > button {background: linear-gradient(45deg, #ff416c, #ff4b2b); border-radius: 25px}
</style>
""", unsafe_allow_html=True)

st.title("👁️ **Real-Time Face Detector**")
st.markdown("### *Project #5 | YOLOv8 | OpenCV | ML Portfolio*")

@st.cache_resource
def load_model():
    """Load YOLOv8 face detection model"""
    model = YOLO('yolov8n.pt')  # Nano model (fast)
    return model

model = load_model()

# Sidebar
st.sidebar.title("📸 Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5)
max_detections = st.sidebar.slider("Max Faces", 1, 10, 3)

# File uploader
uploaded_file = st.file_uploader("Choose image or video", 
                               type=['png','jpg','jpeg','mp4','avi'])

if uploaded_file is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if uploaded_file.type.startswith('video') else '.jpg')
    tfile.write(uploaded_file.read())
    file_path = tfile.name
    tfile.close()

    st.success(f"✅ Loaded: {uploaded_file.name}")
    
    if uploaded_file.type.startswith('image'):
        # Single image
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect
        results = model(image_rgb, conf=confidence, max_det=max_detections)
        
        # Plot results
        annotated = results[0].plot()
        st.image(annotated, caption="Detected Faces", use_column_width=True)
        
        # Stats
        boxes = results[0].boxes
        if boxes is not None:
            st.metric("Faces Found", len(boxes))
            st.write("**Confidence Scores:**")
            for i, conf in enumerate(boxes.conf):
                st.write(f"Face {i+1}: {conf:.2f}")
    
    elif uploaded_file.type.startswith('video'):
        st.video(file_path)
        st.info("Video face detection coming in v2!")

# Webcam (Streamlit 1.28+)
if st.checkbox("📱 Use Webcam (Beta)"):
    st.info("Webcam support in next update!")

# Instructions
with st.expander("🚀 Deploy Instructions"):
    st.code("""
# 1. Install
pip install ultralytics opencv-python streamlit

# 2. Run locally
streamlit run face_detector.py

# 3. GitHub + share.streamlit.io
    """)

st.markdown("---")
st.markdown("*Project #5 Computer Vision | [GitHub](https://github.com/HARISH77224/face-detector) | Day 44*")
#streamlit run face_detector.py