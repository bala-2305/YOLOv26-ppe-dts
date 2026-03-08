import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- App Config ---
st.set_page_config(page_title="PPE Safety Monitor - YOLO26", layout="wide")
st.title("🚧 Real-Time PPE Detection (YOLO26)")
st.sidebar.header("Settings")

# --- Load YOLO26 Model ---
# Replace 'best.pt' with your custom trained PPE model path if different
@st.cache_resource
def load_model():
    # using 'best.pt' by default as your trained PPE model
    return YOLO("best.pt") 

model = load_model()

# Confidence threshold slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)

# --- Source Selection ---
source = st.sidebar.radio("Select Source", ("Image Upload", "Webcam (Live)"))

if source == "Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        # Convert PIL to OpenCV format
        img_array = np.array(img)
        
        # Run Inference
        results = model.predict(source=img_array, conf=conf_threshold)
        
        # Plot results
        res_plotted = results[0].plot()
        
        # display result image; specify a fixed width instead of deprecated use_column_width
        st.image(res_plotted, caption="Detection Results", width=700)
        
        # Breakdown of detections
        st.subheader("Safety Checklist")
        labels = results[0].boxes.cls.tolist()
        names = model.names
        total = len(labels)
        # display counts per class
        if total:
            st.write(f"🔢 Total objects detected: **{total}**")
            counts = {}
            for lbl in labels:
                name = names[int(lbl)]
                counts[name] = counts.get(name, 0) + 1
            for name, cnt in counts.items():
                st.write(f"- {name}: {cnt}")
            # simple danger check: look for keywords
            danger_keywords = ["danger", "hazard", "no ", "missing", "unsafe"]
            danger_count = sum(cnt for name, cnt in counts.items() if any(kw in name.lower() for kw in danger_keywords))
            if danger_count:
                st.warning(f"⚠️ Objects in danger-related classes: {danger_count}")
        else:
            st.write("❌ **No PPE detected**")

elif source == "Webcam (Live)":
    st.info("Ensure your browser has webcam permissions enabled.")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO26 Inference
        # Note: YOLO26 is NMS-free, making it much faster for live frames
        results = model.predict(source=frame, conf=conf_threshold)
        
        res_plotted = results[0].plot()
        FRAME_WINDOW.image(res_plotted)

        # feedback when nothing detected and show summary
        labels = results[0].boxes.cls.tolist()
        if not labels:
            status_text.warning("No PPE detected in current frame.")
        else:
            total = len(labels)
            counts = {}
            for lbl in labels:
                name = model.names[int(lbl)]
                counts[name] = counts.get(name, 0) + 1
            danger_keywords = ["danger", "hazard", "no ", "missing", "unsafe"]
            danger_count = sum(cnt for name, cnt in counts.items() if any(kw in name.lower() for kw in danger_keywords))
            status_text.info(f"Detected {total} object(s){' – danger: '+str(danger_count) if danger_count else ''}")
    else:
        camera.release()
        st.write("Webcam Stopped.")
