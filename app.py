import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import torch
import time
import os
import requests
# Streamlit page setup
st.set_page_config(
    page_title="YOLOv11 Live Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile optimization
st.markdown("""
    <style>
        /* General body style */
        .stApp {
            background-color: #0e1117;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Buttons & sliders */
        div.stButton > button {
            width: 100%;
            font-size: 18px !important;
            border-radius: 10px !important;
            padding: 0.6em !important;
            background-color: #00aaff !important;
            color: white !important;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #0088cc !important;
        }

        /* Sidebar header style */
        section[data-testid="stSidebar"] {
            background-color: #1a1d25 !important;
        }

        /* Video frame */
        img {
            border-radius: 10px;
        }

        /* Center titles on mobile */
        @media (max-width: 768px) {
            h1, h2, h3 {
                text-align: center;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.title("📱 YOLOv11 Live Damage Detection")

MODEL_PATH = "car_damage_data_V5_model_6.torchscript"
MODEL_URL = "https://huggingface.co/akil11101/car_parts_and_damage_detection_model/resolve/main/car_damage_data_V5_model_6.torchscript"

# Auto-download the model if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face Hub...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully!")

# Load the YOLO model
model = YOLO(MODEL_PATH)

class_names = {0: 'crack', 1: 'dent', 2: 'glass_shatter', 3: 'lamp_broken', 4: 'scratch', 5: 'tire_flat'}

# Sidebar options
st.sidebar.header("⚙️ Options")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Video"])

# Track webcam session
if "run" not in st.session_state:
    st.session_state.run = False

# Webcam Control
if source == "Webcam":
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("▶️ Start Webcam")
    with col2:
        stop = st.button("⏹️ Stop Webcam")

    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    use_half = torch.cuda.is_available()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam")
            break

        start_time = time.time()

        # YOLO inference
        if use_half:
            results = model(frame, conf=confidence, half=True)
        else:
            results = model(frame, conf=confidence)

        annotated_frame = results[0].plot()
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

    cap.release()
    FRAME_WINDOW.image([])

# Uploaded video mode
elif source == "Upload Video":
    uploaded_file = st.file_uploader("🎥 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.time()
            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            FRAME_WINDOW.image(annotated_frame, channels="BGR")

        cap.release()
