import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import pyttsx3

# -------------------------------
# 🔧 PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="✨ Gesture Genie", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #fff8fc; }
        h1 { color: #ff69b4; }
        .stButton>button {
            background-color: #ffb6c1;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .stSidebar { 
            position: fixed;
            right: 0;
            top: 0;
            width: 25%;
            padding: 10px;
            background-color: #ffe6f1;
            box-shadow: -5px 0px 15px rgba(0, 0, 0, 0.1);
        }
        .stApp {
            padding-right: 28%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Gesture Genie")
st.caption("Express yourself with sign language recognition")

# -------------------------------
# 🔊 INIT TEXT-TO-SPEECH ENGINE
# -------------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# -------------------------------
# 📦 CLASS NAMES
# -------------------------------
CLASS_NAMES = ['beautiful', 'best of luck', 'call', 'camera', 'eat', 'hey', 'hurt', 'love', 'namaste',
               'no', 'opticals', 'paisa', 'sad', 'shut up', 'smile', 'sorry', 'strong', 'thinking']

# -------------------------------
# 📥 LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\Lenovo\Downloads\Gesture-genie\pyolo_model_11_best.pt")

model = load_model()

# -------------------------------
# 🔊 Speak Detected Labels
# -------------------------------
def speak_labels(result):
    if not result.boxes or len(result.boxes) == 0:
        tts_engine.say("No gesture detected.")
    else:
        spoken_labels = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
            spoken_labels.append(label)
        tts_engine.say("Detected: " + ", ".join(spoken_labels))
    tts_engine.runAndWait()

# -------------------------------
# 🔍 IMAGE DETECTION
# -------------------------------
def detect_image(image):
    resized = cv2.resize(image, (640, 640))
    results = model(resized, conf=0.25)
    result = results[0]
    annotated = result.plot()
    return annotated, result

def format_results(result):
    if not result.boxes or len(result.boxes) == 0:
        return "No gestures detected."
    out = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
        out.append(f"**{label}**: {conf * 100:.2f}%")
    return "\n".join(out)

# -------------------------------
# 📁 SIDEBAR FILE UPLOAD
# -------------------------------
st.sidebar.header("📁 Upload File")
uploaded_file = st.sidebar.file_uploader("Choose Image or Video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv", "mpeg4"])
st.sidebar.markdown("Limit 200MB per file")

# Optional volume control
volume = st.sidebar.slider("🔊 Volume", 0.0, 1.0, 0.8)
tts_engine.setProperty('volume', volume)

# -------------------------------
# 📤 HANDLE UPLOADED FILE
# -------------------------------
if uploaded_file:
    suffix = uploaded_file.name.split(".")[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    temp_file.write(uploaded_file.getvalue())
    file_path = temp_file.name

    if suffix.lower() in ["jpg", "jpeg", "png", "bmp"]:
        st.subheader("🖼️ Uploaded Image")
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated, result = detect_image(img_rgb)
        speak_labels(result)
        st.image(annotated, caption="Detected Gesture", use_container_width=True)
        st.code(format_results(result), language="markdown")

    elif suffix.lower() in ["mp4", "avi", "mov", "mkv", "mpeg4"]:
        st.subheader("🎞️ Processing Uploaded Video")
        cap = cv2.VideoCapture(file_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        progress = st.progress(0)
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (640, 640))
            result = model(resized, conf=0.25)[0]
            annotated = result.plot()
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            out.write(annotated_bgr)
            count += 1
            progress.progress(min(count / total, 1.0))

        cap.release()
        out.release()
        st.video(output_path)
        st.success("✅ Video processed successfully!")

# -------------------------------
# 📸 LIVE CAMERA DETECTION (AUTO + AUDIO)
# -------------------------------
st.subheader("📸 Live Camera Detection")

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

start = st.button("📷 Start Live Detection", key="start_live")
stop = st.button("🛑 Stop Detection", key="stop_live")

FRAME_WINDOW = st.empty()

if start:
    st.session_state.camera_on = True
if stop:
    st.session_state.camera_on = False

if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access the camera.")
        st.session_state.camera_on = False
    else:
        st.info("🎥 Detecting gestures in real-time with audio...")

        frame_count = 0
        spoken_label = ""

        while st.session_state.camera_on and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, result = detect_image(frame)
            FRAME_WINDOW.image(annotated, channels="RGB", use_container_width=True)

            frame_count += 1
            if frame_count % 30 == 0:  # Speak every ~1 second
                if result.boxes and len(result.boxes) > 0:
                    cls_id = int(result.boxes[0].cls[0])
                    label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
                    if label != spoken_label:
                        spoken_label = label
                        tts_engine.say(f"Detected {label}")
                        tts_engine.runAndWait()

        cap.release()
        FRAME_WINDOW.empty()
        st.success("📴 Camera stopped.")

    