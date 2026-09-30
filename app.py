import os
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO


st.set_page_config(page_title="OCULAI")
st.title("OCULAI: Retinal Image Segmentation")
st.write("Upload an image to analyze (AMD, Glaucoma, Myopia)")
st.sidebar.header("Settings")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 1.00, 0.25, 0.05)

# model upload "best.pt"
if not os.path.exists("best.pt"):
    st.error("app.py and 'best.pt' must be in the same directory.")
    st.stop()


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Run", type="primary"):
        with st.spinner("Analyzing image..."):
            results = model.predict(source=image, conf=conf_thresh, imgsz=1024)
            res_plotted = results[0].plot(line_width=2)
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            st.image(res_rgb, caption="Result", use_container_width=True)

            st.subheader("📋 Detected:")
            boxes = results[0].boxes

            if len(boxes) == 0:
                st.write("No features detected. Lower confidence threshold.")
            else:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    st.write(f"- **{class_name}** (Confidence: {confidence:.0%})")