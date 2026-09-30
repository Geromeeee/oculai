import os
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

#page setup
st.set_page_config(page_title="OCULAI")
st.title("OCULAI: Multi-Source Instance Segmentation of Retinal Disease Features in Fundus Images")
# sidebar for settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.25, 0.05) #confidence interval
iou_threshold = st.sidebar.slider("IoU Threshold (Overlap)", 0.05, 1.00, 0.45, 0.05) #intersection over union

# model upload "best.pt"
if not os.path.exists("best.pt"):
    st.error("Model not found. Upload 'best,pt' to the same directory as app.py.")
    st.stop()

# load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")
model = load_model()

uploaded_image = st.file_uploader("Upload an image to analyze (AMD, Glaucoma, Myopia)", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Run", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, imgsz=640)
                res_plotted = results[0].plot(line_width=2, boxes=True)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

                st.subheader("Result")
                st.image(res_rgb, use_container_width=True)

                st.subheader("Detected:")
                boxes = results[0].boxes

                if boxes is None or len(boxes) == 0:
                    st.success("Normal")
                    st.write("No pathological features detected.")
                else:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        st.write(f"- {class_name} (Confidence: {confidence:.0%})")

            except Exception as e:
                st.error("Error analyzing image. Ensure image is valid.")