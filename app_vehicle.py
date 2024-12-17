import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('best_hm.pt')  # Ensure the path to your model is correct

def detect_objects(image):
    # Convert the PIL image to a NumPy array
    img_array = np.array(image)

    # Perform prediction using the YOLO model
    results = model.predict(source=img_array, save=False, conf=0.25)  # Adjust confidence threshold if needed

    # Get the annotated image from results
    annotated_img = results[0].plot()  # Annotated image in NumPy array format

    # Convert NumPy array back to a PIL image for Streamlit display
    annotated_pil_img = Image.fromarray(annotated_img)
    return annotated_pil_img

# Streamlit app layout
st.title("Vehicle Damage Detection App")
st.write("Upload an image to detect damages using YOLOv8.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection
    with st.spinner("Detecting damages..."):
        detected_image = detect_objects(image)

    # Display detected image
    st.image(detected_image, caption="Detected Damages", use_column_width=True)
