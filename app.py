import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Set the title and description of the app
st.title("Black Dot Detection App")
st.write("Upload an image with a black dot, and the app will find the dot and give its center coordinates.")

# File uploader in the UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Load the YOLOv8 model (using the custom weights)
    model = YOLO("best.pt")  # Ensure best.pt is in the same directory
    # Perform object detection on the image
    results = model(img_array)  # predict on the image  [oai_citation:1‡medium.com](https://medium.com/@codeaigo/building-an-object-detection-app-with-yolov8-and-streamlit-d3aa416f7b6a#:~:text=uploaded_file%20%3D%20st.file_uploader%28,Processed%20Image%20with%20Detections) [oai_citation:2‡github.com](https://github.com/ultralytics/ultralytics/issues/13419#:~:text=from%20ultralytics%20import%20YOLO)

    # Check if any detection is found
    result = results[0]  # first (and only) image's result
    annotated_img = result.plot()  # draw boxes on the image
    st.image(annotated_img, caption="Image with detected dot")
    if result.boxes:
        # Get the first detected box coordinates
        box = result.boxes[0]
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2] of the bounding box  [oai_citation:3‡stackoverflow.com](https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box#:~:text=for%20box%20in%20r)
        x1, y1, x2, y2 = coords
        # Calculate center of the box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # Display the coordinates
        st.success(f"**Detected dot center:** (x = {center_x}, y = {center_y})")
    else:
        st.error("No dot detected. Please try another image.")