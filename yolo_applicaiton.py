import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.title("YOLO Image and Video Processing")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

def predict_and_save_image(path_test_car, output_image_path):
    try:
        if not os.path.exists(path_test_car):
            st.error(f"Input image not found: {path_test_car}")
            return None
        image = cv2.imread(path_test_car)
        if image is None:
            st.error(f"Failed to read image: {path_test_car}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(image_rgb, device='cpu')
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        success = cv2.imwrite(output_image_path, image)
        if not success:
            st.error(f"Failed to save output image: {output_image_path}")
            return None
        if os.path.exists(output_image_path) and os.path.getsize(output_image_path) > 0:
            st.write(f"Image saved successfully at {output_image_path}")
            return output_image_path
        else:
            st.error(f"Output image file is empty or not created: {output_image_path}")
            return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            st.warning("FPS is 0, defaulting to 30")
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            st.error("Error initializing video writer")
            cap.release()
            return None
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            out.write(frame)
            frame_count += 1
        st.write(f"Processed {frame_count} frames")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.write(f"Video saved successfully at {output_path}")
            return output_path
        else:
            st.error("Output video file is empty or not created")
            return None
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def process_media(input_path, output_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

if uploaded_file is not None:
    os.makedirs("temp", exist_ok=True)
    input_path = os.path.join("temp", uploaded_file.name)
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        output_path = os.path.join("temp", f"output_{os.path.splitext(uploaded_file.name)[0]}.mp4")
    else:
        output_path = os.path.join("temp", f"output_{os.path.splitext(uploaded_file.name)[0]}{file_extension}")
    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Processing...")
        result_path = process_media(input_path, output_path)
        if result_path:
            if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                try:
                    video_file = open(result_path, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes, format="video/mp4", start_time=0)
                    st.download_button(
                        label="Download output video",
                        data=video_bytes,
                        file_name=os.path.basename(result_path),
                        mime="video/mp4"
                    )
                    video_file.close()
                except Exception as e:
                    st.error(f"Error displaying video: {e}")
            else:
                try:
                    st.image(result_path)
                    with open(result_path, 'rb') as f:
                        st.download_button(
                            label="Download output image",
                            data=f.read(),
                            file_name=os.path.basename(result_path),
                            mime=f"image/{file_extension.lstrip('.')}"
                        )
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
    except Exception as e:
        st.error(f"Error uploading or processing file: {e}")