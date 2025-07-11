# YOLO Automatic License Plate Detection Model

This project provides a Streamlit web application for automatic license plate detection in images and videos using a YOLO (You Only Look Once) model.

## Features

- **Upload Images or Videos:** Supports common image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`) and video formats (`.mp4`, `.avi`, `.mov`, `.mkv`).
- **Automatic Detection:** Uses a trained YOLO model (`best.pt`) to detect license plates.
- **Visualization:** Draws bounding boxes and confidence scores on detected license plates.
- **Download Results:** Download processed images or videos with detected license plates highlighted.

## Installation


1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Add your YOLO model:**
   - Place your trained YOLO weights file (`best.pt`) in the project directory.

## Usage

1. **Run the Streamlit app:**
   ```sh
   streamlit run yolo_applicaiton.py
   ```

2. **Open the app in your browser:**
   - Upload an image or video.
   - Wait for processing.
   - View and download the output with detected license plates.

## Requirements

- Python 3.7+
- See `requirements.txt` for Python package dependencies.

## File Structure

- `yolo_applicaiton.py` — Main Streamlit application.
- `requirements.txt` — Python dependencies.
- `best.pt` — YOLO model weights (not included; add your own).
- `temp/` — Temporary folder for uploaded and processed files.

