# PPE Safety Monitor - YOLO26

A Streamlit application for real-time PPE (Personal Protective Equipment) detection using the YOLO26 model.

## Prerequisites

- Python (>=3.8)
- A model trained on PPE data. You can use `best.pt` (your trained model) or `yolo26n.pt` for testing, or replace it with another custom `.pt` file.

```bash
pip install streamlit ultralytics opencv-python-headless
```

## Running the App

1. Place your PPE model file in the project directory (e.g. `best.pt` for your trained model, `yolo26n.pt` for testing, or another `.pt` file).
2. Start the Streamlit server:

```bash
cd "d:\Github repo\YOLOv26-ppe-dts"
streamlit run app.py
```

3. Open your browser to `http://localhost:8501` (Streamlit will display the URL).

## Features

- Upload an image for PPE detection.
- Live webcam feed for continuous monitoring.
- Confidence threshold adjustment.

## Notes

- The YOLO26 model is NMS-free, enabling faster inference on crowded scenes.
- Replace the model path in `app.py` with your custom PPE-trained model if needed.
