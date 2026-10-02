# Advanced Emotion, Motion & Object State Detection System

A state-of-the-art computer vision platform featuring **live camera object detection**, **adaptive background motion sensing**, **multi-object motion state tracking (speed, direction, motion history)**, **facial emotion recognition**, an interactive **glassmorphic Web Dashboard**, and a **Desktop GUI Window**.

---

## 🌟 Key Features

1. **Live Camera Object Detection**:
   - High-precision multi-class detection via **YOLOv8** (`yolov8n.pt`).
   - Classifies objects (person, car, bottle, cell phone, chair, dog, laptop, etc.) with bounding boxes and confidence scores.

2. **Advanced Motion Sensor Engine**:
   - OpenCV **MOG2 Background Subtraction** + Adaptive Running Average + Frame Differencing.
   - Calculates real-time **Frame Motion Index %** and ROI motion density per object.

3. **Multi-Object State & Direction Tracker**:
   - Assigns persistent IDs to objects across frames.
   - Calculates velocity vectors \((v_x, v_y)\), motion speed (px/s), and movement direction (*North, South, East, West, North-West, etc.*).
   - Classifies state: `MOVING` vs `STATIC`.
   - Draws dynamic motion trajectory trails.

4. **PyTorch Facial Emotion Recognition Engine**:
   - Neural classification predicting 7 facial emotions (*Happy, Neutral, Surprised, Sad, Angry, Fear, Disgust*) with confidence scores.
   - Intelligent face ROI cropping from person detections with temporal smoothing.
   - 100% offline PyTorch implementation, Python 3.14 compatible.

5. **Interactive Glassmorphic Web Dashboard**:
   - Live camera stream feed (`/video_feed`).
   - Real-time analytics stats (Active Objects, Motion Level %, Dominant Emotion, System FPS).
   - Controls panel: Sliders for confidence threshold, toggles for bounding boxes, motion trails, face emotions, and motion heatmaps.
   - Live Event Log & optional Web Audio Alerts.
   - One-click **Snapshot Capture** and **CSV Log Export**.

6. **Dual Mode Execution & Packaging**:
   - Web App Mode (`python main.py --mode web`)
   - Desktop OpenCV Mode (`python main.py --mode desktop`)
   - Automated ZIP packager (`export_zip.py`).

---

## 🚀 Quick Start

### 1. Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Web Dashboard (Recommended)

Run via launcher script or command line:

- **Windows Batch**: Double-click `run_web.bat`
- **Command Line**:
  ```bash
  python main.py --mode web --port 5000
  ```

Open your browser at **`http://localhost:5000`**.

### 3. Running in Desktop GUI Mode

- **Windows Batch**: Double-click `run_desktop.bat`
- **Command Line**:
  ```bash
  python main.py --mode desktop --source 0
  ```

#### Desktop Keyboard Shortcuts:
- `Q` or `ESC`: Quit application
- `S`: Save snapshot image to `/snapshots/`
- `B`: Toggle Bounding Boxes
- `E`: Toggle Facial Emotion Badges
- `M`: Toggle Motion Trajectory Trails
- `H`: Toggle Motion Heatmap Overlay

---

## 📦 Export Project as ZIP

To generate the complete distribution archive:

```bash
python export_zip.py
# or
python main.py --export-zip
```

This creates `Advanced_Emotion_Motion_Object_Detection_Project.zip` containing all code, models, web UI, launcher scripts, and test suite.

---

## 🧪 Running Automated Tests

Verify system integrity:

```bash
python test_system.py
```

---

## 📁 Project Structure

```
Advanced_Emotion_Motion_Object_Detection_Project/
├── main.py               # Main CLI launcher (Web & Desktop modes)
├── app.py                # Flask Web Server & REST API endpoints
├── detector.py           # Unified Vision Detection Pipeline
├── motion_detection.py   # MOG2 Motion Sensor & ROI calculator
├── emotion_detection.py  # PyTorch Facial Emotion Classifier
├── object_tracker.py     # Centroid Multi-Object Tracker & Velocity Engine
├── export_zip.py         # Automated ZIP distribution generator
├── test_system.py        # Automated test suite
├── requirements.txt      # Python dependencies
├── run_web.bat           # One-click Windows Web launcher
├── run_desktop.bat       # One-click Windows Desktop launcher
├── templates/
│   └── index.html        # Glassmorphic web dashboard
├── static/
│   ├── css/style.css     # Dark theme glassmorphism CSS
│   └── js/app.js         # Frontend live stats & controls JS
└── README.md             # System documentation
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
|---|---|---|
| `/video_feed` | `GET` | MJPEG Live Video Stream with overlays |
| `/api/stats` | `GET` | Real-time system stats (FPS, Motion %, Objects list, Emotion summary) |
| `/api/events` | `GET` | Timestamped event alerts log |
| `/api/settings` | `GET / POST` | Query or update vision pipeline settings |
| `/api/snapshot` | `POST` | Capture and save full-resolution snapshot image |
| `/api/export_logs` | `GET` | Download CSV event log file |

---

## Custom Object Training

To accurately detect objects specific to your camera, collect and label images
in YOLO format. Each image needs a matching `.txt` label containing
`class_id center_x center_y width height`, normalized from 0 to 1.

```
dataset/
  images/train/   images/val/
  labels/train/   labels/val/
```

Edit `custom_dataset.yaml` to list your classes, then run:

```bash
python train_custom_model.py --data custom_dataset.yaml --epochs 100
```

The trained model is saved as `runs/detect/train/weights/best.pt`. Select that
file in the dashboard. Use images from the real camera, across its lighting,
angles, distances, and backgrounds, and keep a separate validation set.

## Accurate Object Speed

Speed is calculated in **pixels per second** from elapsed time, so it remains
correct when inference cadence varies. Physical speed requires calibration of
the fixed camera scene. Send the measured `pixels_per_meter` setting through
`/api/settings`, for example:

```json
{ "pixels_per_meter": 82.5 }
```

Measure this on the same ground plane where objects move. A single calibration
cannot provide road-speed-grade accuracy for a tilted or wide-angle camera;
that requires perspective calibration (a homography).
