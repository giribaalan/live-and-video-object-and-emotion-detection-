import os
import cv2
import time
import json
import csv
import io
import threading
import requests
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename
from detector import VisionDetectorPipeline
from camera import ThreadedCamera

app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads", "images")
VIDEO_UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads", "videos")

# Global Detector and Camera state
pipeline = VisionDetectorPipeline(model_name="yolov8n.pt", conf_threshold=0.35)
camera_source = 0  # Default webcam index or video file path
camera_width = 640
camera_height = 480
camera_fps = 30
camera_brightness = -1.0
camera_contrast = -1.0
camera_saturation = -1.0
camera_exposure = -1.0
camera_auto_exposure = -1
threaded_camera = None
latest_metadata = {}
is_streaming = False
# One worker owns inference and encoding; all browser clients consume its latest
# result. This keeps CPU/GPU work constant even when the dashboard is opened twice.
latest_frame_bytes = None
latest_annotated_frame = None
stream_version = 0
stream_lock = threading.Condition()
processor_thread = None
processor_running = False
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Persistent settings config path
CONFIG_FILE = "config.json"
webhook_url = ""

# Video Recording state
is_recording = False
video_writer = None
recording_filename = ""
recording_start_time = 0

def load_config():
    global camera_source, webhook_url
    global camera_width, camera_height, camera_fps
    global camera_brightness, camera_contrast, camera_saturation, camera_exposure, camera_auto_exposure
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                camera_source = config.get("camera_source", 0)
                # An uploaded video is a temporary detection job.  Never resume
                # it automatically when the dashboard is opened again.
                if _is_uploaded_video_source(camera_source):
                    camera_source = 0
                    save_config({"camera_source": camera_source})
                webhook_url = config.get("webhook_url", "")
                
                # Load camera settings
                camera_width = int(config.get("camera_width", 640))
                camera_height = int(config.get("camera_height", 480))
                camera_fps = int(config.get("camera_fps", 30))
                camera_brightness = float(config.get("camera_brightness", -1.0))
                camera_contrast = float(config.get("camera_contrast", -1.0))
                camera_saturation = float(config.get("camera_saturation", -1.0))
                camera_exposure = float(config.get("camera_exposure", -1.0))
                camera_auto_exposure = int(config.get("camera_auto_exposure", -1))
                
                # Update pipeline settings from saved config
                pipeline.update_settings(config)
                print(f"[Config] Loaded settings from {CONFIG_FILE}")
        except Exception as e:
            print(f"[Config] Error loading config: {e}")

def save_config(new_settings):
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        except Exception:
            pass
    config.update(new_settings)
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"[Config] Error saving config: {e}")

# Webhook Alert Dispatcher
def send_webhook_async(event_message):
    global webhook_url
    if not webhook_url:
        return
        
    def send():
        try:
            payload = {
                "event": event_message,
                "timestamp": time.time(),
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            requests.post(webhook_url, json=payload, timeout=2.0)
        except Exception as e:
            print(f"[Webhook] Dispatch failed: {e}")
            
    threading.Thread(target=send, daemon=True).start()

def get_camera_stream():
    global threaded_camera, camera_source
    global camera_width, camera_height, camera_fps
    global camera_brightness, camera_contrast, camera_saturation, camera_exposure, camera_auto_exposure
    if threaded_camera is None or not threaded_camera.running:
        print(f"[Web Server] Instantiating ThreadedCamera with source: {camera_source}")
        threaded_camera = ThreadedCamera(
            camera_source,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            brightness=camera_brightness,
            contrast=camera_contrast,
            saturation=camera_saturation,
            exposure=camera_exposure,
            auto_exposure=camera_auto_exposure
        )
        threaded_camera.start()
    return threaded_camera

def _is_uploaded_video_source(source):
    """Return whether a source points to this app's temporary video uploads."""
    if not isinstance(source, str):
        return False
    try:
        return os.path.normcase(os.path.abspath(source)).startswith(
            os.path.normcase(os.path.abspath(VIDEO_UPLOAD_DIR)) + os.sep
        )
    except (TypeError, ValueError):
        return False

def _activate_source(source):
    """Release the current source and start detection for the requested source."""
    global threaded_camera, camera_source, is_streaming
    global latest_frame_bytes, latest_annotated_frame, latest_metadata, stream_version
    if threaded_camera is not None:
        threaded_camera.release()
    camera_source = source
    threaded_camera = None
    is_streaming = True
    with stream_lock:
        latest_frame_bytes = None
        latest_annotated_frame = None
        latest_metadata = {}
        stream_version += 1
        stream_lock.notify_all()
    ensure_processor_started()

def _allowed_file(filename, extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions

def _public_upload_path(category, filename):
    """Return a browser-safe URL for an uploaded detection result."""
    return f"/static/uploads/{category}/{filename}"

def _video_is_readable(path):
    """Verify OpenCV can decode a frame before activating an uploaded file."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size)
    finally:
        capture.release()

def _process_stream():
    global latest_metadata, latest_frame_bytes, latest_annotated_frame, stream_version, video_writer
    synthetic_angle, last_processed_event_idx = 0.0, 0
    last_camera_frame_id = -1
    while processor_running:
        camera = get_camera_stream()
        ret, frame, camera_frame_id = camera.read_latest() if camera and camera.isOpened() else (False, None, -1)
        # Do not burn CPU/GPU processing the same camera image repeatedly while
        # the capture thread is waiting for the next exposure.
        if ret and frame is not None and camera_frame_id == last_camera_frame_id:
            time.sleep(0.002)
            continue
        last_camera_frame_id = camera_frame_id
        if not ret or frame is None:
            synthetic_angle += 0.08
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            px = int(320 + np.sin(synthetic_angle) * 180)
            cv2.circle(frame, (px, 190), 30, (200, 180, 150), -1)
            cv2.rectangle(frame, (px - 35, 220), (px + 35, 320), (100, 140, 220), -1)
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
        annotated_frame, metadata = pipeline.process_frame(frame)
        events = pipeline.latest_events
        if len(events) < last_processed_event_idx:
            last_processed_event_idx = 0
        for event in events[last_processed_event_idx:]:
            send_webhook_async(event["message"])
        last_processed_event_idx = len(events)
        if is_recording:
            if video_writer is None:
                h, w = annotated_frame.shape[:2]
                video_writer = cv2.VideoWriter(recording_filename, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))
            if video_writer is not None:
                video_writer.write(annotated_frame)
        ok, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with stream_lock:
                latest_metadata = metadata
                latest_annotated_frame = annotated_frame.copy()
                latest_frame_bytes = buffer.tobytes()
                stream_version += 1
                stream_lock.notify_all()

def ensure_processor_started():
    global processor_thread, processor_running
    processor_running = True
    if processor_thread is None or not processor_thread.is_alive():
        processor_thread = threading.Thread(target=_process_stream, name="vision-processor", daemon=True)
        processor_thread.start()

def generate_frames():
    last_version = -1
    while is_streaming:
        with stream_lock:
            stream_lock.wait_for(lambda: stream_version != last_version, timeout=2.0)
            frame_bytes, last_version = latest_frame_bytes, stream_version
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    # Inject recording states in stats
    stats_data = latest_metadata.copy()
    stats_data["is_recording"] = is_recording
    stats_data["recording_duration"] = int(time.time() - recording_start_time) if is_recording else 0
    stats_data["is_streaming"] = is_streaming
    
    # Inject camera stats
    if threaded_camera is not None:
        stats_data["camera"] = threaded_camera.get_properties()
    else:
        stats_data["camera"] = {
            "status": "disconnected",
            "capture_fps": 0.0,
            "reconnect_count": 0,
            "last_error": "Camera not initialized"
        }
    return jsonify(stats_data)

@app.route('/api/events')
def get_events():
    return jsonify({
        "events": pipeline.latest_events,
        "count": len(pipeline.latest_events)
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global webhook_url
    global camera_width, camera_height, camera_fps
    global camera_brightness, camera_contrast, camera_saturation, camera_exposure, camera_auto_exposure
    if request.method == 'POST':
        data = request.json or {}
        
        # Save config file settings
        save_config(data)
        
        # Update pipeline configuration
        pipeline.update_settings(data)
        
        # Update flask local webhook variable
        if "webhook_url" in data:
            webhook_url = data["webhook_url"]
            
        # Update camera settings variables
        if "camera_width" in data:
            camera_width = int(data["camera_width"])
        if "camera_height" in data:
            camera_height = int(data["camera_height"])
        if "camera_fps" in data:
            camera_fps = int(data["camera_fps"])
        if "camera_brightness" in data:
            camera_brightness = float(data["camera_brightness"])
        if "camera_contrast" in data:
            camera_contrast = float(data["camera_contrast"])
        if "camera_saturation" in data:
            camera_saturation = float(data["camera_saturation"])
        if "camera_exposure" in data:
            camera_exposure = float(data["camera_exposure"])
        if "camera_auto_exposure" in data:
            camera_auto_exposure = int(data["camera_auto_exposure"])
            
        # Dynamically apply settings to camera if running
        if threaded_camera is not None and threaded_camera.running:
            threaded_camera.update_properties(
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                brightness=camera_brightness,
                contrast=camera_contrast,
                saturation=camera_saturation,
                exposure=camera_exposure,
                auto_exposure=camera_auto_exposure
            )
            
        return jsonify({"status": "success", "settings": data})
    else:
        return jsonify({
            "conf_threshold": pipeline.conf_threshold,
            "show_boxes": pipeline.show_boxes,
            "show_motion_trails": pipeline.show_motion_trails,
            "show_emotions": pipeline.show_emotions,
            "show_hud": pipeline.show_hud,
            "show_heatmap": pipeline.show_heatmap,
            "model_name": pipeline.model_name,
            "intrusion_zone_enabled": pipeline.intrusion_zone_enabled,
            "intrusion_zone": pipeline.intrusion_zone,
            "motion_history": pipeline.motion_detector.mog2.getHistory(),
            "motion_threshold": pipeline.motion_detector.mog2.getVarThreshold(),
            "min_contour_area": pipeline.motion_detector.min_contour_area,
            "pixels_per_meter": pipeline.tracker.pixels_per_meter,
            "webhook_url": webhook_url,
            "camera_source": str(camera_source),
            "camera_width": camera_width,
            "camera_height": camera_height,
            "camera_fps": camera_fps,
            "camera_brightness": camera_brightness,
            "camera_contrast": camera_contrast,
            "camera_saturation": camera_saturation,
            "camera_exposure": camera_exposure,
            "camera_auto_exposure": camera_auto_exposure
        })

@app.route('/api/change_source', methods=['POST'])
def change_source():
    global threaded_camera, camera_source
    global camera_width, camera_height, camera_fps
    global camera_brightness, camera_contrast, camera_saturation, camera_exposure, camera_auto_exposure
    data = request.json or {}
    new_source = data.get("source", 0)
    
    if isinstance(new_source, str) and new_source.isdigit():
        new_source = int(new_source)
        
    print(f"[Web Server] Changing camera source to: {new_source}")
    _activate_source(new_source)
    # A user-entered camera/RTSP source is a persistent preference.
    save_config({"camera_source": camera_source})
    
    return jsonify({"status": "success", "source": str(camera_source)})

@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start live detection only after the user explicitly requests it."""
    data = request.json or {}
    source = data.get("source", camera_source)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    _activate_source(source)
    return jsonify({"status": "success", "source": str(camera_source)})

@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop inference and release the active camera or uploaded video."""
    global threaded_camera, processor_running, is_streaming, is_recording, video_writer
    global latest_frame_bytes, latest_annotated_frame, latest_metadata, stream_version
    is_streaming = False
    processor_running = False
    if threaded_camera is not None:
        threaded_camera.release()
        threaded_camera = None
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    is_recording = False
    with stream_lock:
        latest_frame_bytes = None
        latest_annotated_frame = None
        latest_metadata = {}
        stream_version += 1
        stream_lock.notify_all()
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/api/upload/image', methods=['POST'])
def upload_image_for_detection():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename or not _allowed_file(uploaded.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"status": "error", "message": "Upload a JPG, PNG, BMP, or WebP image."}), 400

    os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)
    filename = f"{int(time.time() * 1000)}_{secure_filename(uploaded.filename)}"
    input_path = os.path.join(IMAGE_UPLOAD_DIR, filename)
    uploaded.save(input_path)
    frame = cv2.imread(input_path)
    if frame is None:
        return jsonify({"status": "error", "message": "The uploaded image could not be read."}), 400

    # The live stream and upload analysis share one YOLO model, so serialize
    # inference instead of allowing two concurrent model calls.
    try:
        with pipeline.inference_lock:
            result = pipeline.model(frame, conf=pipeline.conf_threshold, imgsz=pipeline.inference_imgsz, verbose=False)[0]
    except Exception as exc:
        app.logger.exception("Image detection failed")
        return jsonify({"status": "error", "message": f"Image detection failed: {exc}"}), 500
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = pipeline.model.names[int(box.cls[0])]
        confidence = round(float(box.conf[0]), 2)
        detections.append({"label": label, "confidence": confidence, "rect": [x1, y1, x2, y2]})
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
        cv2.putText(frame, f"{label} {confidence:.0%}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 127), 2)

    # Image uploads used to run only YOLO, which meant the emotion engine was
    # never invoked.  Analyse each detected face and render the result on the
    # same image returned to the browser.
    faces = []
    if pipeline.show_emotions:
        try:
            h_img, w_img = frame.shape[:2]
            for fx, fy, fw, fh, landmarks, face_score in pipeline.emotion_detector.detect_faces(frame):
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(w_img - fx, fw), min(h_img - fy, fh)
                if fw <= 0 or fh <= 0:
                    continue

                # Face landmarks are expressed in full-image coordinates;
                # translate them for analysis of the cropped face.
                local_landmarks = None
                if landmarks is not None:
                    local_landmarks = np.asarray(landmarks, dtype=np.float32) - np.array([fx, fy], dtype=np.float32)
                analysis = pipeline.emotion_detector.analyze_face_advanced(
                    frame[fy:fy + fh, fx:fx + fw], local_landmarks
                )
                emotion = analysis["emotion"]
                confidence = analysis["confidence"]
                faces.append({
                    "emotion": emotion,
                    "confidence": confidence,
                    "rect": [fx, fy, fx + fw, fy + fh],
                    "face_confidence": round(float(face_score), 2),
                })

                color = (0, 230, 118) if emotion == "Happy" else (0, 215, 255) if emotion == "Surprised" else (0, 0, 255) if emotion == "Angry" else (255, 255, 255)
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
                cv2.putText(
                    frame, f"FACE: {emotion.upper()} ({confidence:.0%})",
                    (fx, max(20, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
                )
        except Exception:
            # Object detection should still succeed if an optional facial
            # emotion dependency/model is unavailable.
            app.logger.exception("Emotion analysis failed for uploaded image")
    output_name = f"detected_{filename.rsplit('.', 1)[0]}.jpg"
    output_path = os.path.join(IMAGE_UPLOAD_DIR, output_name)
    if not cv2.imwrite(output_path, frame):
        return jsonify({"status": "error", "message": "Could not save the annotated image."}), 500
    return jsonify({
        "status": "success",
        "file": _public_upload_path("images", output_name),
        "detections": detections,
        "faces": faces,
    })

@app.route('/api/upload/video', methods=['POST'])
def upload_video_source():
    global threaded_camera, camera_source
    global camera_width, camera_height, camera_fps
    global camera_brightness, camera_contrast, camera_saturation, camera_exposure, camera_auto_exposure
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename or not _allowed_file(uploaded.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"status": "error", "message": "Upload an MP4, AVI, MOV, MKV, or WebM video."}), 400
    os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
    filename = f"{int(time.time() * 1000)}_{secure_filename(uploaded.filename)}"
    video_path = os.path.join(VIDEO_UPLOAD_DIR, filename)
    uploaded.save(video_path)
    if not _video_is_readable(video_path):
        return jsonify({
            "status": "error",
            "message": "This video could not be decoded. Convert it to an H.264 MP4 or AVI and try again."
        }), 400
    _activate_source(video_path)
    return jsonify({"status": "success", "source": camera_source})

@app.route('/api/snapshot', methods=['POST'])
def capture_snapshot():
    os.makedirs("static/snapshots", exist_ok=True)
    filename = f"static/snapshots/snapshot_{int(time.time())}.jpg"
    
    # Reuse the processing worker's frame. Calling the pipeline here used to race
    # with live inference and could stall or corrupt the preview.
    with stream_lock:
        frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else None
    
    if frame is None:
        # Fallback synthetic frame if physical webcam is unavailable
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "SNAPSHOT DEMO CAPTURE", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

    cv2.imwrite(filename, frame)
    return jsonify({"status": "success", "file": "/" + filename})

# Video Recording APIs
@app.route('/api/record/start', methods=['POST'])
def start_recording():
    global is_recording, recording_filename, recording_start_time, video_writer
    if is_recording:
        return jsonify({"status": "error", "message": "Already recording"}), 400
        
    os.makedirs("static/recordings", exist_ok=True)
    recording_filename = f"static/recordings/recording_{int(time.time())}.mp4"
    video_writer = None 
    is_recording = True
    recording_start_time = time.time()
    return jsonify({"status": "success", "file": "/" + recording_filename})

@app.route('/api/record/stop', methods=['POST'])
def stop_recording():
    global is_recording, video_writer, recording_filename
    if not is_recording:
        return jsonify({"status": "error", "message": "Not recording"}), 400
        
    is_recording = False
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    return jsonify({"status": "success", "file": "/" + recording_filename})

# Media Library API
@app.route('/api/media')
def list_media():
    media = []
    
    # List Snapshots
    if os.path.exists("static/snapshots"):
        for f in os.listdir("static/snapshots"):
            if f.endswith((".jpg", ".jpeg", ".png")):
                path = f"static/snapshots/{f}"
                media.append({
                    "name": f,
                    "path": "/static/snapshots/" + f,
                    "type": "snapshot",
                    "time": os.path.getmtime(path)
                })
                
    # List Recordings
    if os.path.exists("static/recordings"):
        for f in os.listdir("static/recordings"):
            if f.endswith(".mp4"):
                path = f"static/recordings/{f}"
                media.append({
                    "name": f,
                    "path": "/static/recordings/" + f,
                    "type": "recording",
                    "time": os.path.getmtime(path)
                })
                
    # Sort by creation time (newest first)
    media.sort(key=lambda x: x["time"], reverse=True)
    return jsonify({"media": media})

@app.route('/api/media/delete', methods=['POST'])
def delete_media():
    data = request.json or {}
    file_path = data.get("path", "")
    
    # Security filter: restrict deletion to assets directories
    clean_path = file_path.lstrip('/')
    if not (clean_path.startswith("static/snapshots/") or clean_path.startswith("static/recordings/")):
        return jsonify({"status": "error", "message": "Access denied"}), 403
        
    if os.path.exists(clean_path):
        try:
            os.remove(clean_path)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "File not found"}), 404

# Historical CSV records
@app.route('/api/history')
def get_history():
    records = []
    csv_path = "surveillance_records.csv"
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        except Exception as e:
            print(f"[History] Error reading CSV: {e}")
    # Return reverse chronological order (newest ID first)
    try:
        records.sort(key=lambda x: int(x.get("ID", 0)), reverse=True)
    except Exception:
        pass
    return jsonify({"history": records})

@app.route('/api/export_logs')
def export_logs():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Event Message"])
    for ev in pipeline.latest_events:
        writer.writerow([ev.get("time", ""), ev.get("message", "")])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=detection_events_log.csv"}
    )

def start_server(port=5000, debug=False):
    load_config()
    print(f"Starting Advanced Vision Web Dashboard on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    start_server()
