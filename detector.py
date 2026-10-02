import cv2
import time
import numpy as np
import torch
import threading
from ultralytics import YOLO

from motion_detection import MotionDetector
from emotion_detection import AdvancedEmotionDetector
from object_tracker import CentroidTracker

class VisionDetectorPipeline:
    def __init__(self, model_name="yolov8s.pt", conf_threshold=0.45):
        print(f"Initializing YOLOv8 model '{model_name}'...")
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Fallback to yolov8n.pt due to error loading {model_name}: {e}")
            self.model = YOLO("yolov8n.pt")
            
        self.model_name = model_name
        self.inference_lock = threading.Lock()
        self.conf_threshold = conf_threshold
        
        self.motion_detector = MotionDetector()
        self.emotion_detector = AdvancedEmotionDetector()
        self.tracker = CentroidTracker()

        # Performance counters
        self.prev_frame_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        # Keep rendering every camera frame, but do the neural work at a cadence
        # the machine can sustain.  Reusing the last result is far preferable to
        # letting a slow inference call freeze the entire live preview.
        self.object_detection_interval = 1 if torch.cuda.is_available() else 4
        self.face_detection_interval = 6 if torch.cuda.is_available() else 15
        self.inference_imgsz = 640 if torch.cuda.is_available() else 416
        self._cached_detections = []
        self._cached_tracked_objects = {}
        self._cached_faces = []
        self._cached_face_analytics = []

        # UI Overlay toggles
        self.show_boxes = True
        self.show_motion_trails = True
        self.show_emotions = True
        self.show_hud = True
        self.show_heatmap = False

        # Intrusion Zone Settings
        self.intrusion_zone_enabled = False
        self.intrusion_zone = None  # [x1, y1, x2, y2]

        # Event log
        self.latest_events = []

    def update_settings(self, settings: dict):
        if "conf_threshold" in settings:
            self.conf_threshold = float(settings["conf_threshold"])
        if "show_boxes" in settings:
            self.show_boxes = bool(settings["show_boxes"])
        if "show_motion_trails" in settings:
            self.show_motion_trails = bool(settings["show_motion_trails"])
        if "show_emotions" in settings:
            self.show_emotions = bool(settings["show_emotions"])
        if "show_hud" in settings:
            self.show_hud = bool(settings["show_hud"])
        if "show_heatmap" in settings:
            self.show_heatmap = bool(settings["show_heatmap"])
        if "pixels_per_meter" in settings:
            self.tracker.set_calibration(settings["pixels_per_meter"])

        # Intrusion zone settings
        if "intrusion_zone_enabled" in settings:
            self.intrusion_zone_enabled = bool(settings["intrusion_zone_enabled"])
        if "intrusion_zone" in settings:
            val = settings["intrusion_zone"]
            if isinstance(val, list) and len(val) == 4:
                self.intrusion_zone = [int(v) for v in val]
            else:
                self.intrusion_zone = None

        # Motion detector sensitivity parameters
        if "motion_history" in settings or "motion_threshold" in settings or "min_contour_area" in settings:
            history = int(settings.get("motion_history", self.motion_detector.mog2.getHistory()))
            var_threshold = int(settings.get("motion_threshold", self.motion_detector.mog2.getVarThreshold()))
            min_contour_area = int(settings.get("min_contour_area", self.motion_detector.min_contour_area))
            self.motion_detector = MotionDetector(history=history, var_threshold=var_threshold, min_contour_area=min_contour_area)

        # Dynamic model loading
        if "model_name" in settings:
            model_name = settings["model_name"]
            if model_name != self.model_name:
                self.load_yolo_model(model_name)

    def load_yolo_model(self, model_name):
        print(f"Dynamically loading YOLOv8 model '{model_name}'...")
        try:
            self.model = YOLO(model_name)
            self.model_name = model_name
        except Exception as e:
            print(f"Failed loading {model_name}: {e}. Fallback to yolov8n.pt")
            try:
                self.model = YOLO("yolov8n.pt")
                self.model_name = "yolov8n.pt"
            except Exception:
                pass

    def process_frame(self, frame):
        """
        Main processing method.
        Returns:
            annotated_frame (np.ndarray): BGR image frame with visual overlays.
            metadata (dict): Structured real-time analytics JSON.
        """
        curr_time = time.time()
        time_diff = curr_time - self.prev_frame_time
        if time_diff > 0:
            self.fps = round(1.0 / time_diff, 1)
        self.prev_frame_time = curr_time
        self.frame_count += 1

        h_img, w_img = frame.shape[:2]

        # 1. Motion Sensor Analysis
        motion_boxes, motion_mask, motion_index = self.motion_detector.detect_motion(frame)

        # 2. YOLO Object Detection.  On CPU, running YOLO on every captured
        # frame makes the browser appear stuck.  Cached boxes are drawn over the
        # newest frame while a fresh inference is performed every few frames.
        run_object_inference = not self._cached_detections or self.frame_count % self.object_detection_interval == 1
        if run_object_inference:
            with self.inference_lock:
                results = self.model(frame, conf=self.conf_threshold, imgsz=self.inference_imgsz, verbose=False)[0]
            detected_rects = []
            raw_detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                bw, bh = x2 - x1, y2 - y1
                if label.lower() in ["cat", "dog", "bear"] and conf < 0.65 and (bw < 180 and bh < 180):
                    label = "mouse"
                rect = (x1, y1, x2, y2)
                detected_rects.append((rect, label))
                raw_detections.append({"rect": rect, "label": label, "confidence": round(conf, 2), "class_id": cls_id})
            self._cached_detections = raw_detections
            self._cached_tracked_objects = self.tracker.update(detected_rects)
        else:
            raw_detections = self._cached_detections
        tracked_objects = self._cached_tracked_objects

        # 3. Face detection and emotion classification are both expensive.  Run
        # them together, then reuse the complete analytics result between runs.
        run_face_inference = self.show_emotions and (
            not self._cached_face_analytics or self.frame_count % self.face_detection_interval == 1
        )
        if run_face_inference:
            self._cached_faces = self.emotion_detector.detect_faces(frame)
            face_analytics = []
            for face in self._cached_faces:
                fx, fy, fw, fh, landmarks, fscore = face
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(w_img - fx, fw), min(h_img - fy, fh)
                if fw <= 0 or fh <= 0:
                    continue
                res = self.emotion_detector.analyze_face_advanced(frame[fy:fy+fh, fx:fx+fw], landmarks)
                res["rect"] = [fx, fy, fx + fw, fy + fh]
                res["landmarks"] = landmarks.tolist() if landmarks is not None else []
                face_analytics.append(res)
            self._cached_face_analytics = face_analytics
        else:
            face_analytics = self._cached_face_analytics
        emotion_summary = {}
        for face in face_analytics:
            emotion_summary[face["emotion"]] = emotion_summary.get(face["emotion"], 0) + 1

        # 5. Combine Objects, Motion, & Facial Analytics
        processed_objects = []
        events_this_frame = []
        annotated_frame = frame.copy()

        # Motion Heatmap overlay option
        if self.show_heatmap:
            heatmap = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.75, heatmap, 0.25, 0)

        for det in raw_detections:
            rect = det["rect"]
            x1, y1, x2, y2 = rect
            label = det["label"]
            conf = det["confidence"]

            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

            matched_id = None
            min_dist = 9999.0
            for obj_id, obj in tracked_objects.items():
                tcX, tcY = obj["centroid"]
                dist = np.hypot(cX - tcX, cY - tcY)
                if dist < min_dist and dist < 60:
                    min_dist = dist
                    matched_id = obj_id

            is_roi_moving, roi_motion_ratio = self.motion_detector.check_roi_motion(motion_mask, rect)
            
            speed = tracked_objects[matched_id]["speed"] if (matched_id is not None and matched_id in tracked_objects) else 0.0
            speed_kmh = tracked_objects[matched_id].get("speed_kmh") if (matched_id is not None and matched_id in tracked_objects) else None
            direction = tracked_objects[matched_id]["direction"] if (matched_id is not None and matched_id in tracked_objects) else "Static"
            motion_state = "MOVING" if (is_roi_moving or speed > 2.0) else "STATIC"

            # Associate face analytics to person
            face_info = None
            if label.lower() == "person":
                for fa in face_analytics:
                    f_rect = fa["rect"]
                    if f_rect[0] >= x1 - 30 and f_rect[2] <= x2 + 30 and f_rect[1] >= y1 - 30 and f_rect[3] <= y2 + int((y2-y1)*0.6):
                        face_info = fa
                        break

            is_intruder = False
            if self.intrusion_zone_enabled and self.intrusion_zone:
                zx1, zy1, zx2, zy2 = self.intrusion_zone
                if zx1 <= cX <= zx2 and zy1 <= cY <= zy2:
                    is_intruder = True

            obj_info = {
                "id": matched_id if matched_id is not None else -1,
                "label": label,
                "confidence": conf,
                "rect": [x1, y1, x2, y2],
                "centroid": [cX, cY],
                "state": motion_state,
                "speed": speed,
                "speed_px_s": speed,
                "speed_kmh": speed_kmh,
                "direction": direction,
                "roi_motion_ratio": roi_motion_ratio,
                "face_analytics": face_info,
                "emotion": face_info["emotion"] if face_info else None,
                "emotion_confidence": face_info["confidence"] if face_info else 0.0,
                "intrusion": is_intruder
            }
            processed_objects.append(obj_info)

            # Log telemetry updates to the tracker for database tracking
            if run_object_inference and matched_id is not None:
                emo = face_info["emotion"] if face_info else None
                val = face_info["valence"] if face_info else None
                ar = face_info["arousal"] if face_info else None
                self.tracker.log_object_update(
                    matched_id, label, speed, direction, motion_state,
                    emo, val, ar
                )

            # Event logging
            if run_object_inference and is_intruder:
                events_this_frame.append(f"Security Alert: Zone Intrusion! {label} #{matched_id or ''} detected in restricted area")
            if run_object_inference and motion_state == "MOVING" and speed > 5.0:
                events_this_frame.append(f"Fast Motion: {label} #{matched_id or ''} moving {direction} ({speed} px/s)")
            if run_object_inference and face_info and face_info["emotion"] in ["Angry", "Surprised", "Happy", "Fear"]:
                events_this_frame.append(f"Emotion Alert: Person #{matched_id or ''} expressed {face_info['emotion']} (Valence: {face_info['valence']}, Arousal: {face_info['arousal']})")

            # 6. Render HUD Bounding Boxes & Motion Trails
            if self.show_boxes:
                box_color = (0, 255, 127) if motion_state == "MOVING" else (200, 200, 200)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                if self.show_motion_trails and matched_id is not None and matched_id in tracked_objects:
                    history = tracked_objects[matched_id]["history"]
                    for i in range(1, len(history)):
                        pt1 = history[i - 1]
                        pt2 = history[i]
                        thickness = int(np.sqrt(20 / float(i + 1)) * 2)
                        cv2.line(annotated_frame, pt1, pt2, (0, 230, 255), thickness)

                id_str = f"#{matched_id} " if matched_id is not None else ""
                label_text = f"{id_str}{label.upper()} | {motion_state}"
                if speed > 0.0:
                    label_text += f" ({speed}px/s {direction})"

                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), box_color, -1)
                cv2.putText(annotated_frame, label_text, (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 7. Render Advanced Facial Analytics Overlays
        if self.show_emotions:
            emotion_colors = {
                "Angry": (0, 0, 255),       # Vibrant Red
                "Happy": (0, 230, 118),     # Bright Green
                "Surprised": (0, 215, 255), # Yellow / Gold
                "Sad": (255, 120, 0),      # Deep Orange
                "Neutral": (220, 220, 220), # Light Silver
                "Fear": (255, 0, 255),     # Magenta
                "Disgust": (0, 140, 255),   # Orange
                "Contempt": (180, 100, 255) # Light Purple / Pink
            }

            for fa in face_analytics:
                fx1, fy1, fx2, fy2 = fa["rect"]
                emo = fa["emotion"]
                econf = fa["confidence"]
                valence = fa["valence"]
                arousal = fa["arousal"]
                aus = fa["action_units"]
                pose = fa["pose"]
                color = emotion_colors.get(emo, (0, 255, 200))

                # Draw face bounding box
                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), color, 2)

                # Draw 5-point facial landmarks with glowing connections
                landmarks = fa.get("landmarks", [])
                if len(landmarks) == 5:
                    pts = [tuple(map(int, p)) for p in landmarks]
                    for pt in pts:
                        cv2.circle(annotated_frame, pt, 3, (0, 255, 255), -1)
                    # Connect eyes and mouth
                    cv2.line(annotated_frame, pts[0], pts[1], (0, 255, 255), 1)
                    cv2.line(annotated_frame, pts[3], pts[4], (0, 255, 255), 1)

                # Render Multi-Attribute Facial Analytics Pill
                header_text = f"FACE: {emo.upper()} ({int(econf*100)}%)"
                sub_text = f"Valence: {valence:+.2f} | Arousal: {arousal:.2f} | Pose: {pose}"
                
                (hw, hh), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                (sw, sh), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                card_w = max(hw, sw) + 16
                card_h = hh + sh + 18

                # Analytics card background banner
                cv2.rectangle(annotated_frame, (fx1, fy1 - card_h - 4), (fx1 + card_w, fy1), (15, 15, 25), -1)
                cv2.rectangle(annotated_frame, (fx1, fy1 - card_h - 4), (fx1 + card_w, fy1), color, 1)

                cv2.putText(annotated_frame, header_text, (fx1 + 8, fy1 - card_h + hh + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, sub_text, (fx1 + 8, fy1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 240), 1, cv2.LINE_AA)

                # AU tags below face
                if len(aus) > 0:
                    au_str = " | ".join(aus)
                    (aw, ah), _ = cv2.getTextSize(au_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(annotated_frame, (fx1, fy2), (fx1 + aw + 10, fy2 + ah + 8), (40, 40, 50), -1)
                    cv2.putText(annotated_frame, au_str, (fx1 + 5, fy2 + ah + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1, cv2.LINE_AA)

        # Draw Intrusion Zone overlay
        if self.intrusion_zone_enabled and self.intrusion_zone:
            zx1, zy1, zx2, zy2 = self.intrusion_zone
            # Check if there is an active intrusion this frame
            has_intrusion = any(obj.get("intrusion", False) for obj in processed_objects)
            color = (0, 0, 255) if has_intrusion else (0, 255, 200) # Red if intruded, cyan if secure
            thickness = 2 if not has_intrusion else 3
            
            # Draw semi-transparent rectangle
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, dst=annotated_frame)
            
            # Draw boundary border
            cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), color, thickness)
            
            # Draw label
            zone_label = "RESTRICTED AREA - INTRUSION!" if has_intrusion else "SECURE ZONE"
            cv2.putText(annotated_frame, zone_label, (zx1 + 5, zy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 8. Render Top HUD Bar
        if self.show_hud:
            cv2.rectangle(annotated_frame, (0, 0), (w_img, 45), (15, 15, 25), -1)
            cv2.line(annotated_frame, (0, 45), (w_img, 45), (0, 255, 200), 2)

            status_txt = f"FPS: {self.fps:.1f} | Objects: {len(processed_objects)} | Faces Analyzed: {len(face_analytics)} | Motion: {motion_index}%"
            cv2.putText(annotated_frame, status_txt, (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

            dot_color = (0, 0, 255) if (self.frame_count // 10) % 2 == 0 else (0, 255, 0)
            cv2.circle(annotated_frame, (w_img - 25, 22), 7, dot_color, -1)

        # Store recent events
        for ev in events_this_frame:
            timestamp = time.strftime("%H:%M:%S")
            self.latest_events.append({"time": timestamp, "message": ev})
            if len(self.latest_events) > 50:
                self.latest_events.pop(0)

        metadata = {
            "fps": self.fps,
            "motion_index": motion_index,
            "active_objects_count": len(processed_objects),
            "objects": processed_objects,
            "face_analytics": face_analytics,
            "emotions_summary": emotion_summary,
            "frame_number": self.frame_count,
            "latest_events": self.latest_events[-10:],
            "total_recorded_persons": len(self.tracker.recorded_persons),
            "total_recorded_objects": len(self.tracker.recorded_objects)
        }

        return annotated_frame, metadata
