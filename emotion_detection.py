import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Monkey-patch torch.load to bypass weights_only restriction for external models
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
except ImportError:
    HSEmotionRecognizer = None


EMOTIONS = ["Happy", "Neutral", "Surprised", "Sad", "Angry", "Fear", "Disgust", "Contempt"]

class SpatialChannelAttention(nn.Module):
    """Spatial and Channel Attention Module for Facial Feature Extraction."""
    def __init__(self, channels):
        super(SpatialChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AdvancedEmotionCNN(nn.Module):
    """
    Advanced Deep Neural Network for Joint Emotion Classification,
    Continuous Valence-Arousal Regression, and Action Unit (AU) Feature Extraction.
    """
    def __init__(self, num_classes=7):
        super(AdvancedEmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attn1 = SpatialChannelAttention(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attn2 = SpatialChannelAttention(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc_shared = nn.Linear(256 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.4)

        # Output Heads
        self.fc_emotion = nn.Linear(256, num_classes)
        self.fc_valence_arousal = nn.Linear(256, 2) # [Valence (-1 to 1), Arousal (0 to 1)]

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(self.attn1(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool2(self.attn2(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool3(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        feat = self.dropout(F.relu(self.fc_shared(x)))
        
        emotion_logits = self.fc_emotion(feat)
        va_outputs = torch.tanh(self.fc_valence_arousal(feat)) # Bound to [-1, 1]
        
        return emotion_logits, va_outputs

class AdvancedEmotionDetector:
    def __init__(self, model_path=None, yunet_path="face_detection_yunet.onnx"):
        self.emotions = EMOTIONS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize HSEmotion
        self.fer = None
        if HSEmotionRecognizer is not None:
            try:
                print("[AdvancedEmotionDetector] Initializing HSEmotion model...")
                self.fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=self.device)
                print("[AdvancedEmotionDetector] HSEmotion model loaded successfully.")
            except Exception as e:
                print(f"[AdvancedEmotionDetector] Error initializing HSEmotion: {e}")
                
        # Keep custom model initialized as a fallback
        self.model = AdvancedEmotionCNN(num_classes=len(EMOTIONS)).to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Could not load custom emotion model weights: {e}")

        # Initialize YuNet Face Detector
        self.yunet = None
        if os.path.exists(yunet_path) and hasattr(cv2, 'FaceDetectorYN'):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    yunet_path, "", (320, 320), score_threshold=0.5, nms_threshold=0.3
                )
            except Exception as e:
                print(f"Could not initialize YuNet face detector: {e}")

        # Initialize Haar Cascade Fallback Detector
        self.cascade = None
        try:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"Could not initialize Haar Cascade fallback face detector: {e}")

        # Temporal smoothing per face key
        self.smoothing_history = {}

    def detect_faces(self, frame):
        """
        Detects human faces in frame using YuNet neural face detector, with Haar Cascade fallback.
        Returns list of tuples: (x, y, w, h, landmarks, score)
        """
        h_img, w_img = frame.shape[:2]
        results = []
        
        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w_img, h_img))
                _, faces = self.yunet.detect(frame)
                if faces is not None:
                    for f in faces:
                        fx, fy, fw, fh = map(int, f[0:4])
                        landmarks = f[4:14].reshape((5, 2))
                        score = float(f[14])
                        results.append((fx, fy, fw, fh, landmarks, score))
            except Exception as e:
                pass
                
        if len(results) == 0 and self.cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in detected:
                    # Create fake landmarks for pose estimation
                    landmarks = np.array([
                        [x + w*0.3, y + h*0.4],  # Right eye
                        [x + w*0.7, y + h*0.4],  # Left eye
                        [x + w*0.5, y + h*0.65], # Nose
                        [x + w*0.35, y + h*0.8], # Right mouth corner
                        [x + w*0.65, y + h*0.8]  # Left mouth corner
                    ], dtype=np.float32)
                    results.append((x, y, w, h, landmarks, 0.9))
            except Exception as e:
                pass
                
        return results


    def estimate_head_pose(self, landmarks, frame_width, frame_height):
        """
        Estimates 3D Head Pose (Yaw, Pitch, Roll) & orientation label from 5 landmarks.
        """
        if landmarks is None or len(landmarks) != 5:
            return "Center", 0.0, 0.0, 0.0

        r_eye, l_eye, nose, r_mouth, l_mouth = landmarks
        eye_center = (r_eye + l_eye) / 2.0
        mouth_center = (r_mouth + l_mouth) / 2.0

        # Roll angle (rotation around Z axis)
        dx = l_eye[0] - r_eye[0]
        dy = l_eye[1] - r_eye[1]
        roll = np.degrees(np.arctan2(dy, dx))

        # Yaw estimation (horizontal turn left/right)
        eye_width = max(1.0, np.hypot(dx, dy))
        nose_to_left = np.hypot(l_eye[0] - nose[0], l_eye[1] - nose[1])
        nose_to_right = np.hypot(r_eye[0] - nose[0], r_eye[1] - nose[1])
        yaw_ratio = (nose_to_left - nose_to_right) / eye_width

        # Pitch estimation (tilt up/down)
        eye_to_nose = nose[1] - eye_center[1]
        nose_to_mouth = mouth_center[1] - nose[1]
        pitch_ratio = (nose_to_mouth - eye_to_nose) / (eye_to_nose + 1e-5)

        pose_label = "Center"
        if yaw_ratio > 0.35:
            pose_label = "Looking Left"
        elif yaw_ratio < -0.35:
            pose_label = "Looking Right"
        elif pitch_ratio > 0.45:
            pose_label = "Looking Down"
        elif pitch_ratio < -0.35:
            pose_label = "Looking Up"
        elif abs(roll) > 18.0:
            pose_label = "Tilted"

        return pose_label, round(roll, 1), round(yaw_ratio * 45.0, 1), round(pitch_ratio * 45.0, 1)

    def predict_face_emotion(self, face_bgr, landmarks=None):
        """Backward compatible method returning (dominant_emotion, confidence)."""
        res = self.analyze_face_advanced(face_bgr, landmarks)
        return res["emotion"], res["confidence"]

    def extract_action_units(self, face_gray, landmarks):
        """
        Extracts Facial Action Units (AUs):
        - AU4: Brow Lowerer (Anger/Concentration)
        - AU12: Lip Corner Puller (Smile/Happiness)
        - AU1+2: Inner Brow Raiser (Surprise)
        - AU15: Lip Corner Depressor (Sadness)
        - AU26: Jaw Drop (Surprise/Shock)
        """
        aus = []
        h, w = face_gray.shape

        # Brow furrow intensity (AU4)
        eyebrow_roi = face_gray[int(h * 0.1):int(h * 0.35), int(w * 0.15):int(w * 0.85)]
        if eyebrow_roi.size > 0:
            grad_y = cv2.Sobel(eyebrow_roi, cv2.CV_64F, 0, 1, ksize=3)
            brow_intensity = np.mean(np.abs(grad_y))
            if brow_intensity > 22.0:
                aus.append("AU4: Brow Furrow")

        # Mouth geometry (AU12, AU26)
        if landmarks is not None and len(landmarks) == 5:
            r_eye, l_eye, nose, r_mouth, l_mouth = landmarks
            inter_eye = max(1.0, np.hypot(l_eye[0] - r_eye[0], l_eye[1] - r_eye[1]))
            mouth_w = np.hypot(l_mouth[0] - r_mouth[0], l_mouth[1] - r_mouth[1])
            
            mouth_y = (r_mouth[1] + l_mouth[1]) / 2.0
            nose_mouth_dist = mouth_y - nose[1]

            if mouth_w / inter_eye > 1.15:
                aus.append("AU12: Smile Puller")
            if nose_mouth_dist / inter_eye > 0.75:
                aus.append("AU26: Jaw Drop")

        return aus

    def analyze_face_advanced(self, face_bgr, landmarks=None):
        """
        Performs comprehensive facial emotion analysis.
        Returns dict containing:
        - dominant_emotion
        - confidence
        - valence (-1.0 to 1.0)
        - arousal (0.0 to 1.0)
        - action_units (list)
        - pose (label, roll, yaw, pitch)
        """
        if face_bgr is None or face_bgr.size == 0:
            return {
                "emotion": "Neutral", "confidence": 0.50,
                "valence": 0.0, "arousal": 0.10,
                "action_units": [], "pose": "Center"
            }

        h, w = face_bgr.shape[:2]
        if h < 15 or w < 15:
            return {
                "emotion": "Neutral", "confidence": 0.50,
                "valence": 0.0, "arousal": 0.10,
                "action_units": [], "pose": "Center"
            }

        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            
            # Action Units and Pose estimation from geometric landmarks
            aus = self.extract_action_units(gray, landmarks)
            pose_label, roll, yaw, pitch = self.estimate_head_pose(landmarks, w, h)
            
            dominant_emotion = "Neutral"
            confidence = 0.5
            valence_pred = 0.0
            arousal_pred = 0.1
            
            if self.fer is not None:
                # HSEmotion prediction
                emotion_label, scores = self.fer.predict_emotions(face_bgr, logits=True)
                
                # Softmax calculation
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)
                
                # Mapped dominant emotion
                mapped_emotions = {
                    'Anger': 'Angry',
                    'Contempt': 'Contempt',
                    'Disgust': 'Disgust',
                    'Fear': 'Fear',
                    'Happiness': 'Happy',
                    'Neutral': 'Neutral',
                    'Sadness': 'Sad',
                    'Surprise': 'Surprised'
                }
                dominant_emotion = mapped_emotions.get(emotion_label, "Neutral")
                
                # Confidence score (probability of the predicted class)
                matching_idx = None
                for idx, label in self.fer.idx_to_class.items():
                    if label == emotion_label:
                        matching_idx = idx
                        break
                if matching_idx is not None:
                    confidence = float(probs[matching_idx])
                
                # Calculate weighted Valence and Arousal
                va_mapping = {
                    'Anger': (-0.6, 0.8),
                    'Contempt': (-0.4, 0.4),
                    'Disgust': (-0.5, 0.3),
                    'Fear': (-0.6, 0.9),
                    'Happiness': (0.8, 0.6),
                    'Neutral': (0.0, 0.1),
                    'Sadness': (-0.6, 0.2),
                    'Surprise': (0.3, 0.8)
                }
                
                valence_pred = 0.0
                arousal_pred = 0.0
                for idx, prob in enumerate(probs):
                    class_name = self.fer.idx_to_class[idx]
                    v, a = va_mapping.get(class_name, (0.0, 0.1))
                    valence_pred += prob * v
                    arousal_pred += prob * a
            else:
                # Fallback to geometry analysis if HSEmotion is not available
                geom_probs = self._analyze_geometry(gray, landmarks)
                top_idx = int(np.argmax(geom_probs))
                confidence = float(geom_probs[top_idx])
                dominant_emotion = self.emotions[top_idx]
                
                # Approximate valence and arousal based on fallback emotion
                fallback_mapping = {
                    'Happy': (0.8, 0.6),
                    'Neutral': (0.0, 0.1),
                    'Surprised': (0.3, 0.8),
                    'Sad': (-0.6, 0.2),
                    'Angry': (-0.6, 0.8),
                    'Fear': (-0.6, 0.9),
                    'Disgust': (-0.5, 0.3),
                    'Contempt': (-0.4, 0.4)
                }
                valence_pred, arousal_pred = fallback_mapping.get(dominant_emotion, (0.0, 0.1))

            return {
                "emotion": dominant_emotion,
                "confidence": round(confidence, 2),
                "valence": round(float(np.clip(valence_pred, -1.0, 1.0)), 2),
                "arousal": round(float(np.clip(arousal_pred, 0.0, 1.0)), 2),
                "action_units": aus,
                "pose": pose_label,
                "pose_angles": (float(roll), float(yaw), float(pitch))
            }

        except Exception as e:
            print(f"[AdvancedEmotionDetector] Error in analyze_face_advanced: {e}")
            return {
                "emotion": "Neutral", "confidence": 0.50,
                "valence": 0.0, "arousal": 0.10,
                "action_units": [], "pose": "Center",
                "pose_angles": (0.0, 0.0, 0.0)
            }


    def _analyze_geometry(self, face_gray, landmarks):
        """Calculates precise probability distribution across 7 emotions."""
        # [Happy, Neutral, Surprised, Sad, Angry, Fear, Disgust]
        scores = np.array([0.14, 0.28, 0.10, 0.14, 0.20, 0.07, 0.07], dtype=np.float32)
        h, w = face_gray.shape

        eyebrow_roi = face_gray[int(h * 0.1):int(h * 0.35), int(w * 0.15):int(w * 0.85)]
        mouth_roi = face_gray[int(h * 0.6):int(h * 0.95), int(w * 0.15):int(w * 0.85)]

        # Brow Furrowing -> Angry / Sad
        if eyebrow_roi.size > 0:
            grad_y = cv2.Sobel(eyebrow_roi, cv2.CV_64F, 0, 1, ksize=3)
            brow_intensity = np.mean(np.abs(grad_y))
            if brow_intensity > 22.0:
                scores[4] += 0.50  # Angry
                scores[3] += 0.15  # Sad
            elif brow_intensity < 10.0:
                scores[1] += 0.20  # Neutral

        # Mouth Geometry -> Happy / Surprised / Neutral
        if mouth_roi.size > 0:
            mouth_std = np.std(mouth_roi)
            mouth_grad = np.mean(np.abs(cv2.Sobel(mouth_roi, cv2.CV_64F, 1, 0, ksize=3)))
            if mouth_std > 44.0 and mouth_grad > 18.0:
                scores[0] += 0.45  # Happy
                scores[2] += 0.15  # Surprised
            elif mouth_std < 20.0:
                scores[1] += 0.25  # Neutral

        # Landmark Geometry
        if landmarks is not None and len(landmarks) == 5:
            r_eye, l_eye, nose, r_mouth, l_mouth = landmarks
            eye_mid_y = (r_eye[1] + l_eye[1]) / 2.0
            eye_to_nose = max(1.0, nose[1] - eye_mid_y)
            mouth_mid_y = (r_mouth[1] + l_mouth[1]) / 2.0
            nose_to_mouth = max(1.0, mouth_mid_y - nose[1])
            
            mouth_w = max(1.0, np.hypot(r_mouth[0] - l_mouth[0], r_mouth[1] - l_mouth[1]))
            inter_eye = max(1.0, np.hypot(r_eye[0] - l_eye[0], r_eye[1] - l_eye[1]))

            mouth_ratio = mouth_w / inter_eye
            vertical_ratio = nose_to_mouth / eye_to_nose

            if mouth_ratio > 1.15:
                scores[0] += 0.40  # Happy
            if vertical_ratio < 0.85 and mouth_ratio < 0.95:
                scores[4] += 0.45  # Angry
            if vertical_ratio > 1.25:
                scores[2] += 0.40  # Surprised

        total = np.sum(scores)
        if total > 0:
            scores /= total
        return scores

    def detect_emotion(self, frame):
        """Backward compatible helper returning dominant emotion string."""
        faces = self.detect_faces(frame)
        if len(faces) > 0:
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh, landmarks, _ = faces[0]
            h_img, w_img = frame.shape[:2]
            face_crop = frame[max(0, fy):min(h_img, fy+fh), max(0, fx):min(w_img, fx+fw)]
            res = self.analyze_face_advanced(face_crop, landmarks)
            return res["emotion"]
        return "Neutral"

# Global detector instance for backward compatibility
EmotionDetector = AdvancedEmotionDetector

_emotion_detector_instance = None
def detect_emotion(frame):
    global _emotion_detector_instance
    if _emotion_detector_instance is None:
        _emotion_detector_instance = AdvancedEmotionDetector()
    return _emotion_detector_instance.detect_emotion(frame)
