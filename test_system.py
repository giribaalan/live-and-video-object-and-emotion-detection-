import unittest
import numpy as np
import cv2
import os

from object_tracker import CentroidTracker
from motion_detection import MotionDetector
from emotion_detection import EmotionDetector
from detector import VisionDetectorPipeline
from export_zip import create_project_zip
from app import app
85
class TestVisionSystem(unittest.TestCase):
    
    def test_centroid_tracker(self):
        tracker = CentroidTracker()
        rects_frame1 = [((10, 10, 50, 50), "person"), ((100, 100, 150, 150), "car")]
        tracked1 = tracker.update(rects_frame1)
        self.assertEqual(len(tracked1), 2)

        # Move first object to the right
        rects_frame2 = [((30, 10, 70, 50), "person"), ((100, 100, 150, 150), "car")]
        tracked2 = tracker.update(rects_frame2)
        
        # Verify object 1 velocity is positive X and moving East
        obj1 = tracked2[1]
        self.assertGreater(obj1["speed"], 0)
        self.assertEqual(obj1["direction"], "East")
        self.assertEqual(obj1["state"], "MOVING")
        print("✔ Centroid Tracker Test Passed")

    def test_motion_detector(self):
        detector = MotionDetector()
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        boxes1, mask1, motion_index1 = detector.detect_motion(frame1)
        
        # Frame 2 with a bright moving rectangle
        frame2 = frame1.copy()
        cv2.rectangle(frame2, (50, 50), (120, 120), (255, 255, 255), -1)
        boxes2, mask2, motion_index2 = detector.detect_motion(frame2)
        
        self.assertGreater(motion_index2, 0.0)
        print("✔ Motion Detector Test Passed")

    def test_emotion_detector(self):
        detector = EmotionDetector()
        # Synthetic face image
        face_img = np.full((64, 64, 3), 128, dtype=np.uint8)
        emotion, conf = detector.predict_face_emotion(face_img)
        self.assertIn(emotion, ["Happy", "Neutral", "Surprised", "Sad", "Angry", "Fear", "Disgust"])
        self.assertGreater(conf, 0.0)
        print("✔ Emotion Detector Test Passed")

    def test_vision_pipeline(self):
        pipeline = VisionDetectorPipeline(model_name="yolov8n.pt", conf_threshold=0.35)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic circle
        cv2.circle(frame, (320, 240), 40, (0, 255, 0), -1)
        
        annotated, metadata = pipeline.process_frame(frame)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertIn("fps", metadata)
        self.assertIn("motion_index", metadata)
        print("✔ Vision Pipeline Test Passed")

    def test_flask_app(self):
        client = app.test_client()
        res = client.get('/')
        self.assertEqual(res.status_code, 200)

        res_stats = client.get('/api/stats')
        self.assertEqual(res_stats.status_code, 200)

        res_events = client.get('/api/events')
        self.assertEqual(res_events.status_code, 200)

        print("✔ Flask Web App Test Passed")

    def test_zip_exporter(self):
        zip_path = create_project_zip("test_export.zip")
        self.assertTrue(os.path.exists(zip_path))
        self.assertGreater(os.path.getsize(zip_path), 0)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("✔ Zip Exporter Test Passed")

    def test_threaded_camera(self):
        from camera import ThreadedCamera
        cam = ThreadedCamera("test_video.mp4")
        cam.start()
        self.assertTrue(cam.running)
        self.assertEqual(cam.status, "connecting")
        
        props = cam.get_properties()
        self.assertEqual(props["status"], "connecting")
        
        cam.release()
        self.assertFalse(cam.running)
        print("✔ Threaded Camera Test Passed")

if __name__ == '__main__':
    unittest.main()
