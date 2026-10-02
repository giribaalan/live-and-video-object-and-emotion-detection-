import math
import numpy as np
import time
import os
import csv

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80, pixels_per_meter=None, smoothing_window=5):
        self.next_object_id = 1
        self.objects = {}         # object_id -> centroid (x, y)
        self.disappeared = {}       # object_id -> count of disappeared frames
        self.history = {}           # object_id -> list of past centroids [(x, y), ...]
        self.velocities = {}        # object_id -> (vx, vy, speed_magnitude)
        self.position_samples = {}  # object_id -> [(timestamp, centroid), ...]
        self.pixels_per_meter = float(pixels_per_meter) if pixels_per_meter else None
        self.smoothing_window = max(2, int(smoothing_window))
        self.labels = {}            # object_id -> class label
        self.states = {}            # object_id -> "MOVING" | "STATIC"
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Cumulative tracking stats
        self.recorded_persons = set()
        self.recorded_objects = set()
        self.record_history = {}    # object_id -> stats history dictionary
        self.csv_path = "surveillance_records.csv"
        
        self._initialize_csv()

    def _initialize_csv(self):
        """Creates the surveillance records CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            except Exception:
                pass
            try:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "ID", "Type", "Time_Start", "Time_End", "Duration_Seconds",
                        "Max_Speed", "Avg_Speed", "Final_Direction", "Dominant_State",
                        "Dominant_Emotion", "Avg_Valence", "Avg_Arousal"
                    ])
                print(f"[CentroidTracker] Initialized records CSV: {self.csv_path}")
            except Exception as e:
                print(f"[CentroidTracker] Error creating records CSV: {e}")

    def register(self, centroid, label="object"):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = [centroid]
        self.velocities[self.next_object_id] = (0.0, 0.0, 0.0)
        self.position_samples[self.next_object_id] = [(time.perf_counter(), centroid)]
        self.labels[self.next_object_id] = label
        self.states[self.next_object_id] = "STATIC"

        # Log entry in history database
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.record_history[self.next_object_id] = {
            "id": self.next_object_id,
            "label": label,
            "time_start": timestamp,
            "time_end": timestamp,
            "time_start_epoch": time.time(),
            "speeds": [0.0],
            "directions": ["Static"],
            "states": ["STATIC"],
            "emotions": [],
            "valences": [],
            "arousals": []
        }

        # Track unique counts
        if label.lower() == "person":
            self.recorded_persons.add(self.next_object_id)
        else:
            self.recorded_objects.add(self.next_object_id)

        self.next_object_id += 1

    def log_object_update(self, object_id, label, speed, direction, state, emotion=None, valence=None, arousal=None):
        """Records telemetry update for a tracked object."""
        if object_id not in self.record_history:
            return

        hist = self.record_history[object_id]
        hist["label"] = label
        hist["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        hist["speeds"].append(speed)
        hist["directions"].append(direction)
        hist["states"].append(state)

        # Handle label switching dynamically
        if label.lower() == "person":
            self.recorded_persons.add(object_id)
            self.recorded_objects.discard(object_id)
        else:
            self.recorded_objects.add(object_id)
            self.recorded_persons.discard(object_id)

        if emotion is not None:
            hist["emotions"].append(emotion)
        if valence is not None:
            hist["valences"].append(valence)
        if arousal is not None:
            hist["arousals"].append(arousal)

    def write_to_csv(self, object_id):
        """Summarizes and writes tracking history for a single object to the CSV database."""
        if object_id not in self.record_history:
            return

        hist = self.record_history[object_id]
        
        duration = round(time.time() - hist["time_start_epoch"], 1)
        max_speed = round(max(hist["speeds"]) if hist["speeds"] else 0.0, 1)
        avg_speed = round(sum(hist["speeds"]) / len(hist["speeds"]) if hist["speeds"] else 0.0, 1)
        final_dir = hist["directions"][-1] if hist["directions"] else "Static"

        # Calculate dominant state
        states = hist["states"]
        dom_state = max(set(states), key=states.count) if states else "STATIC"

        # Calculate dominant emotion and averages for persons
        dom_emotion = "N/A"
        avg_valence = 0.0
        avg_arousal = 0.0

        if hist["label"].lower() == "person":
            if hist["emotions"]:
                dom_emotion = max(set(hist["emotions"]), key=hist["emotions"].count)
            avg_valence = round(sum(hist["valences"]) / len(hist["valences"]), 2) if hist["valences"] else 0.0
            avg_arousal = round(sum(hist["arousals"]) / len(hist["arousals"]), 2) if hist["arousals"] else 0.1

        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    hist["id"], hist["label"], hist["time_start"], hist["time_end"],
                    duration, max_speed, avg_speed, final_dir, dom_state,
                    dom_emotion, avg_valence, avg_arousal
                ])
        except Exception as e:
            print(f"[CentroidTracker] Failed writing to CSV log: {e}")

    def save_all_active(self):
        """Writes all active tracking records to CSV on shutdown."""
        active_ids = list(self.record_history.keys())
        print(f"[CentroidTracker] Saving {len(active_ids)} active tracks to database...")
        for obj_id in active_ids:
            self.write_to_csv(obj_id)
            if obj_id in self.record_history:
                del self.record_history[obj_id]

    def deregister(self, object_id):
        # Save tracking data to CSV before deregistration
        self.write_to_csv(object_id)
        
        if object_id in self.record_history:
            del self.record_history[object_id]

        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]
        del self.velocities[object_id]
        del self.position_samples[object_id]
        del self.labels[object_id]
        del self.states[object_id]

    def update(self, rects_with_labels):
        """
        rects_with_labels: list of tuples ((x1, y1, x2, y2), label)
        """
        observation_time = time.perf_counter()
        if len(rects_with_labels) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_tracked_objects()

        input_centroids = np.zeros((len(rects_with_labels), 2), dtype="int")
        input_rects = []
        input_labels = []

        for i, (rect, label) in enumerate(rects_with_labels):
            x1, y1, x2, y2 = rect
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
            input_rects.append(rect)
            input_labels.append(label)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_labels[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix between existing objects and new centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                new_centroid = input_centroids[col]
                prev_centroid = self.objects[object_id]

                # Use elapsed time rather than frames. Inference is deliberately
                # sampled at a variable cadence, so pixels/frame is inaccurate.
                samples = self.position_samples[object_id]
                samples.append((observation_time, new_centroid))
                if len(samples) > self.smoothing_window:
                    samples.pop(0)
                start_time, start_centroid = samples[0]
                elapsed = max(observation_time - start_time, 1e-3)
                vx = float(new_centroid[0] - start_centroid[0]) / elapsed
                vy = float(new_centroid[1] - start_centroid[1]) / elapsed
                speed = math.sqrt(vx * vx + vy * vy)  # pixels per second

                # Update history
                hist = self.history[object_id]
                hist.append(new_centroid)
                if len(hist) > 20:
                    hist.pop(0)

                self.objects[object_id] = new_centroid
                self.velocities[object_id] = (vx, vy, speed)
                self.labels[object_id] = input_labels[col]
                self.disappeared[object_id] = 0
                self.states[object_id] = "MOVING" if speed >= 15.0 else "STATIC"

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_labels[col])

        return self.get_tracked_objects()

    def set_calibration(self, pixels_per_meter):
        """Set image-plane calibration. Use None to report pixels/second only."""
        self.pixels_per_meter = float(pixels_per_meter) if pixels_per_meter else None

    def get_direction(self, vx, vy):
        if math.sqrt(vx * vx + vy * vy) < 15.0:
            return "Static"
        angle = math.atan2(-vy, vx) * 180.0 / math.pi  # -vy because image Y grows downward
        if -22.5 <= angle < 22.5:
            return "East"
        elif 22.5 <= angle < 67.5:
            return "North-East"
        elif 67.5 <= angle < 112.5:
            return "North"
        elif 112.5 <= angle < 157.5:
            return "North-West"
        elif angle >= 157.5 or angle < -157.5:
            return "West"
        elif -157.5 <= angle < -112.5:
            return "South-West"
        elif -112.5 <= angle < -67.5:
            return "South"
        else:
            return "South-East"

    def get_tracked_objects(self):
        tracked = {}
        for obj_id, centroid in self.objects.items():
            vx, vy, speed = self.velocities.get(obj_id, (0.0, 0.0, 0.0))
            speed_mps = speed / self.pixels_per_meter if self.pixels_per_meter else None
            tracked[obj_id] = {
                "id": obj_id,
                "centroid": centroid,
                "label": self.labels.get(obj_id, "object"),
                "state": self.states.get(obj_id, "STATIC"),
                "speed": round(speed, 1),
                "speed_px_s": round(speed, 1),
                "speed_mps": round(speed_mps, 2) if speed_mps is not None else None,
                "speed_kmh": round(speed_mps * 3.6, 1) if speed_mps is not None else None,
                "direction": self.get_direction(vx, vy),
                "history": self.history.get(obj_id, []),
                "velocity": (round(vx, 1), round(vy, 1))
            }
        return tracked
