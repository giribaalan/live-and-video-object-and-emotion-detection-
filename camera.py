import cv2
import time
import threading
import os

class ThreadedCamera:
    """
    Acquires frames from a VideoCapture source in a background daemon thread asynchronously.
    Handles automatic reconnection, settings configuration, and stream timeout recovery.
    """
    def __init__(self, source, width=640, height=480, fps=30, brightness=-1, contrast=-1, saturation=-1, exposure=-1, auto_exposure=-1):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        
        # Image controls (-1 means use driver defaults)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.exposure = exposure
        self.auto_exposure = auto_exposure
        
        self.cap = None
        self.frame = None
        self.frame_id = 0
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Stats & Status
        self.status = "disconnected"  # "disconnected", "connecting", "connected"
        self.reconnect_count = 0
        self.capture_fps = 0.0
        self.frames_captured = 0
        self.last_frame_time = 0.0
        self.last_error = ""
        
        # Signals for dynamic properties update
        self.pending_properties_update = False
        self.pending_reopen = False

    def start(self):
        if self.running:
            return self
        
        self.running = True
        self.status = "connecting"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[ThreadedCamera] Background thread started (asynchronous).")
        return self

    def _apply_properties(self):
        if self.cap is None or not self.cap.isOpened():
            return
        
        try:
            # Resolution & Frame Rate
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Brightness, Contrast, Saturation, Exposure, Auto Exposure
            # Scale 0-100 values to standard float 0.0 - 1.0 commonly used by OpenCV
            if self.brightness >= 0:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness / 100.0)
            if self.contrast >= 0:
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast / 100.0)
            if self.saturation >= 0:
                self.cap.set(cv2.CAP_PROP_SATURATION, self.saturation / 100.0)
            if self.exposure >= 0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure / 100.0)
            if self.auto_exposure >= 0:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)
        except Exception as e:
            print(f"[ThreadedCamera] Error applying properties: {e}")

    def update_properties(self, width, height, fps, brightness, contrast, saturation, exposure, auto_exposure):
        with self.lock:
            reopen_required = (self.width != width or self.height != height or self.fps != fps)
            
            self.width = width
            self.height = height
            self.fps = fps
            self.brightness = brightness
            self.contrast = contrast
            self.saturation = saturation
            self.exposure = exposure
            self.auto_exposure = auto_exposure
            
            if reopen_required:
                self.pending_reopen = True
            else:
                self.pending_properties_update = True

    def _run(self):
        last_fps_calc_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            # 1. Handle background connect
            if self.cap is None or not self.cap.isOpened():
                self.status = "connecting"
                try:
                    source_val = int(self.source) if str(self.source).isdigit() else self.source
                except ValueError:
                    source_val = self.source
                
                backend = cv2.CAP_DSHOW if isinstance(source_val, int) and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
                
                print(f"[ThreadedCamera] Opening video source: {source_val} (backend: {backend})")
                cap = cv2.VideoCapture(source_val, backend)
                if not cap.isOpened() and backend != cv2.CAP_ANY:
                    cap.release()
                    cap = cv2.VideoCapture(source_val)
                
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if isinstance(source_val, int):
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    
                    self.cap = cap
                    self._apply_properties()
                    self.status = "connected"
                    self.last_frame_time = time.time()
                    print(f"[ThreadedCamera] Successfully connected to source: {source_val}")
                else:
                    self.status = "disconnected"
                    self.last_error = "Failed to open video source"
                    print(f"[ThreadedCamera] Connection failed. Retrying in 3 seconds...")
                    self.reconnect_count += 1
                    # Graceful sleep to exit quickly if self.running is set to False
                    for _ in range(30):
                        if not self.running:
                            break
                        time.sleep(0.1)
                    continue

            # 2. Check for dynamic settings update signals
            if self.pending_reopen:
                print("[ThreadedCamera] Reopening video source to apply resolution/FPS changes...")
                self.cap.release()
                self.cap = None
                self.pending_reopen = False
                continue
                
            if self.pending_properties_update:
                self._apply_properties()
                self.pending_properties_update = False

            # 3. Fetch frame
            ret, frame = self.cap.read()
            if ret and frame is not None:
                curr_time = time.time()
                with self.lock:
                    self.ret = True
                    self.frame = frame
                    self.frame_id += 1
                    self.frames_captured += 1
                    self.last_frame_time = curr_time
                
                # Capture FPS calculation
                fps_frame_count += 1
                time_elapsed = curr_time - last_fps_calc_time
                if time_elapsed >= 1.0:
                    self.capture_fps = round(fps_frame_count / time_elapsed, 1)
                    fps_frame_count = 0
                    last_fps_calc_time = curr_time
            else:
                # Frame read failed. Is it a video file that reached the end?
                is_file = isinstance(self.source, str) and not self.source.isdigit() and os.path.exists(self.source)
                if is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                else:
                    print("[ThreadedCamera] Frame read failed (disconnected). Reconnection scheduled.")
                    self.ret = False
                    self.status = "disconnected"
                    self.last_error = "Frame read failed"
                    self.cap.release()
                    self.cap = None
                    self.reconnect_count += 1
                    for _ in range(30):
                        if not self.running:
                            break
                        time.sleep(0.1)
                    continue

            # 4. Check for stream frozen timeout (for live cams, e.g. 5s with no frame)
            if self.status == "connected" and isinstance(self.source, int):
                if time.time() - self.last_frame_time > 5.0:
                    print("[ThreadedCamera] Camera response timeout. Reconnecting...")
                    self.status = "disconnected"
                    self.last_error = "Camera response timeout"
                    self.cap.release()
                    self.cap = None
                    self.reconnect_count += 1
                    continue

            # Brief sleep to release GIL and prevent CPU spinning
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def read_latest(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy(), self.frame_id
            return self.ret, None, self.frame_id

    def isOpened(self):
        # Compatibility helper: returns True if cap exists and is opened
        return self.cap is not None and self.cap.isOpened()

    def get_properties(self):
        with self.lock:
            props = {
                "source": str(self.source),
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "brightness": self.brightness,
                "contrast": self.contrast,
                "saturation": self.saturation,
                "exposure": self.exposure,
                "auto_exposure": self.auto_exposure,
                "capture_fps": self.capture_fps,
                "reconnect_count": self.reconnect_count,
                "status": self.status,
                "last_error": self.last_error,
                "frames_captured": self.frames_captured
            }
            if self.cap is not None and self.cap.isOpened():
                try:
                    props["actual_width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    props["actual_height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    props["actual_fps"] = float(self.cap.get(cv2.CAP_PROP_FPS))
                except Exception:
                    pass
            return props

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status = "disconnected"
        print("[ThreadedCamera] Released.")
