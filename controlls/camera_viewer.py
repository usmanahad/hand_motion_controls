import cv2
import numpy as np
from collections import deque
import time
import urllib.request
from threading import Thread, Lock

class SmoothCamera:
    def __init__(self, stream_url, buffer_size=5, target_fps=20):
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.frame_buffer = deque(maxlen=buffer_size)
        self.lock = Lock()
        self.running = False

    def fetch_frames(self):
        """Thread to continuously fetch frames from ESP32"""
        while self.running:
            try:
                stream = urllib.request.urlopen(self.stream_url, timeout=30)
                bytes_data = bytes()

                while self.running:
                    chunk = stream.read(1024)
                    if not chunk:
                        break

                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')  # JPEG start
                    b = bytes_data.find(b'\xff\xd9')  # JPEG end

                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]

                        if len(jpg) > 100:  # Valid JPEG check
                            try:
                                frame = cv2.imdecode(
                                    np.frombuffer(jpg, dtype=np.uint8),
                                    cv2.IMREAD_COLOR
                                )

                                if frame is not None and frame.size > 0:
                                    with self.lock:
                                        self.frame_buffer.append(frame)
                            except:
                                pass

                stream.close()

            except Exception as e:
                print(f"Connection error: {e}")
                time.sleep(2)

    def get_frame(self):
        """Get latest frame from buffer"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]  # Just get the latest frame
            return None

    def start(self):
        """Start capturing stream"""
        self.running = True
        thread = Thread(target=self.fetch_frames, daemon=True)
        thread.start()
        print("Stream started...")

        # Wait for first frame
        timeout = 10
        start = time.time()
        while not self.frame_buffer and time.time() - start < timeout:
            time.sleep(0.1)

        if not self.frame_buffer:
            print("Failed to receive frames!")
            return False

        print("Receiving frames!")
        return True

    def stop(self):
        """Stop capturing"""
        self.running = False

    def display_smooth(self):
        """Display stream at constant framerate"""
        if not self.start():
            return

        frame_time = 1.0 / self.target_fps
        fps_history = deque(maxlen=30)

        print(f"Displaying at constant {self.target_fps} FPS")
        print("Press 'q' to quit, 's' to save screenshot")

        window_name = 'ESP32 Camera'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_frame = None

        while True:
            loop_start = time.time()

            frame = self.get_frame()

            if frame is not None:
                last_frame = frame

                # Calculate display FPS
                fps_history.append(1.0 / (time.time() - loop_start + 0.001))
                display_fps = sum(fps_history) / len(fps_history)

                # Show FPS and buffer status
                info = f"Display FPS: {int(display_fps)} | Buffer: {len(self.frame_buffer)}/{self.buffer_size}"
                cv2.putText(frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow(window_name, frame)
            elif last_frame is not None:
                # Keep showing last frame if no new frame available
                cv2.imshow(window_name, last_frame)
            else:
                # Show waiting message
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for frames...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, blank)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame is not None:
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

            # Maintain constant display rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        self.stop()
        cv2.destroyAllWindows()
        print("Stream stopped")

    def save_video(self, output_file, duration=10):
        """Save video at constant framerate"""
        if not self.start():
            return

        time.sleep(1)

        frame = self.get_frame()
        if frame is None:
            print("No frames available!")
            return

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.target_fps, (width, height))

        print(f"Recording {duration} seconds to {output_file}...")

        start_time = time.time()
        frame_count = 0
        frame_time = 1.0 / self.target_fps

        while time.time() - start_time < duration:
            loop_start = time.time()

            frame = self.get_frame()
            if frame is not None:
                out.write(frame)
                frame_count += 1

                elapsed = time.time() - start_time
                print(f"\rRecording: {elapsed:.1f}s / {duration}s | Frames: {frame_count}", end='')

            # Maintain constant recording rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        print()
        out.release()
        self.stop()
        print(f"Video saved! Total frames: {frame_count}")


if __name__ == "__main__":
    ESP32_IP = "10.130.1.70"
    STREAM_URL = f"http://{ESP32_IP}/stream"

    camera = SmoothCamera(
        stream_url=STREAM_URL,
        buffer_size=10000000,      # Buffer to smooth out jitter
        target_fps=20       # Constant display rate
    )

    # Display at constant rate
    camera.display_smooth()

    # Or save video
    # camera.save_video("output.mp4", duration=10)