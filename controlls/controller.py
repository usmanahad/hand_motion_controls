import cv2
import numpy as np
import socket
import urllib.request
from collections import deque
import time

class HandController:
    def __init__(self, stream_url, esp32_ip, control_port=8888):
        self.stream_url = stream_url
        self.esp32_ip = esp32_ip
        self.control_port = control_port

        # Control state
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        self.shoot = False

        # Hand tracking
        self.hand_positions = deque(maxlen=10)

        # White glove detection (HSV ranges for white)
        # White has low saturation and high value
        self.target_lower = np.array([0, 0, 180], dtype=np.uint8)  # Low saturation, high brightness
        self.target_upper = np.array([180, 60, 255], dtype=np.uint8)  # Any hue, low sat, max brightness

        self.calibrated = False
        self.calibration_samples = []

        # Socket for sending controls to ESP32
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Add brightness adaptation
        self.adaptive_threshold = True

    def calibrate_color(self, frame, roi):
        """Calibrate tracking color from ROI"""
        x, y, w, h = roi
        sample = frame[y:y+h, x:x+w]

        # Convert to HSV
        hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

        # Store samples
        self.calibration_samples.append(hsv_sample)

        # Need at least 3 samples
        if len(self.calibration_samples) < 3:
            return False

        # Combine all samples
        all_samples = np.vstack([s.reshape(-1, 3) for s in self.calibration_samples])

        # Calculate percentiles for robust range
        lower_percentile = np.percentile(all_samples, 10, axis=0)
        upper_percentile = np.percentile(all_samples, 90, axis=0)

        # For white, focus on high V (brightness) and low S (saturation)
        self.target_lower = np.array([
            0,  # Any hue for white
            0,  # Very low saturation
            max(150, int(lower_percentile[2] - 30))  # High brightness
        ], dtype=np.uint8)

        self.target_upper = np.array([
            180,  # Any hue
            min(80, int(upper_percentile[1] + 30)),  # Low-medium saturation
            255  # Maximum brightness
        ], dtype=np.uint8)

        self.calibrated = True
        self.calibration_samples = []
        print(f"✓ Calibrated for WHITE! Lower: {self.target_lower}, Upper: {self.target_upper}")
        return True

    def detect_white_glove(self, frame):
        """Detect white glove"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for white
        mask = cv2.inRange(hsv, self.target_lower, self.target_upper)

        # Additionally, use grayscale brightness threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Combine masks
        mask = cv2.bitwise_and(mask, bright_mask)

        # Aggressive morphological operations to remove noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        # Final smoothing
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        # Re-threshold after blur
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, mask

        # Filter contours by area and shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area for glove
                # Check aspect ratio (hand should be somewhat rectangular)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:  # Reasonable hand proportions
                    valid_contours.append(contour)

        if not valid_contours:
            return None, mask

        # Get largest valid contour (the glove)
        contour = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        # Calculate center using moments
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Get convex hull (outline of hand)
        hull = cv2.convexHull(contour)

        return {
            'center': (cx, cy),
            'contour': contour,
            'hull': hull,
            'bbox': (x, y, w, h),
            'area': area
        }, mask

    def process_hand_position(self, hand_data, frame_width, frame_height):
        """Convert position to game controls with heavy smoothing"""
        if hand_data is None:
            return

        cx, cy = hand_data['center']

        # Store position
        self.hand_positions.append((cx, cy))

        # Need enough samples for stability
        if len(self.hand_positions) < 5:
            return

        # Weighted average (recent positions matter more)
        weights = np.linspace(0.5, 1.0, len(self.hand_positions))
        weights /= weights.sum()

        avg_x = sum(p[0] * w for p, w in zip(self.hand_positions, weights))
        avg_y = sum(p[1] * w for p, w in zip(self.hand_positions, weights))

        # Reset controls
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        self.shoot = False

        # Very large dead zones for maximum stability
        center_x = frame_width / 2
        center_y = frame_height / 2

        # Horizontal (turn) - divide screen into thirds
        left_zone = frame_width * 0.3
        right_zone = frame_width * 0.7

        if avg_x < left_zone:
            self.turn_left = True
        elif avg_x > right_zone:
            self.turn_right = True

        # Vertical (move) - divide screen into thirds
        top_zone = frame_height * 0.3
        bottom_zone = frame_height * 0.7

        if avg_y < top_zone:
            self.move_forward = True
        elif avg_y > bottom_zone:
            self.move_backward = True

        # Shoot: top-left corner
        if avg_x < frame_width * 0.25 and avg_y < frame_height * 0.25:
            self.shoot = True

    def send_controls_to_esp32(self):
        """Send control state to ESP32 via UDP"""
        control_byte = (
            (self.move_forward << 0) |
            (self.move_backward << 1) |
            (self.turn_left << 2) |
            (self.turn_right << 3) |
            (self.shoot << 4)
        )

        try:
            self.sock.sendto(bytes([control_byte]), (self.esp32_ip, self.control_port))
        except:
            pass

    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("DOOM HAND CONTROLLER - WHITE GLOVE TRACKING")
        print("="*60)
        print("\n🧤 Setup:")
        print("1. Wear your WHITE GLOVE")
        print("2. Stand against a DARK background")
        print("3. Good lighting on your hand")
        print("4. (Optional) Press 'c' to calibrate for your specific glove")
        print("\n🎯 Controls (SIMPLE 3x3 GRID):")
        print("┌─────────┬─────────┬─────────┐")
        print("│ SHOOT   │ FORWARD │         │")
        print("├─────────┼─────────┼─────────┤")
        print("│  LEFT   │  IDLE   │  RIGHT  │")
        print("├─────────┼─────────┼─────────┤")
        print("│         │ BACKWARD│         │")
        print("└─────────┴─────────┴─────────┘")
        print("\n⌨️  Keys:")
        print("- 'c': Calibrate for your glove")
        print("- SPACE: Capture calibration sample (3x)")
        print("- 'r': Reset to default white")
        print("- 'q': Quit")
        print("="*60 + "\n")

        # Open camera stream
        stream = urllib.request.urlopen(self.stream_url, timeout=30)
        bytes_data = bytes()

        cv2.namedWindow('Doom Controller - WHITE GLOVE', cv2.WINDOW_NORMAL)
        cv2.namedWindow('White Detection Mask', cv2.WINDOW_NORMAL)

        calibration_mode = False
        last_detection_time = time.time()

        while True:
            chunk = stream.read(1024)
            if not chunk:
                break

            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]

                if len(jpg) > 100:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if frame is not None:
                        h, w = frame.shape[:2]

                        # Calibration UI
                        if calibration_mode:
                            roi_size = 70
                            roi_x = w // 2 - roi_size // 2
                            roi_y = h // 2 - roi_size // 2
                            cv2.rectangle(frame, (roi_x, roi_y),
                                        (roi_x + roi_size, roi_y + roi_size),
                                        (0, 255, 0), 3)

                            samples_needed = 3 - len(self.calibration_samples)
                            cv2.putText(frame, f"Place WHITE GLOVE in box - {samples_needed} left",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, "Press SPACE to capture",
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Detect white glove
                        hand_data, mask = self.detect_white_glove(frame)

                        # Draw 3x3 grid zones
                        third_w = w // 3
                        third_h = h // 3

                        # Vertical lines
                        cv2.line(frame, (third_w, 0), (third_w, h), (100, 100, 100), 2)
                        cv2.line(frame, (2*third_w, 0), (2*third_w, h), (100, 100, 100), 2)
                        # Horizontal lines
                        cv2.line(frame, (0, third_h), (w, third_h), (100, 100, 100), 2)
                        cv2.line(frame, (0, 2*third_h), (w, 2*third_h), (100, 100, 100), 2)

                        # Zone labels
                        cv2.putText(frame, "SHOOT", (10, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frame, "FWD", (third_w + 40, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, "LEFT", (10, third_h + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, "RIGHT", (2*third_w + 10, third_h + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frame, "BACK", (third_w + 30, 2*third_h + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        if hand_data:
                            last_detection_time = time.time()

                            # Draw convex hull (cleaner outline)
                            cv2.drawContours(frame, [hand_data['hull']], -1, (0, 255, 0), 3)

                            # Draw center point
                            cx, cy = hand_data['center']
                            cv2.circle(frame, (cx, cy), 12, (255, 0, 255), -1)
                            cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)

                            # Draw smooth trail
                            if len(self.hand_positions) > 1:
                                pts = np.array(list(self.hand_positions), dtype=np.int32)
                                for i in range(1, len(pts)):
                                    thickness = int(2 + (i / len(pts)) * 3)
                                    cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]),
                                            (0, 255, 255), thickness)

                            # Show detection info
                            cv2.putText(frame, f"Area: {int(hand_data['area'])}",
                                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (255, 255, 255), 1)

                            # Process controls
                            self.process_hand_position(hand_data, w, h)
                        else:
                            # Clear positions if no detection
                            if time.time() - last_detection_time > 0.5:
                                self.hand_positions.clear()

                        # Send controls
                        self.send_controls_to_esp32()

                        # Display status
                        status_y = 90
                        status_color = (0, 255, 0) if self.calibrated else (0, 165, 255)
                        status_text = "CALIBRATED ✓" if self.calibrated else "DEFAULT WHITE"
                        cv2.putText(frame, status_text, (10, status_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                        # Display active controls (large and clear)
                        control_y = 120
                        if self.move_forward:
                            cv2.putText(frame, "▲ FORWARD", (10, control_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            control_y += 40
                        if self.move_backward:
                            cv2.putText(frame, "▼ BACKWARD", (10, control_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                            control_y += 40
                        if self.turn_left:
                            cv2.putText(frame, "◄ LEFT", (10, control_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                            control_y += 40
                        if self.turn_right:
                            cv2.putText(frame, "► RIGHT", (10, control_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            control_y += 40
                        if self.shoot:
                            cv2.putText(frame, "💥 SHOOT!", (10, control_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                        cv2.imshow('Doom Controller - WHITE GLOVE', frame)
                        cv2.imshow('White Detection Mask', mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                calibration_mode = True
                self.calibration_samples = []
                print("📸 Calibration mode ON - capture your white glove")
            elif key == ord(' ') and calibration_mode:
                roi_size = 70
                roi = (w//2 - roi_size//2, h//2 - roi_size//2, roi_size, roi_size)
                if self.calibrate_color(frame, roi):
                    calibration_mode = False
                    print("✓ Calibration complete!")
                else:
                    print(f"✓ Sample {len(self.calibration_samples)}/3 captured")
            elif key == ord('r'):
                # Reset to default white
                self.target_lower = np.array([0, 0, 180], dtype=np.uint8)
                self.target_upper = np.array([180, 60, 255], dtype=np.uint8)
                self.calibrated = False
                self.hand_positions.clear()
                print("🔄 Reset to default WHITE tracking")

        stream.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ESP32_IP = "10.130.1.70"
    STREAM_URL = f"http://{ESP32_IP}/stream"

    controller = HandController(
        stream_url=STREAM_URL,
        esp32_ip=ESP32_IP,
        control_port=8888
    )

    controller.run()