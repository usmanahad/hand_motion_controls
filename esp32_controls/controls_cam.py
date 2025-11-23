import cv2
import numpy as np
import urllib.request
from collections import deque
import time
from pynput.keyboard import Key, Controller

class ESP32GameController:
    def __init__(self, stream_url, key_profile='arrows'):
        self.stream_url = stream_url
        self.keyboard = Controller()

        # Control state
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        self.shoot = False

        # Previous state for detecting changes
        self.prev_forward = False
        self.prev_backward = False
        self.prev_left = False
        self.prev_right = False
        self.prev_shoot = False

        # Hand tracking
        self.hand_positions = deque(maxlen=10)

        # White glove detection
        self.target_lower = np.array([0, 0, 180], dtype=np.uint8)
        self.target_upper = np.array([180, 60, 255], dtype=np.uint8)

        self.calibrated = False
        self.calibration_samples = []

        # Key profiles for different games
        self.key_profiles = {
            'arrows': {
                'forward': Key.up,
                'backward': Key.down,
                'left': Key.left,
                'right': Key.right,
                'shoot': Key.space
            },
            'wasd': {
                'forward': 'w',
                'backward': 's',
                'left': 'a',
                'right': 'd',
                'shoot': Key.space
            },
            'dino': {  # Chrome Dinosaur
                'forward': Key.space,  # Jump
                'backward': Key.down,  # Duck
                'left': Key.space,     # Jump
                'right': Key.space,    # Jump
                'shoot': Key.space     # Jump
            },
            'platformer': {
                'forward': Key.up,     # Jump
                'backward': Key.down,
                'left': Key.left,
                'right': Key.right,
                'shoot': 'z'           # Common action button
            }
        }

        self.current_profile = key_profile
        self.key_map = self.key_profiles.get(key_profile, self.key_profiles['arrows'])

        # Track pressed keys
        self.pressed_keys = set()

    def calibrate_color(self, frame, roi):
        """Calibrate tracking color from ROI"""
        x, y, w, h = roi
        sample = frame[y:y+h, x:x+w]

        hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        self.calibration_samples.append(hsv_sample)

        if len(self.calibration_samples) < 3:
            return False

        all_samples = np.vstack([s.reshape(-1, 3) for s in self.calibration_samples])
        lower_percentile = np.percentile(all_samples, 10, axis=0)
        upper_percentile = np.percentile(all_samples, 90, axis=0)

        self.target_lower = np.array([
            0,
            0,
            max(150, int(lower_percentile[2] - 30))
        ], dtype=np.uint8)

        self.target_upper = np.array([
            180,
            min(80, int(upper_percentile[1] + 30)),
            255
        ], dtype=np.uint8)

        self.calibrated = True
        self.calibration_samples = []
        print(f"✓ Calibrated! Lower: {self.target_lower}, Upper: {self.target_upper}")
        return True

    def detect_white_glove(self, frame):
        """Detect white glove"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.target_lower, self.target_upper)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, bright_mask)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, mask

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    valid_contours.append(contour)

        if not valid_contours:
            return None, mask

        contour = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)

        return {
            'center': (cx, cy),
            'contour': contour,
            'hull': hull,
            'bbox': (x, y, w, h),
            'area': area
        }, mask

    def process_hand_position(self, hand_data, frame_width, frame_height):
        """Convert position to game controls"""
        if hand_data is None:
            return

        cx, cy = hand_data['center']
        self.hand_positions.append((cx, cy))

        if len(self.hand_positions) < 5:
            return

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

        # 3x3 grid
        left_zone = frame_width * 0.3
        right_zone = frame_width * 0.7
        top_zone = frame_height * 0.3
        bottom_zone = frame_height * 0.7

        if avg_x < left_zone:
            self.turn_left = True
        elif avg_x > right_zone:
            self.turn_right = True

        if avg_y < top_zone:
            self.move_forward = True
        elif avg_y > bottom_zone:
            self.move_backward = True

        if avg_x < frame_width * 0.25 and avg_y < frame_height * 0.25:
            self.shoot = True

    def update_keyboard(self):
        """Send keyboard events based on control state changes"""
        # Forward
        if self.move_forward != self.prev_forward:
            key = self.key_map['forward']
            if self.move_forward:
                self.keyboard.press(key)
                self.pressed_keys.add(key)
                print("🎮 PRESS: FORWARD")
            else:
                self.keyboard.release(key)
                self.pressed_keys.discard(key)
                print("🎮 RELEASE: FORWARD")
            self.prev_forward = self.move_forward

        # Backward
        if self.move_backward != self.prev_backward:
            key = self.key_map['backward']
            if self.move_backward:
                self.keyboard.press(key)
                self.pressed_keys.add(key)
                print("🎮 PRESS: BACKWARD")
            else:
                self.keyboard.release(key)
                self.pressed_keys.discard(key)
                print("🎮 RELEASE: BACKWARD")
            self.prev_backward = self.move_backward

        # Left
        if self.turn_left != self.prev_left:
            key = self.key_map['left']
            if self.turn_left:
                self.keyboard.press(key)
                self.pressed_keys.add(key)
                print("🎮 PRESS: LEFT")
            else:
                self.keyboard.release(key)
                self.pressed_keys.discard(key)
                print("🎮 RELEASE: LEFT")
            self.prev_left = self.turn_left

        # Right
        if self.turn_right != self.prev_right:
            key = self.key_map['right']
            if self.turn_right:
                self.keyboard.press(key)
                self.pressed_keys.add(key)
                print("🎮 PRESS: RIGHT")
            else:
                self.keyboard.release(key)
                self.pressed_keys.discard(key)
                print("🎮 RELEASE: RIGHT")
            self.prev_right = self.turn_right

        # Shoot
        if self.shoot != self.prev_shoot:
            key = self.key_map['shoot']
            if self.shoot:
                self.keyboard.press(key)
                self.pressed_keys.add(key)
                print("🎮 PRESS: SHOOT")
            else:
                self.keyboard.release(key)
                self.pressed_keys.discard(key)
                print("🎮 RELEASE: SHOOT")
            self.prev_shoot = self.shoot

    def release_all_keys(self):
        """Release all pressed keys"""
        for key in list(self.pressed_keys):
            self.keyboard.release(key)
        self.pressed_keys.clear()

        # Reset state
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        self.shoot = False

    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("ESP32 UNIVERSAL GAME CONTROLLER")
        print("="*60)
        print(f"\n🎮 Current Profile: {self.current_profile.upper()}")
        print("\n🎯 Key Mapping:")
        for action, key in self.key_map.items():
            print(f"  {action.upper():12} → {key}")
        print("\n🧤 Setup:")
        print("1. Wear your WHITE GLOVE")
        print("2. Stand against a DARK background")
        print("3. Good lighting on your hand")
        print("\n🎯 Controls (3x3 GRID):")
        print("┌─────────┬─────────┬─────────┐")
        print("│ SHOOT   │ FORWARD │         │")
        print("├─────────┼─────────┼─────────┤")
        print("│  LEFT   │  IDLE   │  RIGHT  │")
        print("├─────────┼─────────┼─────────┤")
        print("│         │ BACKWARD│         │")
        print("└─────────┴─────────┴─────────┘")
        print("\n⌨️  Keys:")
        print("- '1': Switch to ARROW KEYS profile")
        print("- '2': Switch to WASD profile")
        print("- '3': Switch to DINO (Chrome) profile")
        print("- '4': Switch to PLATFORMER profile")
        print("- 'c': Calibrate for your glove")
        print("- SPACE: Capture calibration sample (3x)")
        print("- 'r': Reset to default white")
        print("- 'q': Quit")
        print("="*60 + "\n")

        # Open camera stream
        stream = urllib.request.urlopen(self.stream_url, timeout=30)
        bytes_data = bytes()

        cv2.namedWindow('ESP32 Game Controller', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection Mask', cv2.WINDOW_NORMAL)

        calibration_mode = False
        last_detection_time = time.time()

        try:
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

                            # Draw 3x3 grid
                            third_w = w // 3
                            third_h = h // 3

                            cv2.line(frame, (third_w, 0), (third_w, h), (100, 100, 100), 2)
                            cv2.line(frame, (2*third_w, 0), (2*third_w, h), (100, 100, 100), 2)
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

                                # Draw convex hull
                                cv2.drawContours(frame, [hand_data['hull']], -1, (0, 255, 0), 3)

                                # Draw center point
                                cx, cy = hand_data['center']
                                cv2.circle(frame, (cx, cy), 12, (255, 0, 255), -1)
                                cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)

                                # Draw trail
                                if len(self.hand_positions) > 1:
                                    pts = np.array(list(self.hand_positions), dtype=np.int32)
                                    for i in range(1, len(pts)):
                                        thickness = int(2 + (i / len(pts)) * 3)
                                        cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]),
                                                (0, 255, 255), thickness)

                                # Process controls
                                self.process_hand_position(hand_data, w, h)
                            else:
                                # Clear positions if no detection
                                if time.time() - last_detection_time > 0.5:
                                    self.hand_positions.clear()
                                    self.release_all_keys()

                            # Update keyboard based on controls
                            self.update_keyboard()

                            # Display status
                            status_y = 90
                            profile_text = f"Profile: {self.current_profile.upper()}"
                            cv2.putText(frame, profile_text, (10, status_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

                            status_color = (0, 255, 0) if self.calibrated else (0, 165, 255)
                            status_text = "CALIBRATED ✓" if self.calibrated else "DEFAULT WHITE"
                            cv2.putText(frame, status_text, (10, status_y + 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                            # Display active controls
                            control_y = 150
                            if self.move_forward:
                                cv2.putText(frame, "▲ FORWARD", (10, control_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                control_y += 30
                            if self.move_backward:
                                cv2.putText(frame, "▼ BACKWARD", (10, control_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                control_y += 30
                            if self.turn_left:
                                cv2.putText(frame, "◄ LEFT", (10, control_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                control_y += 30
                            if self.turn_right:
                                cv2.putText(frame, "► RIGHT", (10, control_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                control_y += 30
                            if self.shoot:
                                cv2.putText(frame, "💥 SHOOT!", (10, control_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                            cv2.imshow('ESP32 Game Controller', frame)
                            cv2.imshow('Detection Mask', mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    calibration_mode = True
                    self.calibration_samples = []
                    print("📸 Calibration mode ON")
                elif key == ord(' ') and calibration_mode:
                    roi_size = 70
                    roi = (w//2 - roi_size//2, h//2 - roi_size//2, roi_size, roi_size)
                    if self.calibrate_color(frame, roi):
                        calibration_mode = False
                        print("✓ Calibration complete!")
                    else:
                        print(f"✓ Sample {len(self.calibration_samples)}/3 captured")
                elif key == ord('r'):
                    self.target_lower = np.array([0, 0, 180], dtype=np.uint8)
                    self.target_upper = np.array([180, 60, 255], dtype=np.uint8)
                    self.calibrated = False
                    self.hand_positions.clear()
                    print("🔄 Reset to default WHITE tracking")
                elif key == ord('1'):
                    self.release_all_keys()
                    self.current_profile = 'arrows'
                    self.key_map = self.key_profiles['arrows']
                    print("🎮 Switched to ARROW KEYS profile")
                elif key == ord('2'):
                    self.release_all_keys()
                    self.current_profile = 'wasd'
                    self.key_map = self.key_profiles['wasd']
                    print("🎮 Switched to WASD profile")
                elif key == ord('3'):
                    self.release_all_keys()
                    self.current_profile = 'dino'
                    self.key_map = self.key_profiles['dino']
                    print("🎮 Switched to DINO profile (Chrome Dinosaur)")
                elif key == ord('4'):
                    self.release_all_keys()
                    self.current_profile = 'platformer'
                    self.key_map = self.key_profiles['platformer']
                    print("🎮 Switched to PLATFORMER profile")

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping controller...")

        finally:
            self.cleanup()
            stream.close()
            cv2.destroyAllWindows()

    def cleanup(self):
        """Release all keys"""
        print("Releasing all keys...")
        self.release_all_keys()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    ESP32_IP = "10.130.1.70"
    STREAM_URL = f"http://{ESP32_IP}/stream"

    # Choose profile: 'arrows', 'wasd', 'dino', 'platformer'
    controller = ESP32GameController(
        stream_url=STREAM_URL,
        key_profile='arrows'  # Change this for different games
    )

    controller.run()