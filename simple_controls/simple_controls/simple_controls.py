#!/usr/bin/env python3
"""
ESP32 Hand Controller with OpenCV
Uses advanced computer vision for hand tracking
"""

import cv2
import numpy as np
import pyautogui
import requests
from urllib.parse import urlparse
import time

# ESP32 camera stream URL
ESP32_IP = "10.130.1.70"  # Replace with your ESP32 IP
STREAM_URL = f"http://{ESP32_IP}/stream"

# Control states
class HandController:
    def __init__(self):
        self.prev_state = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'shoot': False
        }

        # HSV range for white detection (adjust these!)
        self.lower_white = np.array([0, 0, 180])    # Lower bound
        self.upper_white = np.array([180, 30, 255])  # Upper bound

        # Smoothing
        self.positions = []
        self.max_positions = 5

    def detect_white_glove(self, frame):
        """Detect white glove using HSV color space"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for white color
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, mask

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Minimum area threshold
        if area < 500:
            return None, mask

        # Get centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), mask

        return None, mask

    def smooth_position(self, pos):
        """Smooth position using moving average"""
        if pos is None:
            self.positions.clear()
            return None

        self.positions.append(pos)
        if len(self.positions) > self.max_positions:
            self.positions.pop(0)

        if len(self.positions) < 3:
            return None

        avg_x = sum(p[0] for p in self.positions) // len(self.positions)
        avg_y = sum(p[1] for p in self.positions) // len(self.positions)
        return (avg_x, avg_y)

    def process_controls(self, pos, frame_shape):
        """Convert position to game controls"""
        if pos is None:
            # Release all keys
            for key in self.prev_state:
                if self.prev_state[key]:
                    self.release_key(key)
            return

        height, width = frame_shape[:2]
        x, y = pos

        # Divide into 3x3 grid
        third_w = width // 3
        third_h = height // 3

        new_state = {
            'forward': y < third_h,
            'backward': y > 2 * third_h,
            'left': x < third_w,
            'right': x > 2 * third_w,
            'shoot': x < width // 4 and y < height // 4
        }

        # Send key press/release events
        for key, pressed in new_state.items():
            if pressed != self.prev_state[key]:
                if pressed:
                    self.press_key(key)
                else:
                    self.release_key(key)

        self.prev_state = new_state

    def press_key(self, action):
        """Press keyboard key"""
        key_map = {
            'forward': 'w',
            'backward': 's',
            'left': 'a',
            'right': 'd',
            'shoot': 'space'
        }

        key = key_map.get(action)
        if key:
            print(f"🎮 PRESS: {action.upper()} ({key})")
            pyautogui.keyDown(key)

    def release_key(self, action):
        """Release keyboard key"""
        key_map = {
            'forward': 'w',
            'backward': 's',
            'left': 'a',
            'right': 'd',
            'shoot': 'space'
        }

        key = key_map.get(action)
        if key:
            print(f"🎮 RELEASE: {action.upper()}")
            pyautogui.keyUp(key)


def main():
    print("=" * 50)
    print("ESP32 Hand Controller with OpenCV")
    print("=" * 50)
    print(f"\n📹 Connecting to: {STREAM_URL}")

    # Initialize controller
    controller = HandController()

    # Open video stream
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("❌ Failed to open camera stream!")
        print(f"Make sure ESP32 is running at {ESP32_IP}")
        return

    print("✅ Connected to camera!")
    print("\n🎮 Controls:")
    print("  TOP = Forward (W)")
    print("  BOTTOM = Backward (S)")
    print("  LEFT = Turn Left (A)")
    print("  RIGHT = Turn Right (D)")
    print("  TOP-LEFT = Shoot (SPACE)")
    print("\n⚙️ Press 'q' to quit")
    print("⚙️ Press 't' to adjust thresholds")
    print("\n" + "=" * 50 + "\n")

    # Trackbars for tuning
    cv2.namedWindow('Hand Detection')
    cv2.createTrackbar('Lower H', 'Hand Detection', 0, 180, lambda x: None)
    cv2.createTrackbar('Lower S', 'Hand Detection', 0, 255, lambda x: None)
    cv2.createTrackbar('Lower V', 'Hand Detection', 180, 255, lambda x: None)
    cv2.createTrackbar('Upper H', 'Hand Detection', 180, 180, lambda x: None)
    cv2.createTrackbar('Upper S', 'Hand Detection', 30, 255, lambda x: None)
    cv2.createTrackbar('Upper V', 'Hand Detection', 255, 255, lambda x: None)

    fps_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Failed to grab frame")
            time.sleep(0.1)
            continue

        # Get trackbar values
        controller.lower_white[0] = cv2.getTrackbarPos('Lower H', 'Hand Detection')
        controller.lower_white[1] = cv2.getTrackbarPos('Lower S', 'Hand Detection')
        controller.lower_white[2] = cv2.getTrackbarPos('Lower V', 'Hand Detection')
        controller.upper_white[0] = cv2.getTrackbarPos('Upper H', 'Hand Detection')
        controller.upper_white[1] = cv2.getTrackbarPos('Upper S', 'Hand Detection')
        controller.upper_white[2] = cv2.getTrackbarPos('Upper V', 'Hand Detection')

        # Detect hand
        position, mask = controller.detect_white_glove(frame)
        smoothed_pos = controller.smooth_position(position)

        # Process controls
        controller.process_controls(smoothed_pos, frame.shape)

        # Draw visualization
        height, width = frame.shape[:2]

        # Draw grid
        cv2.line(frame, (width//3, 0), (width//3, height), (0, 255, 0), 1)
        cv2.line(frame, (2*width//3, 0), (2*width//3, height), (0, 255, 0), 1)
        cv2.line(frame, (0, height//3), (width, height//3), (0, 255, 0), 1)
        cv2.line(frame, (0, 2*height//3), (width, 2*height//3), (0, 255, 0), 1)

        # Draw labels
        cv2.putText(frame, "FORWARD", (width//2-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "BACKWARD", (width//2-60, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "LEFT", (10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "RIGHT", (width-80, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SHOOT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw hand position
        if smoothed_pos:
            cv2.circle(frame, smoothed_pos, 15, (0, 0, 255), -1)
            cv2.circle(frame, smoothed_pos, 20, (255, 255, 255), 2)
            cv2.putText(frame, f"({smoothed_pos[0]}, {smoothed_pos[1]})",
                       (smoothed_pos[0] + 25, smoothed_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        cv2.putText(frame, f"FPS: {fps}", (10, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frames
        cv2.imshow('Hand Detection', frame)
        cv2.imshow('Mask', mask)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Release all keys
    for action in controller.prev_state:
        if controller.prev_state[action]:
            controller.release_key(action)

    print("\n✅ Stopped")


if __name__ == "__main__":
    main()