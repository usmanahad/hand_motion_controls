import serial
import time
from pynput.keyboard import Key, Controller
import sys

class ESP32GameController:
    def __init__(self, port='/dev/cu.usbserial-0001', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.keyboard = Controller()
        self.ser = None

        # Key mapping (customize for different games)
        self.key_map = {
            'FORWARD': Key.up,      # or 'w' for WASD games
            'BACKWARD': Key.down,   # or 's'
            'LEFT': Key.left,       # or 'a'
            'RIGHT': Key.right,     # or 'd'
            'SHOOT': Key.space      # or Key.ctrl
        }

        # Track pressed keys
        self.pressed_keys = set()

    def connect(self):
        """Connect to ESP32"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"✓ Connected to {self.port}")

            # Wait for READY signal
            while True:
                line = self.ser.readline().decode('utf-8').strip()
                if line == "READY":
                    print("✓ ESP32 ready!")
                    break
                elif line == "INIT_COMPLETE":
                    print("✓ Camera initialized!")
                    break
                time.sleep(0.1)

            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def parse_command(self, command):
        """Parse ESP32 command and trigger keyboard event"""
        try:
            action, state = command.split(':')

            if action not in self.key_map:
                return

            key = self.key_map[action]

            if state == 'PRESS':
                if key not in self.pressed_keys:
                    self.keyboard.press(key)
                    self.pressed_keys.add(key)
                    print(f"🎮 PRESS: {action}")

            elif state == 'RELEASE':
                if key in self.pressed_keys:
                    self.keyboard.release(key)
                    self.pressed_keys.remove(key)
                    print(f"🎮 RELEASE: {action}")

        except Exception as e:
            print(f"Parse error: {e}")

    def run(self):
        """Main loop"""
        if not self.connect():
            return

        print("\n" + "="*60)
        print("ESP32 UNIVERSAL GAME CONTROLLER")
        print("="*60)
        print("\n🎮 Control Mapping:")
        for action, key in self.key_map.items():
            print(f"  {action:12} → {key}")
        print("\n✋ Wear your white glove and start playing!")
        print("Press Ctrl+C to exit\n")
        print("="*60 + "\n")

        try:
            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line and ':' in line:
                        self.parse_command(line)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping controller...")
            self.cleanup()

        except Exception as e:
            print(f"\n✗ Error: {e}")
            self.cleanup()

    def cleanup(self):
        """Release all keys and close connection"""
        print("Releasing all keys...")
        for key in list(self.pressed_keys):
            self.keyboard.release(key)
        self.pressed_keys.clear()

        if self.ser:
            self.ser.close()

        print("✓ Cleanup complete")


if __name__ == "__main__":
    # Auto-detect port or specify manually
    import serial.tools.list_ports

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found!")
        sys.exit(1)

    print("Available ports:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")

    if len(sys.argv) > 1:
        port_index = int(sys.argv[1])
    else:
        port_index = 0

    controller = ESP32GameController(port=ports[port_index].device)
    controller.run()