# ESP32-S3-CAM HAND-MOTION CONTROLS
Made this to play a whole bunch of random games using motion controls via a wireless connection using my ESP32-S3-CAM.
This repo contains the following:
- An arduino code file that takes the live stream of bits from the CAM and detects hand positions (ON ESP32)
- Based off the hand position, it sends some serial monitor output (ON ESP32)
- The Output then goes to a python file that then executes a C file that carries out some keyboard inputs baased off the hand position (ON Laptop/Device)
- The keyboard interrupts can theoretically work with any game as long as the configurations and controls are set-up correctly
- A compatible game that can run on the other core of the ESP32 which was taken from: https://github.com/espressif/esp32-doom/
