#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

const char* ssid = "LUMS-Events";
const char* password = "Epple@2025";

WebServer server(80);

// Pin config
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

// Game controls (shared between cores)
volatile bool move_forward = false;
volatile bool move_backward = false;
volatile bool turn_left = false;
volatile bool turn_right = false;
volatile bool shoot = false;

// Smoothing buffer
#define BUFFER_SIZE 5
int pos_buffer_x[BUFFER_SIZE] = {0};
int pos_buffer_y[BUFFER_SIZE] = {0};
int buffer_index = 0;
int buffer_count = 0;

// White glove detection
bool detect_white_glove(camera_fb_t* fb, int* out_x, int* out_y) {
  if (!fb || fb->format != PIXFORMAT_RGB565) return false;
  
  int width = fb->width;
  int height = fb->height;
  uint16_t* pixels = (uint16_t*)fb->buf;
  
  long sum_x = 0, sum_y = 0, count = 0;
  
  // Sample every 4th pixel for speed
  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      uint16_t pixel = pixels[y * width + x];
      
      // Extract RGB from RGB565
      uint8_t r = ((pixel >> 11) & 0x1F) * 255 / 31;
      uint8_t g = ((pixel >> 5) & 0x3F) * 255 / 63;
      uint8_t b = (pixel & 0x1F) * 255 / 31;
      
      // Calculate brightness
      int brightness = (r + g + b) / 3;
      
      // Check color variation
      int max_c = max(max(r, g), b);
      int min_c = min(min(r, g), b);
      int color_diff = max_c - min_c;
      
      // White detection: high brightness, low color variation
      if (brightness > 180 && color_diff < 50) {
        sum_x += x;
        sum_y += y;
        count++;
      }
    }
  }
  
  if (count < 100) return false;  // Minimum white pixels
  
  *out_x = (sum_x / count) * 4;  // Scale back from subsampling
  *out_y = (sum_y / count) * 4;
  
  return true;
}

// Process hand position to controls
void process_controls(int x, int y, int width, int height) {
  // Add to buffer
  pos_buffer_x[buffer_index] = x;
  pos_buffer_y[buffer_index] = y;
  buffer_index = (buffer_index + 1) % BUFFER_SIZE;
  if (buffer_count < BUFFER_SIZE) buffer_count++;
  
  if (buffer_count < 3) return;
  
  // Calculate average
  int avg_x = 0, avg_y = 0;
  for (int i = 0; i < buffer_count; i++) {
    avg_x += pos_buffer_x[i];
    avg_y += pos_buffer_y[i];
  }
  avg_x /= buffer_count;
  avg_y /= buffer_count;
  
  // Reset all controls
  move_forward = false;
  move_backward = false;
  turn_left = false;
  turn_right = false;
  shoot = false;
  
  // 3x3 grid detection
  int third_w = width / 3;
  int third_h = height / 3;
  
  if (avg_x < third_w) turn_left = true;
  else if (avg_x > 2 * third_w) turn_right = true;
  
  if (avg_y < third_h) move_forward = true;
  else if (avg_y > 2 * third_h) move_backward = true;
  
  // Top-left corner = shoot
  if (avg_x < width / 4 && avg_y < height / 4) shoot = true;
}

// Camera + Tracking Task (Core 0)
void cameraTaskCode(void* parameter) {
  Serial.println("Core 0: Camera + Hand Tracking");
  
  unsigned long last_print = 0;
  
  while (true) {
    server.handleClient();
    
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
      int x, y;
      bool detected = detect_white_glove(fb, &x, &y);
      
      if (detected) {
        process_controls(x, y, fb->width, fb->height);
        
        // Debug print every 500ms
        if (millis() - last_print > 500) {
          Serial.printf("Hand at (%d, %d) | ", x, y);
          if (move_forward) Serial.print("FWD ");
          if (move_backward) Serial.print("BACK ");
          if (turn_left) Serial.print("LEFT ");
          if (turn_right) Serial.print("RIGHT ");
          if (shoot) Serial.print("SHOOT");
          Serial.println();
          last_print = millis();
        }
      } else {
        buffer_count = 0;
      }
      
      esp_camera_fb_return(fb);
    }
    
    vTaskDelay(50 / portTICK_PERIOD_MS);  // ~20 FPS
  }
}

// Game Task (Core 1)
void gameTaskCode(void* parameter) {
  Serial.println("Core 1: Game Logic");
  
  while (true) {
    // TODO: Your Doom game logic here
    // Access the control variables:
    // move_forward, move_backward, turn_left, turn_right, shoot
    
    // Example:
    // doom_update(move_forward, move_backward, turn_left, turn_right, shoot);
    // doom_render();
    
    vTaskDelay(16 / portTICK_PERIOD_MS);  // 60 FPS
  }
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  
  Serial.println("\n=== ESP32-S3 Doom with On-Board Hand Tracking ===\n");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  // Camera config - RGB565 for direct pixel access
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;  // Raw RGB for direct access
  config.frame_size = FRAMESIZE_QVGA;      // 320x240
  config.jpeg_quality = 12;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera failed: 0x%x\n", err);
    while(1) delay(1000);
  }

  Serial.println("Camera initialized (RGB565 mode)!");

  server.on("/", []() {
    String html = "<html><body style='font-family:monospace;background:#000;color:#0f0;padding:20px'>";
    html += "<h1>ESP32 DOOM - Hand Tracking</h1>";
    html += "<p>Forward: " + String(move_forward ? "ON" : "OFF") + "</p>";
    html += "<p>Backward: " + String(move_backward ? "ON" : "OFF") + "</p>";
    html += "<p>Left: " + String(turn_left ? "ON" : "OFF") + "</p>";
    html += "<p>Right: " + String(turn_right ? "ON" : "OFF") + "</p>";
    html += "<p>Shoot: " + String(shoot ? "ON" : "OFF") + "</p>";
    html += "<script>setTimeout(()=>location.reload(),1000)</script>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  });
  
  server.begin();
  
  // Create dual-core tasks
  xTaskCreatePinnedToCore(cameraTaskCode, "Camera", 8192, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(gameTaskCode, "Game", 16384, NULL, 1, NULL, 1);
  
  Serial.println("\n=== SYSTEM READY ===");
  Serial.println("Core 0: Camera + Hand Tracking (~20 FPS)");
  Serial.println("Core 1: Doom Game (~60 FPS)");
  Serial.println("==================\n");
}

void loop() {
  // Empty
}