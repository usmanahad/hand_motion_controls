#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "esp_http_server.h"

// Lightweight OpenCV-like functions for ESP32
// We'll implement our own since full OpenCV is too heavy

const char* ssid = "LUMS-Events";
const char* password = "Epple@2025";

WebServer server(80);
httpd_handle_t camera_httpd = NULL;

// Camera pins
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

// HSV Color space structure
struct HSV {
  uint8_t h, s, v;
};

// Convert RGB to HSV
HSV rgb_to_hsv(uint8_t r, uint8_t g, uint8_t b) {
  HSV hsv;
  
  float rf = r / 255.0f;
  float gf = g / 255.0f;
  float bf = b / 255.0f;
  
  float max_val = max(max(rf, gf), bf);
  float min_val = min(min(rf, gf), bf);
  float delta = max_val - min_val;
  
  // Hue calculation
  if (delta == 0) {
    hsv.h = 0;
  } else if (max_val == rf) {
    hsv.h = (uint8_t)(30 * fmod((gf - bf) / delta, 6.0f));
  } else if (max_val == gf) {
    hsv.h = (uint8_t)(30 * ((bf - rf) / delta + 2));
  } else {
    hsv.h = (uint8_t)(30 * ((rf - gf) / delta + 4));
  }
  
  // Saturation calculation
  if (max_val == 0) {
    hsv.s = 0;
  } else {
    hsv.s = (uint8_t)(255 * delta / max_val);
  }
  
  // Value calculation
  hsv.v = (uint8_t)(255 * max_val);
  
  return hsv;
}

// Blob detection structure
struct Blob {
  int cx, cy;      // Center
  int area;        // Area
  bool valid;
};

// OpenCV-like morphological operations
void erode(uint8_t* mask, int width, int height, int kernel_size = 3) {
  uint8_t* temp = (uint8_t*)malloc(width * height);
  if (!temp) return;
  
  memcpy(temp, mask, width * height);
  
  int half_k = kernel_size / 2;
  
  for (int y = half_k; y < height - half_k; y++) {
    for (int x = half_k; x < width - half_k; x++) {
      bool all_white = true;
      
      // Check kernel
      for (int ky = -half_k; ky <= half_k; ky++) {
        for (int kx = -half_k; kx <= half_k; kx++) {
          if (temp[(y + ky) * width + (x + kx)] == 0) {
            all_white = false;
            break;
          }
        }
        if (!all_white) break;
      }
      
      mask[y * width + x] = all_white ? 255 : 0;
    }
  }
  
  free(temp);
}

void dilate(uint8_t* mask, int width, int height, int kernel_size = 3) {
  uint8_t* temp = (uint8_t*)malloc(width * height);
  if (!temp) return;
  
  memcpy(temp, mask, width * height);
  
  int half_k = kernel_size / 2;
  
  for (int y = half_k; y < height - half_k; y++) {
    for (int x = half_k; x < width - half_k; x++) {
      bool any_white = false;
      
      // Check kernel
      for (int ky = -half_k; ky <= half_k; ky++) {
        for (int kx = -half_k; kx <= half_k; kx++) {
          if (temp[(y + ky) * width + (x + kx)] == 255) {
            any_white = true;
            break;
          }
        }
        if (any_white) break;
      }
      
      mask[y * width + x] = any_white ? 255 : 0;
    }
  }
  
  free(temp);
}

// Blob detection (simplified connected components)
Blob find_largest_blob(uint8_t* mask, int width, int height) {
  Blob blob = {0, 0, 0, false};
  
  long sum_x = 0, sum_y = 0, count = 0;
  
  // Simple centroid calculation
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (mask[y * width + x] == 255) {
        sum_x += x;
        sum_y += y;
        count++;
      }
    }
  }
  
  if (count > 500) {  // Minimum area threshold
    blob.cx = sum_x / count;
    blob.cy = sum_y / count;
    blob.area = count;
    blob.valid = true;
  }
  
  return blob;
}

// Hand detection using HSV color space (OpenCV-like)
Blob detect_white_glove_hsv(camera_fb_t* fb) {
  Blob blob = {0, 0, 0, false};
  
  if (!fb || fb->format != PIXFORMAT_RGB565) {
    return blob;
  }
  
  int width = fb->width;
  int height = fb->height;
  uint16_t* pixels = (uint16_t*)fb->buf;
  
  // Allocate mask
  uint8_t* mask = (uint8_t*)malloc(width * height);
  if (!mask) {
    Serial.println("Failed to allocate mask!");
    return blob;
  }
  
  // HSV thresholds for white (adjustable)
  // White in HSV: Low saturation, High value
  uint8_t h_low = 0, h_high = 180;
  uint8_t s_low = 0, s_high = 30;    // Low saturation
  uint8_t v_low = 200, v_high = 255;  // High brightness
  
  // Convert to HSV and create mask
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      uint16_t pixel = pixels[idx];
      
      // Extract RGB
      uint8_t r = ((pixel >> 11) & 0x1F) * 255 / 31;
      uint8_t g = ((pixel >> 5) & 0x3F) * 255 / 63;
      uint8_t b = (pixel & 0x1F) * 255 / 31;
      
      // Convert to HSV
      HSV hsv = rgb_to_hsv(r, g, b);
      
      // Check if in white range
      if (hsv.h >= h_low && hsv.h <= h_high &&
          hsv.s >= s_low && hsv.s <= s_high &&
          hsv.v >= v_low && hsv.v <= v_high) {
        mask[idx] = 255;
      } else {
        mask[idx] = 0;
      }
    }
  }
  
  // Morphological operations to clean up
  erode(mask, width, height, 5);
  dilate(mask, width, height, 5);
  
  // Find largest blob
  blob = find_largest_blob(mask, width, height);
  
  free(mask);
  return blob;
}

// Smoothing buffer
#define BUFFER_SIZE 5
int pos_buffer_x[BUFFER_SIZE] = {0};
int pos_buffer_y[BUFFER_SIZE] = {0};
int buffer_index = 0;
int buffer_count = 0;

bool move_forward = false;
bool move_backward = false;
bool turn_left = false;
bool turn_right = false;
bool shoot = false;

void process_controls(int x, int y, int width, int height) {
  pos_buffer_x[buffer_index] = x;
  pos_buffer_y[buffer_index] = y;
  buffer_index = (buffer_index + 1) % BUFFER_SIZE;
  if (buffer_count < BUFFER_SIZE) buffer_count++;
  
  if (buffer_count < 3) return;
  
  int avg_x = 0, avg_y = 0;
  for (int i = 0; i < buffer_count; i++) {
    avg_x += pos_buffer_x[i];
    avg_y += pos_buffer_y[i];
  }
  avg_x /= buffer_count;
  avg_y /= buffer_count;
  
  bool prev_forward = move_forward;
  bool prev_backward = move_backward;
  bool prev_left = turn_left;
  bool prev_right = turn_right;
  bool prev_shoot = shoot;
  
  move_forward = false;
  move_backward = false;
  turn_left = false;
  turn_right = false;
  shoot = false;
  
  int third_w = width / 3;
  int third_h = height / 3;
  
  if (avg_x < third_w) turn_left = true;
  else if (avg_x > 2 * third_w) turn_right = true;
  
  if (avg_y < third_h) move_forward = true;
  else if (avg_y > 2 * third_h) move_backward = true;
  
  if (avg_x < width / 4 && avg_y < height / 4) shoot = true;
  
  if (move_forward != prev_forward) {
    Serial.println(move_forward ? "FORWARD:PRESS" : "FORWARD:RELEASE");
  }
  if (move_backward != prev_backward) {
    Serial.println(move_backward ? "BACKWARD:PRESS" : "BACKWARD:RELEASE");
  }
  if (turn_left != prev_left) {
    Serial.println(turn_left ? "LEFT:PRESS" : "LEFT:RELEASE");
  }
  if (turn_right != prev_right) {
    Serial.println(turn_right ? "RIGHT:PRESS" : "RIGHT:RELEASE");
  }
  if (shoot != prev_shoot) {
    Serial.println(shoot ? "SHOOT:PRESS" : "SHOOT:RELEASE");
  }
}

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  if (res != ESP_OK) return res;

  sensor_t *s = esp_camera_sensor_get();
  s->set_pixformat(s, PIXFORMAT_JPEG);

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      if (fb->format != PIXFORMAT_JPEG) {
        esp_camera_fb_return(fb);
        res = ESP_FAIL;
      }
    }
    
    if (res == ESP_OK && fb) {
      size_t hlen = snprintf(part_buf, 64, "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
      res = httpd_resp_send_chunk(req, "\r\n--frame\r\n", 12);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }
    
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
    }
    
    if (res != ESP_OK) break;
  }

  s->set_pixformat(s, PIXFORMAT_RGB565);
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &stream_uri);
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n=== ESP32 OPENCV-LIKE HAND CONTROLLER ===\n");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);
  
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✓ WiFi OK");
  Serial.println("✓ IP: " + WiFi.localIP().toString());

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
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;  // 320x240
  config.jpeg_quality = 12;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("✗ Camera failed: 0x%x\n", err);
    while(1) delay(1000);
  }
  
  Serial.println("✓ Camera init");
  
  sensor_t *s = esp_camera_sensor_get();
  s->set_pixformat(s, PIXFORMAT_RGB565);
  Serial.println("✓ RGB565 mode");

  startCameraServer();
  Serial.println("✓ Stream: http://" + WiFi.localIP().toString() + ":81/stream");

  server.on("/", []() {
    String html = "<!DOCTYPE html><html><head><title>ESP32 OpenCV</title>";
    html += "<style>body{background:#000;color:#0f0;font-family:monospace;padding:20px}";
    html += "img{width:100%;border:2px solid #0f0}</style></head><body>";
    html += "<h1 style='text-align:center;color:#0ff'>ESP32 OpenCV Hand Tracking</h1>";
    html += "<img src='http://" + WiFi.localIP().toString() + ":81/stream'>";
    html += "<p>HSV Color Space + Morphological Operations + Blob Detection</p></body></html>";
    server.send(200, "text/html", html);
  });
  
  server.begin();
  Serial.println("✓ Web: http://" + WiFi.localIP().toString());
  Serial.println("\n🧤 Using OpenCV-like algorithms!\n");
}

void loop() {
  static unsigned long last_debug = 0;
  
  server.handleClient();
  
  camera_fb_t* fb = esp_camera_fb_get();
  
  if (!fb) {
    delay(50);
    return;
  }
  
  // Use OpenCV-like HSV detection
  Blob blob = detect_white_glove_hsv(fb);
  
  if (millis() - last_debug > 2000) {
    Serial.println("--- OPENCV DEBUG ---");
    Serial.printf("Blob detected: %s\n", blob.valid ? "YES ✓" : "NO ✗");
    if (blob.valid) {
      Serial.printf("Position: (%d, %d)\n", blob.cx, blob.cy);
      Serial.printf("Area: %d pixels\n", blob.area);
    }
    Serial.println("--------------------\n");
    last_debug = millis();
  }
  
  if (blob.valid) {
    process_controls(blob.cx, blob.cy, fb->width, fb->height);
  } else {
    buffer_count = 0;
    
    if (move_forward || move_backward || turn_left || turn_right || shoot) {
      if (move_forward) Serial.println("FORWARD:RELEASE");
      if (move_backward) Serial.println("BACKWARD:RELEASE");
      if (turn_left) Serial.println("LEFT:RELEASE");
      if (turn_right) Serial.println("RIGHT:RELEASE");
      if (shoot) Serial.println("SHOOT:RELEASE");
      
      move_forward = false;
      move_backward = false;
      turn_left = false;
      turn_right = false;
      shoot = false;
    }
  }
  
  esp_camera_fb_return(fb);
  delay(100);
}