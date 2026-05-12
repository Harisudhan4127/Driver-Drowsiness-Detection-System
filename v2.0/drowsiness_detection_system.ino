// =============================================================
//  DRIVER DROWSINESS DETECTION — ESP32-CAM FIRMWARE  v2.0
//  Board  : AI-Thinker ESP32-CAM
//  Stream : MJPEG over HTTP  (multipart/x-mixed-replace)
//  Author : github.com/your-username
//
//  ⚠  SETUP  ─────────────────────────────────────────────────
//  1. Fill WIFI_SSID and WIFI_PASSWORD below.
//  2. Board: "AI Thinker ESP32-CAM"  (esp32 board package)
//  3. Upload speed: 115200  |  CPU: 240 MHz
//  4. After upload open Serial Monitor at 115200 baud
//     to see the assigned IP address.
// =============================================================

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include <WiFi.h>

// ─── WiFi Credentials ────────────────────────────────────────
//  Replace with your network details before uploading.
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// ─── Stream quality (trade-off: quality vs FPS) ──────────────
//  Options: FRAMESIZE_QVGA(320x240), FRAMESIZE_VGA(640x480),
//           FRAMESIZE_SVGA(800x600), FRAMESIZE_XGA(1024x768)
#define FRAME_SIZE FRAMESIZE_VGA
#define JPEG_QUALITY 12 // 0–63 — lower = better quality, less FPS

// ─── AI-Thinker ESP32-CAM Pin Map ───────────────────────────
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define LED_GPIO_NUM 4 // Built-in flash LED

// ─── Globals ─────────────────────────────────────────────────
httpd_handle_t stream_httpd = NULL;
httpd_handle_t info_httpd = NULL;

static const char *PART_BOUNDARY = "frame";
static const char *STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary="
    "frame";
static const char *STREAM_BOUNDARY =
    "\r\n--frame\r\n";
static const char *STREAM_PART =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "X-Timestamp: %lld\r\n\r\n";

// =============================================================
//  MJPEG STREAM HANDLER  (/stream or /)
// =============================================================
static esp_err_t stream_handler(httpd_req_t *req)
{

  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];
  int64_t frame_count = 0;

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK)
    return res;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "60");

  while (true)
  {

    fb = esp_camera_fb_get();
    if (!fb)
    {
      Serial.println("[WARN] Frame capture failed — retrying");
      res = ESP_FAIL;
      break;
    }

    // Boundary
    res = httpd_resp_send_chunk(
        req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));

    if (res == ESP_OK)
    {
      size_t hlen = snprintf(
          part_buf, sizeof(part_buf), STREAM_PART,
          fb->len, esp_timer_get_time());
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }

    if (res == ESP_OK)
    {
      res = httpd_resp_send_chunk(
          req, (const char *)fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK)
      break;

    frame_count++;
    if (frame_count % 100 == 0)
    {
      Serial.printf("[INFO] Streamed %lld frames\n", frame_count);
    }
  }

  return res;
}

// =============================================================
//  INFO HANDLER  (/status)  — returns JSON
// =============================================================
static esp_err_t status_handler(httpd_req_t *req)
{
  char buf[256];
  sensor_t *s = esp_camera_sensor_get();

  snprintf(buf, sizeof(buf),
           "{"
           "\"framesize\":%d,"
           "\"quality\":%d,"
           "\"brightness\":%d,"
           "\"contrast\":%d,"
           "\"saturation\":%d,"
           "\"ip\":\"%s\","
           "\"rssi\":%d"
           "}",
           s->status.framesize,
           s->status.quality,
           s->status.brightness,
           s->status.contrast,
           s->status.saturation,
           WiFi.localIP().toString().c_str(),
           WiFi.RSSI());

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, buf, strlen(buf));
}

// =============================================================
//  START HTTP SERVERS
// =============================================================
void startServers()
{

  // ─── Stream server (port 80) ─────────────────────────────
  httpd_config_t stream_cfg = HTTPD_DEFAULT_CONFIG();
  stream_cfg.server_port = 80;
  stream_cfg.ctrl_port = 32768;
  stream_cfg.max_uri_handlers = 4;

  httpd_uri_t stream_root = {
      .uri = "/",
      .method = HTTP_GET,
      .handler = stream_handler,
      .user_ctx = NULL};
  httpd_uri_t stream_path = {
      .uri = "/stream",
      .method = HTTP_GET,
      .handler = stream_handler,
      .user_ctx = NULL};

  if (httpd_start(&stream_httpd, &stream_cfg) == ESP_OK)
  {
    httpd_register_uri_handler(stream_httpd, &stream_root);
    httpd_register_uri_handler(stream_httpd, &stream_path);
    Serial.println("[OK] Stream server started on port 80");
  }

  // ─── Info server (port 81) ──────────────────────────────
  httpd_config_t info_cfg = HTTPD_DEFAULT_CONFIG();
  info_cfg.server_port = 81;
  info_cfg.ctrl_port = 32769;

  httpd_uri_t status_uri = {
      .uri = "/status",
      .method = HTTP_GET,
      .handler = status_handler,
      .user_ctx = NULL};

  if (httpd_start(&info_httpd, &info_cfg) == ESP_OK)
  {
    httpd_register_uri_handler(info_httpd, &status_uri);
    Serial.println("[OK] Info server started on port 81");
  }
}

// =============================================================
//  CAMERA INIT
// =============================================================
bool initCamera()
{

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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  // XCLK: 20 MHz gives best frame-rate stability
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // PSRAM (if available) — enables larger buffers & higher FPS
  if (psramFound())
  {
    config.frame_size = FRAME_SIZE;
    config.jpeg_quality = JPEG_QUALITY;
    config.fb_count = 2; // double-buffer
    config.grab_mode = CAMERA_GRAB_LATEST;
    Serial.println("[INFO] PSRAM found — using dual frame buffer");
  }
  else
  {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 16;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    Serial.println("[WARN] No PSRAM — using single buffer / lower res");
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
    Serial.printf("[ERR] Camera init failed: 0x%x\n", err);
    return false;
  }

  // Fine-tune sensor
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAME_SIZE);
  s->set_quality(s, JPEG_QUALITY);
  s->set_brightness(s, 1); // -2 to 2
  s->set_contrast(s, 1);   // -2 to 2
  s->set_saturation(s, 0); // -2 to 2
  s->set_sharpness(s, 1);  // 0 to 2
  s->set_whitebal(s, 1);   // auto white balance
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1); // auto exposure
  s->set_aec2(s, 1);
  s->set_gain_ctrl(s, 1); // auto gain
  s->set_lenc(s, 1);      // lens correction

  Serial.println("[OK] Camera initialized");
  return true;
}

// =============================================================
//  WIFI CONNECT
// =============================================================
bool connectWifi()
{
  Serial.printf("\n[WIFI] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30)
  {
    delay(500);
    Serial.print(".");
    attempts++;
    // Flash LED while connecting
    digitalWrite(LED_GPIO_NUM, attempts % 2);
  }
  digitalWrite(LED_GPIO_NUM, LOW);

  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("\n[ERR] WiFi connection failed!");
    return false;
  }

  Serial.println("\n[OK] WiFi connected");
  Serial.printf(
      "  IP      : http://%s\n"
      "  Stream  : http://%s/stream\n"
      "  Status  : http://%s:81/status\n"
      "  RSSI    : %d dBm\n",
      WiFi.localIP().toString().c_str(),
      WiFi.localIP().toString().c_str(),
      WiFi.localIP().toString().c_str(),
      WiFi.RSSI());
  return true;
}

// =============================================================
//  SETUP
// =============================================================
void setup()
{
  Serial.begin(115200);
  Serial.println("\n============================================");
  Serial.println("  Drowsiness Detection — ESP32-CAM  v2.0");
  Serial.println("============================================");

  // Flash LED: off by default
  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);

  if (!initCamera())
  {
    Serial.println("[ERR] Camera failed — halting.");
    while (true)
    {
      digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
      delay(300); // fast blink = error
    }
  }

  if (!connectWifi())
  {
    Serial.println("[ERR] WiFi failed — halting.");
    while (true)
    {
      digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
      delay(800); // slow blink = wifi error
    }
  }

  startServers();

  // Confirmation blink
  for (int i = 0; i < 3; i++)
  {
    digitalWrite(LED_GPIO_NUM, HIGH);
    delay(100);
    digitalWrite(LED_GPIO_NUM, LOW);
    delay(100);
  }

  Serial.println("[READY] Streaming — open app and connect!");
}

// =============================================================
//  LOOP  — watchdog & WiFi recovery
// =============================================================
void loop()
{
  // If WiFi drops, attempt reconnect
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("[WARN] WiFi lost — reconnecting...");
    WiFi.disconnect();
    delay(1000);
    connectWifi();
    if (WiFi.status() == WL_CONNECTED)
    {
      startServers();
    }
  }
  delay(5000);
}
