// =============================================================
//  DRIVER DROWSINESS DETECTION — ESP32-CAM FIRMWARE  v3.0
//  Mode    : WIRELESS  —  MJPEG over HTTP / WiFi
//  Board   : AI-Thinker ESP32-CAM
//  Stream  : multipart/x-mixed-replace  (port 80)
//  Status  : JSON endpoint              (port 81)
//
//  ── HOW TO USE ───────────────────────────────────────────────
//  1. Set WIFI_SSID and WIFI_PASSWORD below.
//  2. Arduino IDE → Board: "AI Thinker ESP32-CAM"
//     (install ESP32 board package by Espressif)
//  3. Upload speed : 115200  |  CPU Frequency: 240 MHz
//  4. After upload, open Serial Monitor @ 115200 baud.
//     The board will print its IP address.
//  5. In the Python app, choose:
//       Mode  → Wireless (ESP32-CAM HTTP)
//       IP    → <printed IP>
//       Port  → 80
//       Path  → /stream
//     Then click Connect.
//
//  ── ENDPOINTS ────────────────────────────────────────────────
//  GET http://<IP>/          → MJPEG stream
//  GET http://<IP>/stream    → MJPEG stream (alias)
//  GET http://<IP>:81/status → JSON camera/WiFi info
//  GET http://<IP>:81/led?v=1|0 → Flash LED control
// =============================================================

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include <WiFi.h>

// ─── WiFi Credentials ────────────────────────────────────────
#define WIFI_SSID      "YOUR_WIFI_SSID"
#define WIFI_PASSWORD  "YOUR_WIFI_PASSWORD"

// ─── Stream Quality ──────────────────────────────────────────
//  FRAMESIZE_QVGA  = 320×240   (fastest, ~25 fps on weak WiFi)
//  FRAMESIZE_VGA   = 640×480   (recommended balance)
//  FRAMESIZE_SVGA  = 800×600
//  FRAMESIZE_XGA   = 1024×768  (best quality, ~10 fps)
#define FRAME_SIZE    FRAMESIZE_VGA
#define JPEG_QUALITY  12          // 0–63; lower = better quality

// ─── AI-Thinker ESP32-CAM Pin Map ────────────────────────────
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22
#define LED_GPIO_NUM     4        // Built-in flash LED

// ─── Globals ─────────────────────────────────────────────────
httpd_handle_t stream_httpd = NULL;
httpd_handle_t info_httpd   = NULL;

static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=frame";
static const char* STREAM_BOUNDARY =
    "\r\n--frame\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "X-Timestamp: %lld\r\n\r\n";


// =============================================================
//  MJPEG STREAM HANDLER  (GET /  and  GET /stream)
// =============================================================
static esp_err_t stream_handler(httpd_req_t* req) {
    camera_fb_t* fb    = NULL;
    esp_err_t    res   = ESP_OK;
    char         part_buf[128];
    int64_t      frames = 0;

    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    // CORS — allow browser / Python requests from any origin
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "60");

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[WARN] Frame capture failed — retrying");
            res = ESP_FAIL;
            break;
        }

        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY,
                                    strlen(STREAM_BOUNDARY));

        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, sizeof(part_buf),
                                   STREAM_PART,
                                   fb->len,
                                   esp_timer_get_time());
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }

        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req,
                                        (const char*)fb->buf,
                                        fb->len);
        }

        esp_camera_fb_return(fb);
        fb = NULL;

        if (res != ESP_OK) break;

        if (++frames % 150 == 0) {
            Serial.printf("[INFO] Streamed %lld frames  RSSI: %d dBm\n",
                          frames, WiFi.RSSI());
        }
    }
    return res;
}


// =============================================================
//  STATUS HANDLER  (GET :81/status)  — returns JSON
// =============================================================
static esp_err_t status_handler(httpd_req_t* req) {
    char buf[384];
    sensor_t* s = esp_camera_sensor_get();

    snprintf(buf, sizeof(buf),
        "{"
        "\"firmware\":\"wireless-v3.0\","
        "\"framesize\":%d,"
        "\"quality\":%d,"
        "\"brightness\":%d,"
        "\"contrast\":%d,"
        "\"saturation\":%d,"
        "\"ip\":\"%s\","
        "\"rssi\":%d,"
        "\"uptime_ms\":%lld,"
        "\"free_heap\":%lu"
        "}",
        s->status.framesize,
        s->status.quality,
        s->status.brightness,
        s->status.contrast,
        s->status.saturation,
        WiFi.localIP().toString().c_str(),
        WiFi.RSSI(),
        esp_timer_get_time() / 1000,
        (unsigned long)esp_get_free_heap_size());

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, buf, strlen(buf));
}


// =============================================================
//  LED CONTROL HANDLER  (GET :81/led?v=1 or ?v=0)
// =============================================================
static esp_err_t led_handler(httpd_req_t* req) {
    char query[32] = {0};
    httpd_req_get_url_query_str(req, query, sizeof(query));
    if (strstr(query, "v=1")) {
        digitalWrite(LED_GPIO_NUM, HIGH);
    } else {
        digitalWrite(LED_GPIO_NUM, LOW);
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "OK", 2);
}


// =============================================================
//  START HTTP SERVERS
// =============================================================
void startServers() {
    // ── Stream server  (port 80) ──────────────────────────────
    httpd_config_t scfg      = HTTPD_DEFAULT_CONFIG();
    scfg.server_port          = 80;
    scfg.ctrl_port            = 32768;
    scfg.max_uri_handlers     = 4;
    scfg.stack_size           = 8192;

    httpd_uri_t root_uri   = { "/",       HTTP_GET, stream_handler, NULL };
    httpd_uri_t stream_uri = { "/stream", HTTP_GET, stream_handler, NULL };

    if (httpd_start(&stream_httpd, &scfg) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &root_uri);
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        Serial.println("[OK]  Stream server  → port 80  (/  and  /stream)");
    }

    // ── Info / control server  (port 81) ─────────────────────
    httpd_config_t icfg  = HTTPD_DEFAULT_CONFIG();
    icfg.server_port      = 81;
    icfg.ctrl_port        = 32769;
    icfg.max_uri_handlers = 4;

    httpd_uri_t status_uri = { "/status", HTTP_GET, status_handler, NULL };
    httpd_uri_t led_uri    = { "/led",    HTTP_GET, led_handler,    NULL };

    if (httpd_start(&info_httpd, &icfg) == ESP_OK) {
        httpd_register_uri_handler(info_httpd, &status_uri);
        httpd_register_uri_handler(info_httpd, &led_uri);
        Serial.println("[OK]  Info server    → port 81  (/status  /led)");
    }
}


// =============================================================
//  CAMERA INIT
// =============================================================
bool initCamera() {
    camera_config_t cfg;
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.pin_d0    = Y2_GPIO_NUM;   cfg.pin_d1    = Y3_GPIO_NUM;
    cfg.pin_d2    = Y4_GPIO_NUM;   cfg.pin_d3    = Y5_GPIO_NUM;
    cfg.pin_d4    = Y6_GPIO_NUM;   cfg.pin_d5    = Y7_GPIO_NUM;
    cfg.pin_d6    = Y8_GPIO_NUM;   cfg.pin_d7    = Y9_GPIO_NUM;
    cfg.pin_xclk  = XCLK_GPIO_NUM;
    cfg.pin_pclk  = PCLK_GPIO_NUM;
    cfg.pin_vsync = VSYNC_GPIO_NUM;
    cfg.pin_href  = HREF_GPIO_NUM;
    cfg.pin_sscb_sda = SIOD_GPIO_NUM;
    cfg.pin_sscb_scl = SIOC_GPIO_NUM;
    cfg.pin_pwdn  = PWDN_GPIO_NUM;
    cfg.pin_reset = RESET_GPIO_NUM;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        cfg.frame_size   = FRAME_SIZE;
        cfg.jpeg_quality = JPEG_QUALITY;
        cfg.fb_count     = 2;
        cfg.grab_mode    = CAMERA_GRAB_LATEST;
        Serial.println("[INFO] PSRAM found — dual frame buffer enabled");
    } else {
        cfg.frame_size   = FRAMESIZE_SVGA;
        cfg.jpeg_quality = 16;
        cfg.fb_count     = 1;
        cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
        Serial.println("[WARN] No PSRAM — single buffer / reduced resolution");
    }

    if (esp_camera_init(&cfg) != ESP_OK) {
        Serial.println("[ERR] Camera init failed!");
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    s->set_framesize(s,       FRAME_SIZE);
    s->set_quality(s,         JPEG_QUALITY);
    s->set_brightness(s,      1);
    s->set_contrast(s,        1);
    s->set_saturation(s,      0);
    s->set_sharpness(s,       1);
    s->set_whitebal(s,        1);
    s->set_awb_gain(s,        1);
    s->set_exposure_ctrl(s,   1);
    s->set_aec2(s,            1);
    s->set_gain_ctrl(s,       1);
    s->set_lenc(s,            1);

    Serial.println("[OK]  Camera initialized");
    return true;
}


// =============================================================
//  WIFI CONNECT
// =============================================================
bool connectWifi() {
    Serial.printf("\n[WiFi] Connecting to: %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
        delay(500);
        Serial.print(".");
        digitalWrite(LED_GPIO_NUM, i % 2);
    }
    digitalWrite(LED_GPIO_NUM, LOW);

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n[ERR] WiFi failed!");
        return false;
    }

    String ip = WiFi.localIP().toString();
    Serial.println("\n[OK]  WiFi connected!");
    Serial.println("┌─────────────────────────────────────────┐");
    Serial.printf( "│  IP      : http://%-22s│\n", ip.c_str());
    Serial.printf( "│  Stream  : http://%s/stream       │\n", ip.c_str());
    Serial.printf( "│  Status  : http://%s:81/status    │\n", ip.c_str());
    Serial.printf( "│  RSSI    : %d dBm%-23s│\n", WiFi.RSSI(), "");
    Serial.println("└─────────────────────────────────────────┘");
    return true;
}


// =============================================================
//  SETUP
// =============================================================
void setup() {
    Serial.begin(115200);
    Serial.println("\n╔═══════════════════════════════════════════╗");
    Serial.println("║  Drowsiness Detection  ESP32-CAM  v3.0   ║");
    Serial.println("║  Mode: WIRELESS (HTTP MJPEG)              ║");
    Serial.println("╚═══════════════════════════════════════════╝");

    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);

    if (!initCamera()) {
        Serial.println("[HALT] Camera error — fast-blinking LED");
        while (true) {
            digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
            delay(200);
        }
    }

    if (!connectWifi()) {
        Serial.println("[HALT] WiFi error — slow-blinking LED");
        while (true) {
            digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
            delay(800);
        }
    }

    startServers();

    // Triple blink = ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_GPIO_NUM, HIGH); delay(120);
        digitalWrite(LED_GPIO_NUM, LOW);  delay(120);
    }
    Serial.println("[READY] Open Python app → Wireless mode → Connect");
}


// =============================================================
//  LOOP  — WiFi watchdog & auto-reconnect
// =============================================================
void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WARN] WiFi lost — attempting reconnect...");

        // Stop servers before reconnect to free ports
        if (stream_httpd) { httpd_stop(stream_httpd); stream_httpd = NULL; }
        if (info_httpd)   { httpd_stop(info_httpd);   info_httpd   = NULL; }

        WiFi.disconnect();
        delay(1000);

        if (connectWifi()) {
            startServers();
            Serial.println("[OK]  Reconnected and servers restarted.");
        }
    }
    delay(5000);
}
