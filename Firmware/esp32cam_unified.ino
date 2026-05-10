// =============================================================
//  DRIVER DROWSINESS DETECTION — ESP32-CAM FIRMWARE  v4.0
//  Board   : AI-Thinker ESP32-CAM
//  Author  : github.com/your-username
//
//  ╔══════════════════════════════════════════════════════════╗
//  ║          DUAL-MODE  —  WIRELESS  +  WIRED (SERIAL)      ║
//  ║                                                          ║
//  ║  Mode is selected at boot via GPIO12 (MODE_PIN):        ║
//  ║    GPIO12 → GND   =  WIRED  (UART serial JPEG stream)   ║
//  ║    GPIO12 → open  =  WIRELESS (HTTP MJPEG over WiFi)    ║
//  ║                                                          ║
//  ║  You can also force a mode by setting FORCE_MODE below. ║
//  ╚══════════════════════════════════════════════════════════╝
//
//  ── WIRELESS MODE ────────────────────────────────────────────
//  1. Set WIFI_SSID and WIFI_PASSWORD.
//  2. Leave GPIO12 floating (or set FORCE_MODE = MODE_WIRELESS).
//  3. Upload, open Serial Monitor @ 115200 — note the IP.
//  4. Python app → Wireless → IP:<printed IP> Port:80 Path:/stream
//
//  Endpoints:
//    http://<IP>/           → MJPEG stream
//    http://<IP>/stream     → MJPEG stream (alias)
//    http://<IP>:81/status  → JSON (IP, RSSI, uptime, heap...)
//    http://<IP>:81/led?v=1 → Flash LED on/off
//
//  ── WIRED / SERIAL MODE ──────────────────────────────────────
//  Wire ESP32-CAM → USB-UART adapter (CP2102 / CH340 / FTDI):
//    ESP32-CAM 5V   → adapter 5V   (MUST be 5V, not 3.3V)
//    ESP32-CAM GND  → adapter GND
//    GPIO1 / TX     → adapter RX
//    GPIO3 / RX     → adapter TX
//    GPIO12         → GND          (selects wired mode)
//    IO0            → GND during upload only, open when running
//
//  1. Tie GPIO12 to GND (or set FORCE_MODE = MODE_SERIAL).
//  2. Upload at 115200 — logs print, then baud switches to 921600.
//  3. Python app → Serial (COM Port) → select your COM port.
//
//  Frame protocol: raw JPEG bytes, FF D8 … FF D9.
//  Python app slices on SOI/EOI markers automatically.
//
//  ── FORCE MODE (override GPIO12 strap) ───────────────────────
//  Uncomment ONE of these to hard-code a mode regardless of pin:
//    #define FORCE_MODE  MODE_WIRELESS
//    #define FORCE_MODE  MODE_SERIAL
// =============================================================

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include <WiFi.h>

// ─── WiFi Credentials (wireless mode only) ───────────────────
#define WIFI_SSID      "YOUR_WIFI_SSID"
#define WIFI_PASSWORD  "YOUR_WIFI_PASSWORD"

// ─── Stream Quality ──────────────────────────────────────────
//  Wireless recommended: FRAMESIZE_VGA,  JPEG_QUALITY 12
//  Serial recommended  : FRAMESIZE_VGA,  JPEG_QUALITY 16
//  Options: FRAMESIZE_QVGA(320×240) VGA(640×480) SVGA(800×600)
#define FRAME_SIZE_WIFI    FRAMESIZE_VGA
#define JPEG_QUALITY_WIFI  12

#define FRAME_SIZE_SERIAL  FRAMESIZE_VGA
#define JPEG_QUALITY_SERIAL 16

// ─── Serial mode settings ────────────────────────────────────
#define SERIAL_BAUD    921600
#define TARGET_FPS     15
#define FRAME_DELAY_MS (1000 / TARGET_FPS)

// ─── Mode selection pin ──────────────────────────────────────
//  Tie GPIO12 to GND for SERIAL, leave floating for WIRELESS.
//  GPIO12 is safe to use as input (no camera bus conflict).
#define MODE_PIN       12

// ─── Mode constants ──────────────────────────────────────────
#define MODE_WIRELESS  0
#define MODE_SERIAL    1

// ─── Uncomment to hard-force a mode (ignores MODE_PIN) ───────
// #define FORCE_MODE  MODE_WIRELESS
// #define FORCE_MODE  MODE_SERIAL

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
#define LED_GPIO_NUM     4

// ─── Runtime state ───────────────────────────────────────────
int              g_mode        = MODE_WIRELESS;
httpd_handle_t   stream_httpd  = NULL;
httpd_handle_t   info_httpd    = NULL;
bool             g_streaming   = false;   // true once serial loop starts

// ─── MJPEG framing strings ───────────────────────────────────
static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=frame";
static const char* STREAM_BOUNDARY =
    "\r\n--frame\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "X-Timestamp: %lld\r\n\r\n";


// =============================================================
//  CAMERA INIT
// =============================================================
bool initCamera(framesize_t fsize, int quality) {
    camera_config_t cfg;
    cfg.ledc_channel     = LEDC_CHANNEL_0;
    cfg.ledc_timer       = LEDC_TIMER_0;
    cfg.pin_d0           = Y2_GPIO_NUM;
    cfg.pin_d1           = Y3_GPIO_NUM;
    cfg.pin_d2           = Y4_GPIO_NUM;
    cfg.pin_d3           = Y5_GPIO_NUM;
    cfg.pin_d4           = Y6_GPIO_NUM;
    cfg.pin_d5           = Y7_GPIO_NUM;
    cfg.pin_d6           = Y8_GPIO_NUM;
    cfg.pin_d7           = Y9_GPIO_NUM;
    cfg.pin_xclk         = XCLK_GPIO_NUM;
    cfg.pin_pclk         = PCLK_GPIO_NUM;
    cfg.pin_vsync        = VSYNC_GPIO_NUM;
    cfg.pin_href         = HREF_GPIO_NUM;
    cfg.pin_sscb_sda     = SIOD_GPIO_NUM;
    cfg.pin_sscb_scl     = SIOC_GPIO_NUM;
    cfg.pin_pwdn         = PWDN_GPIO_NUM;
    cfg.pin_reset        = RESET_GPIO_NUM;
    cfg.xclk_freq_hz     = 20000000;
    cfg.pixel_format     = PIXFORMAT_JPEG;

    if (psramFound()) {
        cfg.frame_size   = fsize;
        cfg.jpeg_quality = quality;
        cfg.fb_count     = 2;
        cfg.grab_mode    = CAMERA_GRAB_LATEST;
        Serial.println("[INFO] PSRAM found  — dual frame buffer");
    } else {
        // No PSRAM: drop to QVGA to avoid heap exhaustion
        cfg.frame_size   = FRAMESIZE_QVGA;
        cfg.jpeg_quality = 20;
        cfg.fb_count     = 1;
        cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
        Serial.println("[WARN] No PSRAM     — QVGA single buffer");
    }

    if (esp_camera_init(&cfg) != ESP_OK) {
        Serial.println("[ERR]  Camera init FAILED");
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    s->set_framesize(s,       psramFound() ? fsize : FRAMESIZE_QVGA);
    s->set_quality(s,         psramFound() ? quality : 20);
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
    s->set_hmirror(s,         0);
    s->set_vflip(s,           0);

    Serial.println("[OK]   Camera initialized");
    return true;
}


// =============================================================
//  HELPER — blink LED N times
// =============================================================
void blinkLED(int times, int on_ms = 120, int off_ms = 120) {
    for (int i = 0; i < times; i++) {
        digitalWrite(LED_GPIO_NUM, HIGH); delay(on_ms);
        digitalWrite(LED_GPIO_NUM, LOW);  delay(off_ms);
    }
}

void haltWithBlink(int period_ms) {
    Serial.println("[HALT] System halted — check wiring / config.");
    while (true) {
        digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
        delay(period_ms);
    }
}


// =============================================================
//  ██     ██ ██ ██████  ███████ ██      ███████ ███████ ███████
//  ██     ██ ██ ██   ██ ██      ██      ██      ██      ██
//  ██  █  ██ ██ ██████  █████   ██      █████   ███████ ███████
//  ██ ███ ██ ██ ██   ██ ██      ██      ██           ██      ██
//   ███ ███  ██ ██   ██ ███████ ███████ ███████ ███████ ███████
// =============================================================

// ─── MJPEG HTTP handler ──────────────────────────────────────
static esp_err_t stream_handler(httpd_req_t* req) {
    camera_fb_t* fb    = NULL;
    esp_err_t    res   = ESP_OK;
    char         part_buf[128];
    int64_t      frames = 0;

    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "60");

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[WARN] Frame capture failed");
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
            Serial.printf("[INFO] WiFi: %lld frames  RSSI: %d dBm\n",
                          frames, WiFi.RSSI());
        }
    }
    return res;
}

// ─── /status JSON handler ────────────────────────────────────
static esp_err_t status_handler(httpd_req_t* req) {
    char buf[384];
    sensor_t* s = esp_camera_sensor_get();
    snprintf(buf, sizeof(buf),
        "{"
        "\"firmware\":\"unified-v4.0\","
        "\"mode\":\"wireless\","
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

// ─── /led control handler ────────────────────────────────────
static esp_err_t led_handler(httpd_req_t* req) {
    char query[32] = {0};
    httpd_req_get_url_query_str(req, query, sizeof(query));
    digitalWrite(LED_GPIO_NUM, strstr(query, "v=1") ? HIGH : LOW);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "OK", 2);
}

// ─── Start HTTP servers ──────────────────────────────────────
void startServers() {
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
        Serial.println("[OK]   Stream server → port 80  ( /  /stream )");
    }

    httpd_config_t icfg  = HTTPD_DEFAULT_CONFIG();
    icfg.server_port      = 81;
    icfg.ctrl_port        = 32769;
    icfg.max_uri_handlers = 4;

    httpd_uri_t status_uri = { "/status", HTTP_GET, status_handler, NULL };
    httpd_uri_t led_uri    = { "/led",    HTTP_GET, led_handler,    NULL };

    if (httpd_start(&info_httpd, &icfg) == ESP_OK) {
        httpd_register_uri_handler(info_httpd, &status_uri);
        httpd_register_uri_handler(info_httpd, &led_uri);
        Serial.println("[OK]   Info server   → port 81  ( /status  /led )");
    }
}

// ─── WiFi connect ────────────────────────────────────────────
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
        Serial.println("\n[ERR]  WiFi connection FAILED!");
        return false;
    }

    String ip = WiFi.localIP().toString();
    Serial.println("\n[OK]   WiFi connected!");
    Serial.println("  ┌────────────────────────────────────────────┐");
    Serial.printf( "  │  IP      : http://%-24s│\n", ip.c_str());
    Serial.printf( "  │  Stream  : http://%s/stream         │\n", ip.c_str());
    Serial.printf( "  │  Status  : http://%s:81/status      │\n", ip.c_str());
    Serial.printf( "  │  LED on  : http://%s:81/led?v=1     │\n", ip.c_str());
    Serial.printf( "  │  RSSI    : %d dBm                         │\n", WiFi.RSSI());
    Serial.println("  └────────────────────────────────────────────┘");
    return true;
}

// ─── Wireless setup entry point ──────────────────────────────
void setupWireless() {
    Serial.println("[MODE] WIRELESS — HTTP MJPEG over WiFi");

    if (!initCamera(FRAME_SIZE_WIFI, JPEG_QUALITY_WIFI)) {
        haltWithBlink(200);
    }
    if (!connectWifi()) {
        haltWithBlink(800);
    }

    startServers();
    blinkLED(3);
    Serial.println("[READY] Python app → Wireless mode → Connect");
}

// ─── Wireless loop ───────────────────────────────────────────
void loopWireless() {
    // WiFi watchdog — auto-reconnect if signal drops
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WARN] WiFi lost — reconnecting...");
        if (stream_httpd) { httpd_stop(stream_httpd); stream_httpd = NULL; }
        if (info_httpd)   { httpd_stop(info_httpd);   info_httpd   = NULL; }
        WiFi.disconnect();
        delay(1000);
        if (connectWifi()) {
            startServers();
        }
    }
    delay(5000);
}


// =============================================================
//   ███████ ███████ ██████  ██  █████  ██
//   ██      ██      ██   ██ ██ ██   ██ ██
//   ███████ █████   ██████  ██ ███████ ██
//        ██ ██      ██   ██ ██ ██   ██ ██
//   ███████ ███████ ██   ██ ██ ██   ██ ███████
// =============================================================

// ─── Serial setup entry point ────────────────────────────────
void setupSerial() {
    Serial.println("[MODE] WIRED — Raw JPEG over UART @ 921600 baud");
    Serial.println("[INFO] Protocol: FF D8 ... FF D9 (JPEG SOI/EOI markers)");
    Serial.println("[INFO] Python app → Serial (COM Port) → Connect");

    if (!initCamera(FRAME_SIZE_SERIAL, JPEG_QUALITY_SERIAL)) {
        haltWithBlink(200);
    }

    bool has_psram = psramFound();
    Serial.printf("[INFO] Resolution : %s\n",
        has_psram ? "VGA (640×480)" : "QVGA (320×240)");
    Serial.printf("[INFO] JPEG quality: %d\n",
        has_psram ? JPEG_QUALITY_SERIAL : 20);
    Serial.printf("[INFO] Target FPS : %d\n", TARGET_FPS);
    Serial.println("[INFO] Switching UART to 921600 baud — starting stream...");
    Serial.flush();

    // Switch baud — from this point NO text must be sent to Serial;
    // any non-JPEG byte would corrupt the frame stream.
    Serial.begin(SERIAL_BAUD);
    delay(100);

    // 5 fast blinks = serial streaming active
    blinkLED(5, 60, 60);

    g_streaming = true;   // tell loop() to run the serial path
}

// ─── Serial loop — capture + send JPEG frames ────────────────
void loopSerial() {
    unsigned long t0 = millis();

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        // Camera hiccup — brief double-blink, continue
        blinkLED(2, 40, 40);
        delay(80);
        return;
    }

    // Verify the buffer is a complete JPEG (FF D8 … FF D9)
    bool valid_jpeg =
        fb->len > 4 &&
        fb->buf[0]           == 0xFF && fb->buf[1]           == 0xD8 &&
        fb->buf[fb->len - 2] == 0xFF && fb->buf[fb->len - 1] == 0xD9;

    if (valid_jpeg) {
        // Write raw JPEG bytes — Python side detects SOI/EOI
        Serial.write(fb->buf, fb->len);
        Serial.flush();
    }

    esp_camera_fb_return(fb);

    // Rate-limit to TARGET_FPS to avoid UART TX overflow
    long elapsed = (long)(millis() - t0);
    long wait    = (long)FRAME_DELAY_MS - elapsed;
    if (wait > 0) delay((uint32_t)wait);
}


// =============================================================
//  SETUP  —  detect mode, init accordingly
// =============================================================
void setup() {
    Serial.begin(115200);
    delay(200);

    Serial.println("\n╔══════════════════════════════════════════════╗");
    Serial.println("║   Drowsiness Detection  ESP32-CAM   v4.0    ║");
    Serial.println("║   Dual-Mode: Wireless + Wired (Serial)      ║");
    Serial.println("╚══════════════════════════════════════════════╝");

    // ── LED setup ────────────────────────────────────────────
    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);

    // ── Mode detection ───────────────────────────────────────
#ifdef FORCE_MODE
    g_mode = FORCE_MODE;
    Serial.printf("[BOOT] Mode FORCED: %s\n",
        g_mode == MODE_WIRELESS ? "WIRELESS" : "SERIAL");
#else
    // Read GPIO12: LOW (tied to GND) = serial, HIGH (floating) = wireless
    pinMode(MODE_PIN, INPUT_PULLUP);
    delay(10);   // settle
    g_mode = (digitalRead(MODE_PIN) == LOW) ? MODE_SERIAL : MODE_WIRELESS;
    Serial.printf("[BOOT] GPIO12 = %s → Mode: %s\n",
        digitalRead(MODE_PIN) == LOW ? "LOW" : "HIGH (floating)",
        g_mode == MODE_WIRELESS ? "WIRELESS" : "SERIAL");
#endif

    // Single blink: 1 = wireless, 2 = serial
    blinkLED(g_mode == MODE_WIRELESS ? 1 : 2, 200, 150);

    // ── Branch to mode-specific setup ────────────────────────
    if (g_mode == MODE_WIRELESS) {
        setupWireless();
    } else {
        setupSerial();
    }
}


// =============================================================
//  LOOP
// =============================================================
void loop() {
    if (g_mode == MODE_WIRELESS) {
        loopWireless();
    } else {
        loopSerial();
    }
}
