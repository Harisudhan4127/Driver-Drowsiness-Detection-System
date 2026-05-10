// =============================================================
//  DRIVER DROWSINESS DETECTION — ESP32-CAM FIRMWARE  v3.0
//  Mode    : WIRED  —  Raw JPEG frames over UART / Serial
//  Board   : AI-Thinker ESP32-CAM
//  Baud    : 921600  (USB-to-UART adapter required, e.g. CP2102)
//
//  ── HOW TO WIRE ──────────────────────────────────────────────
//  ESP32-CAM        USB-to-UART adapter (CP2102 / CH340 / FTDI)
//  ─────────────────────────────────────────────────────────────
//  5V       ──────► 5V   (must supply 5V, not 3.3V)
//  GND      ──────► GND
//  GPIO1/TX ──────► RX
//  GPIO3/RX ──────► TX
//  IO0      ──►[GND during upload, open during normal run]
//
//  ⚠  Powering from adapter 3.3V pin causes camera brown-outs.
//     Always use the 5V pin or a dedicated 5V 1A supply.
//
//  ── HOW TO UPLOAD ────────────────────────────────────────────
//  1. Hold IO0 to GND, press RESET, release IO0.
//  2. Arduino IDE → Board: "AI Thinker ESP32-CAM"
//     Upload speed: 115200 for upload, then switch to 921600
//     in Serial Monitor to see runtime logs.
//  3. After upload: release IO0, press RESET.
//
//  ── HOW TO USE IN PYTHON APP ─────────────────────────────────
//  Mode    → Serial (COM Port — ESP32-CAM)
//  COM Port→ select the matching COM port (e.g. COM11)
//  Click Connect — frames arrive immediately.
//
//  ── FRAME PROTOCOL ───────────────────────────────────────────
//  The firmware writes raw JPEG bytes to Serial2 (921600 baud).
//  Each frame is a complete JPEG:  FF D8 ... FF D9
//  The Python app detects SOI (FF D8) and EOI (FF D9) markers
//  to slice individual frames — no extra framing overhead.
//
//  ── THROUGHPUT GUIDE ─────────────────────────────────────────
//  QVGA  (320×240)  JPEG-Q 16  →  ~25–30 fps @ 921600
//  VGA   (640×480)  JPEG-Q 16  →  ~12–15 fps @ 921600
//  SVGA  (800×600)  JPEG-Q 16  →  ~7–10 fps @ 921600
//
//  Higher baud (1.5 Mbps, 2 Mbps) may work depending on the
//  adapter chip but 921600 is universally safe.
// =============================================================

#include "esp_camera.h"
#include "esp_timer.h"
#include "driver/uart.h"

// ─── Stream Quality ──────────────────────────────────────────
//  For serial, QVGA or VGA is recommended for smooth FPS.
//  JPEG_QUALITY: 10 = best, 30 = faster — tune to your link.
#define FRAME_SIZE    FRAMESIZE_VGA
#define JPEG_QUALITY  16
#define SERIAL_BAUD   921600      // Must match Python app setting

// ─── Frame timing ────────────────────────────────────────────
#define TARGET_FPS    15          // Cap to avoid overflowing UART
#define FRAME_DELAY_MS  (1000 / TARGET_FPS)

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

// ─── Internal serial port for debug logs ─────────────────────
//  Serial  (GPIO1/GPIO3) = data link to PC at 921600 baud
//  We reuse the same UART at 921600 for both data and logs.
//  Logs are printed ONLY before streaming starts; once streaming
//  begins, no text is mixed into the data stream.


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
    } else {
        // Without PSRAM, keep resolution low to avoid heap exhaustion
        cfg.frame_size   = FRAMESIZE_QVGA;
        cfg.jpeg_quality = 20;
        cfg.fb_count     = 1;
        cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    }

    if (esp_camera_init(&cfg) != ESP_OK) {
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    s->set_framesize(s,     psramFound() ? FRAME_SIZE : FRAMESIZE_QVGA);
    s->set_quality(s,       psramFound() ? JPEG_QUALITY : 20);
    s->set_brightness(s,    1);
    s->set_contrast(s,      1);
    s->set_saturation(s,    0);
    s->set_sharpness(s,     1);
    s->set_whitebal(s,      1);
    s->set_awb_gain(s,      1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s,          1);
    s->set_gain_ctrl(s,     1);
    s->set_lenc(s,          1);
    s->set_hmirror(s,       0);   // set 1 to mirror horizontally
    s->set_vflip(s,         0);   // set 1 to flip vertically

    return true;
}


// =============================================================
//  SETUP
// =============================================================
void setup() {
    // ── Debug log phase (115200 baud) ────────────────────────
    //  Print startup info at a safe baud, then switch to 921600
    //  for the streaming data link.
    Serial.begin(115200);

    Serial.println("\n╔═══════════════════════════════════════════╗");
    Serial.println("║  Drowsiness Detection  ESP32-CAM  v3.0   ║");
    Serial.println("║  Mode: WIRED (UART / Serial COM Port)     ║");
    Serial.println("╚═══════════════════════════════════════════╝");

    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);

    Serial.println("[INIT] Initializing camera...");
    if (!initCamera()) {
        Serial.println("[ERR]  Camera init FAILED — halting.");
        while (true) {
            digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
            delay(200);   // fast blink = error
        }
    }

    bool has_psram = psramFound();
    Serial.printf("[INFO] PSRAM    : %s\n", has_psram ? "YES" : "NO");
    Serial.printf("[INFO] Res      : %s\n",
        has_psram ? "VGA (640×480)" : "QVGA (320×240)");
    Serial.printf("[INFO] Quality  : %d\n", has_psram ? JPEG_QUALITY : 20);
    Serial.printf("[INFO] Target   : %d fps\n", TARGET_FPS);
    Serial.println("[INFO] Switching UART to 921600 baud for data streaming...");
    Serial.flush();

    // ── Switch to high-speed baud for JPEG stream ────────────
    Serial.begin(SERIAL_BAUD);
    delay(100);   // let UART settle

    // Triple blink = ready to stream
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_GPIO_NUM, HIGH); delay(120);
        digitalWrite(LED_GPIO_NUM, LOW);  delay(120);
    }

    // ── IMPORTANT: stop printing text from this point on ─────
    //  Any text written to Serial after this point will corrupt
    //  the JPEG data stream.  Use LED blinks for status only.
}


// =============================================================
//  LOOP  — capture + transmit JPEG frames
// =============================================================
void loop() {
    unsigned long t0 = millis();

    camera_fb_t* fb = esp_camera_fb_get();

    if (!fb) {
        // Camera error — blink twice and continue
        for (int i = 0; i < 2; i++) {
            digitalWrite(LED_GPIO_NUM, HIGH); delay(50);
            digitalWrite(LED_GPIO_NUM, LOW);  delay(50);
        }
        delay(100);
        return;
    }

    // Sanity check: JPEG must start with FF D8 and end with FF D9
    if (fb->len > 4 &&
        fb->buf[0]          == 0xFF && fb->buf[1]          == 0xD8 &&
        fb->buf[fb->len - 2] == 0xFF && fb->buf[fb->len - 1] == 0xD9)
    {
        // Write the complete JPEG — Python reads FF D8 … FF D9
        Serial.write(fb->buf, fb->len);
        Serial.flush();
    }

    esp_camera_fb_return(fb);

    // ── Rate limiting ─────────────────────────────────────────
    //  Prevent UART overrun on slow host-side USB stacks.
    long elapsed = (long)(millis() - t0);
    long wait    = (long)FRAME_DELAY_MS - elapsed;
    if (wait > 0) {
        delay((uint32_t)wait);
    }
}
