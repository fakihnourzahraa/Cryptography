/*
 * entropy_cam.ino — ESP32-CAM (AI-Thinker) as an entropy source
 *
 * The board waits for a one-byte command over USB serial and answers with a
 * framed binary packet. Commands:
 *
 *   'E'  mixed block : SHA-256( tag || camera frame || 32B hardware RNG || boot TRNG seed || counter || micros )
 *   'C'  camera-only : SHA-256( tag || camera frame || counter || micros )
 *   'H'  hw-RNG only : 32 raw bytes from esp_random() (see note on bootloader_random below)
 *   'R'  raw frame   : u16 width, u16 height, u8 format, u8 bytes/pixel, then pixels
 *                      format: 1 = grayscale, 2 = YUV422, 3 = RGB565
 *                      (the laptop converts to brightness itself)
 *   'I'  info        : ASCII text (firmware version, frame size, counters)
 *
 * Packet format (all little-endian):
 *   0xA5 0x5A | type (1 byte, same letter as the command) | length (u32) | payload | CRC32 (u32)
 * The CRC32 is the standard zlib/IEEE CRC over the payload only, so Python can
 * check it with zlib.crc32(). Anything outside a packet (boot messages etc.)
 * is ignored by the laptop, which re-syncs on the 0xA5 0x5A magic.
 *
 * Arduino IDE settings:
 *   Board:            "AI Thinker ESP32-CAM"
 *   PSRAM:            Enabled (default for this board)
 *   Partition scheme: "Huge APP (3MB No OTA)"
 *   Upload speed:     460800 (drop to 115200 if uploads fail)
 * Needs the "esp32" board package by Espressif (2.x or 3.x). The camera driver
 * and mbedTLS ship with it, so no extra libraries are required.
 */

#include "esp_camera.h"
#include "esp_system.h"
#if __has_include("esp_random.h")       // core 3.x (ESP-IDF 5) moved esp_random() here
  #include "esp_random.h"
#endif
#include "esp_log.h"
#include "mbedtls/sha256.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// bootloader_random_enable() turns on an internal noise source so esp_random()
// is a true RNG even though Wi-Fi/Bluetooth are off. BUT on the ESP32 it takes
// over the I2S peripheral, which is exactly what the camera driver captures
// frames through. Espressif: call bootloader_random_disable() before using I2S.
// So we use it only at boot, to collect a 32-byte true-random seed, switch it
// off, and only then start the camera. After that, esp_random() alone is
// documented as pseudo-random (no RF running), which is why the 'E' blocks also
// mix in the boot seed. Header moved between core versions, so include it
// defensively.
#if __has_include("bootloader_random.h")
  #include "bootloader_random.h"
  #define HAVE_BOOTLOADER_RANDOM 1
#endif

#define FW_VERSION  "entropy_cam 1.6"
#define SERIAL_BAUD 921600

// ---- AI-Thinker ESP32-CAM pin map ----
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_LED_GPIO     4

static const uint8_t DOMAIN_TAG[] = "ESP32CAM-ENTROPY-v1";
static uint32_t block_counter = 0;
static char sensor_name[48] = "none";
static bool camera_ok = false;
static uint8_t boot_seed[32];           // true-random bytes collected at boot, before the camera starts
static uint32_t restarts = 0;        // camera driver restarts after a stalled capture

// ---------------------------------------------------------------- CRC32 (zlib)
static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
  crc = ~crc;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (0xEDB88320u & (-(int32_t)(crc & 1)));
  }
  return ~crc;
}

// ---------------------------------------------------------------- framing
static void send_u32(uint32_t v) {
  uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
  Serial.write(b, 4);
}

// Sends a packet whose payload is the concatenation of up to two buffers
// (lets us send a small header + a large frame without copying the frame).
static void send_packet2(char type, const uint8_t *a, size_t alen, const uint8_t *b, size_t blen) {
  const uint8_t magic[2] = {0xA5, 0x5A};
  Serial.write(magic, 2);
  Serial.write((uint8_t)type);
  send_u32((uint32_t)(alen + blen));
  uint32_t crc = 0;
  if (alen) { Serial.write(a, alen); crc = crc32_update(crc, a, alen); }
  if (blen) {
    // write in chunks so the UART buffer keeps up
    const size_t CH = 1024;
    for (size_t off = 0; off < blen; off += CH) {
      size_t n = (blen - off < CH) ? blen - off : CH;
      Serial.write(b + off, n);
    }
    crc = crc32_update(crc, b, blen);
  }
  send_u32(crc);
  Serial.flush();
}

static void send_packet(char type, const uint8_t *p, size_t len) { send_packet2(type, p, len, nullptr, 0); }

static void send_error(const char *msg) { send_packet('!', (const uint8_t *)msg, strlen(msg)); }

// ---------------------------------------------------------------- camera
static bool init_camera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  // Raw RGB565, NOT JPEG: compression throws away exactly the low-order sensor
  // noise we want to harvest. RGB565 is also the one raw format that works on
  // every sensor these boards ship with, including the GC2145 (RHYX M21-45)
  // that many sellers now fit instead of the OV2640.
  c.pixel_format = PIXFORMAT_RGB565;
  // 240 x 240 is the size Espressif's CameraWebServer example uses for raw
  // formats, and the one confirmed working with the GC2145 on this board
  // (320 x 240 captures fail with it). 240 x 240 x 2 = 115,200 bytes per frame.
  c.frame_size = FRAMESIZE_240X240;
  c.fb_count = 1;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  if (esp_camera_init(&c) != ESP_OK) return false;

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    camera_sensor_info_t *si = esp_camera_sensor_get_info(&s->id);
    snprintf(sensor_name, sizeof(sensor_name), "%s (PID 0x%04x)",
             si ? si->name : "unknown", (unsigned)s->id.PID);
    // Only touch sensor settings on the OV2640, whose driver is known to
    // handle them; other sensors keep their defaults (which already have auto
    // exposure and gain on), exactly like CameraWebServer.
    if (s->id.PID == OV2640_PID) {
      s->set_gain_ctrl(s, 1);
      s->set_exposure_ctrl(s, 1);
      s->set_gainceiling(s, GAINCEILING_128X);
      s->set_denoise(s, 0);               // don't let the ISP smooth the noise away
    }
  }
  return true;
}

// Get one complete frame. Some sensors (notably the GC2145) stall when frames
// are requested back to back in raw mode, so instead of grabbing a "fresh"
// second frame we take one, and if the driver stalls we restart it and retry.
// (A frame captured slightly before the request is still a never-used noisy
// frame, so taking it costs no entropy.)
static bool frame_ok(camera_fb_t *fb) {
  return fb && fb->len == (size_t)fb->width * fb->height * 2;   // RGB565 = 2 bytes/pixel
}

static char last_capture_error[80] = "";

static camera_fb_t *fresh_frame() {
  for (int attempt = 0; attempt < 3; attempt++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (frame_ok(fb)) return fb;
    if (fb) {                                    // partial or odd-sized frame
      snprintf(last_capture_error, sizeof(last_capture_error),
               "bad frame: len=%u w=%u h=%u fmt=%d", (unsigned)fb->len,
               (unsigned)fb->width, (unsigned)fb->height, (int)fb->format);
      esp_camera_fb_return(fb);
    } else {
      snprintf(last_capture_error, sizeof(last_capture_error), "no frame from driver (timeout)");
    }
    // Stalled or partial: restart the camera driver and try again.
    esp_camera_deinit();
    delay(50);
    camera_ok = init_camera();
    esp_log_level_set("*", ESP_LOG_NONE);
    if (!camera_ok) return nullptr;
    restarts++;
  }
  return nullptr;
}

// ---------------------------------------------------------------- commands
static void send_capture_error() {
  char msg[140];
  snprintf(msg, sizeof(msg), "frame capture failed after %lu restarts: %s",
           (unsigned long)restarts, last_capture_error);
  send_error(msg);
}

static void cmd_block(bool mix_hw_rng) {
  if (!camera_ok) { send_error("camera not initialised"); return; }
  camera_fb_t *fb = fresh_frame();
  if (!fb) { send_capture_error(); return; }

  uint8_t hw[32];
  uint32_t ctr = block_counter++;
  uint32_t t = (uint32_t)micros();
  uint8_t out[32];

  mbedtls_sha256_context ctx;
  mbedtls_sha256_init(&ctx);
  mbedtls_sha256_starts(&ctx, 0);                       // 0 = SHA-256 (not 224)
  mbedtls_sha256_update(&ctx, DOMAIN_TAG, sizeof(DOMAIN_TAG) - 1);
  mbedtls_sha256_update(&ctx, fb->buf, fb->len);
  if (mix_hw_rng) {
    esp_fill_random(hw, sizeof(hw));
    mbedtls_sha256_update(&ctx, hw, sizeof(hw));
    mbedtls_sha256_update(&ctx, boot_seed, sizeof(boot_seed));
  }
  mbedtls_sha256_update(&ctx, (const uint8_t *)&ctr, 4);
  mbedtls_sha256_update(&ctx, (const uint8_t *)&t, 4);
  mbedtls_sha256_finish(&ctx, out);
  mbedtls_sha256_free(&ctx);

  esp_camera_fb_return(fb);
  memset(hw, 0, sizeof(hw));
  send_packet(mix_hw_rng ? 'E' : 'C', out, sizeof(out));
  memset(out, 0, sizeof(out));
}

static void cmd_hw_only() {
  uint8_t hw[32];
  esp_fill_random(hw, sizeof(hw));
  send_packet('H', hw, sizeof(hw));
  memset(hw, 0, sizeof(hw));
}

static void cmd_raw() {
  if (!camera_ok) { send_error("camera not initialised"); return; }
  camera_fb_t *fb = fresh_frame();
  if (!fb) { send_capture_error(); return; }
  uint8_t hdr[6] = {(uint8_t)fb->width, (uint8_t)(fb->width >> 8),
                    (uint8_t)fb->height, (uint8_t)(fb->height >> 8),
                    3 /* RGB565 */, 2 /* bytes per pixel */};
  send_packet2('R', hdr, 6, fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

static void cmd_info() {
  char buf[240];
  snprintf(buf, sizeof(buf),
           "%s | camera=%s | sensor=%s | frame=240x240 rgb565 | blocks=%lu | restarts=%lu | psram=%u | trng=%s",
           FW_VERSION, camera_ok ? "ok" : "FAILED", sensor_name, (unsigned long)block_counter,
           (unsigned long)restarts,
           (unsigned)ESP.getPsramSize(),
#ifdef HAVE_BOOTLOADER_RANDOM
           "boot-seed (then esp_random pseudo)"
#else
           "esp_random only (pseudo-random!)"
#endif
  );
  send_packet('I', (const uint8_t *)buf, strlen(buf));
}

// ---------------------------------------------------------------- setup/loop
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // camera inrush often trips the brownout detector
  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);           // keep the bright flash LED off

  Serial.setTxBufferSize(4096);                // must be set before begin()
  Serial.begin(SERIAL_BAUD);

#ifdef HAVE_BOOTLOADER_RANDOM
  bootloader_random_enable();                  // true RNG without Wi-Fi/BT...
  esp_fill_random(boot_seed, sizeof(boot_seed));
  bootloader_random_disable();                 // ...switched OFF before the camera uses I2S
#else
  esp_fill_random(boot_seed, sizeof(boot_seed));
#endif

  camera_ok = init_camera();
  esp_log_level_set("*", ESP_LOG_NONE);        // keep driver logs out of the binary stream
  digitalWrite(FLASH_LED_GPIO, LOW);
}

void loop() {
  if (!Serial.available()) { delay(1); return; }
  char c = (char)Serial.read();
  switch (c) {
    case 'E': cmd_block(true);  break;
    case 'C': cmd_block(false); break;
    case 'H': cmd_hw_only();    break;
    case 'R': cmd_raw();        break;
    case 'I': cmd_info();       break;
    default: break;                            // ignore newlines / noise
  }
}
