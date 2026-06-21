/* Edge Impulse Arduino examples
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// These sketches are tested with 2.0.4 ESP32 Arduino Core
// https://github.com/espressif/arduino-esp32/releases/tag/2.0.4

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 0

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <Final-Skripsi_inferencing.h>
#include <LiquidCrystal_I2C.h>
#include <Stepper.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s.h"

#define I2S_WS 19
#define I2S_SD 5
#define I2S_SCK 18
#define I2S_PORT I2S_NUM_0

const int full = 2048;
Stepper trapDoor(full, 32, 25, 33, 26);
Stepper sortMotor(full, 27, 12, 14, 13);

typedef struct {
  int16_t *buffer;
  uint8_t buf_ready;
  uint32_t buf_count;
  uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false;
static bool record_status = true;

float threshold = 0.8;
int selected = 2;
int position = 0;

LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  Serial.begin(115200);
  lcd_init();
  trapDoor.setSpeed(15.0);
  sortMotor.setSpeed(15.0);
  while (!Serial);
  Serial.println("Edge Impulse Inferencing Demo");

  ei_printf("Inferencing settings:\n");
  ei_printf("\tInterval: ");
  ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
  ei_printf(" ms.\n");
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
  ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

  if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
    ei_printf("ERR: Could not allocate audio buffer\r\n");
    return;
  }
}

void loop() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Recording...");
  ei_printf("\nRecording...\n");

  delay(100);
  char res[10];

  // Start recording microphone
  if (!microphone_inference_record()) {
    ei_printf("ERR: Failed to record audio...\n");
    return;
  }

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT; // Sampling rate
  signal.get_data = &microphone_audio_signal_get_data; // Microphone data
  ei_impulse_result_t result = { 0 };

  EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
  if (r != EI_IMPULSE_OK) {
    ei_printf("ERR: Failed to run classifier (%d)\n", r);
    return;
  }

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
    if (result.classification[ix].value > threshold) {
      strncpy(res, result.classification[ix].label, sizeof(res) - 1);
      res[sizeof(res) - 1] = '\0';
      selected = ix;
    }
  }

  switch (selected) {
    case 0: // BOTTLE
      lcd.setCursor(0, 0);
      lcd.print("Detected:");

      lcd.setCursor(0, 1);
      lcd.print(res);

      if (position != 512) {
        sortMotor.step(512 - position);
        position = 512;
      }

      trapDoor.step(300);
      delay(2000);
      trapDoor.step(-300);

      lcd.clear();
      break;
    case 1: // CAN
      lcd.setCursor(0, 0);
      lcd.print("Detected:");

      lcd.setCursor(0, 1);
      lcd.print(res);

      if (position != 1024) {
        sortMotor.step(1024 - position);
        position = 1024;
      }

      delay(200);
      trapDoor.step(300);
      delay(2000);
      trapDoor.step(-300);
      delay(200);

      lcd.clear();
      break;
    case 3: // Paper
      lcd.setCursor(0, 0);
      lcd.print("Detected:");

      lcd.setCursor(0, 1);
      lcd.print(res);

      if (position != 0) {
        sortMotor.step(0 - position);
        position = 0;
      }

      delay(200);
      trapDoor.step(300);
      delay(2000);
      trapDoor.step(-300);
      delay(200);

      lcd.clear();
      break;
    case 4: // Ping-Pong
      lcd.setCursor(0, 0);
      lcd.print("Detected:");

      lcd.setCursor(0, 1);
      lcd.print(res);

      if (position != 1536) {
        sortMotor.step(1536 - position);
        position = 1536;
      }

      delay(200);
      trapDoor.step(300);
      delay(2000);
      trapDoor.step(-300);
      delay(200);

      lcd.clear();
      break;
  }
}

void lcd_init() {
  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
}

static void audio_inference_callback(uint32_t n_bytes) {
  for (int i = 0; i < n_bytes >> 1; i++) {
    inference.buffer[inference.buf_count++] = sampleBuffer[i];
    if (inference.buf_count >= inference.n_samples) {
      inference.buf_count = 0;
      inference.buf_ready = 1;
    }
  }
}

static void capture_samples(void *arg) {
  const int32_t i2s_bytes_to_read = (uint32_t)arg;
  size_t bytes_read = i2s_bytes_to_read;

  while (record_status) {
    i2s_read(I2S_PORT, (void *)sampleBuffer, i2s_bytes_to_read, &bytes_read, portMAX_DELAY);
    if (bytes_read > 0) {
      // Loop for gain amplifier
      for (int x = 0; x < i2s_bytes_to_read / 2; x++) {
        int amplified = (int)(sampleBuffer[x] * 4);
        if (amplified > 32767) amplified = 32767;
        if (amplified < -32768) amplified = -32768;
        sampleBuffer[x] = amplified;
      }
      audio_inference_callback(i2s_bytes_to_read);
    }
  }
  vTaskDelete(NULL);
}

static bool microphone_inference_start(uint32_t n_samples) {
  inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
  if (!inference.buffer) return false;
  inference.buf_count = 0;
  inference.n_samples = n_samples;
  inference.buf_ready = 0;

  i2sInit();
  ei_sleep(100);
  record_status = true;
  xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void *)sample_buffer_size, 10, NULL);
  return true;
}

static bool microphone_inference_record(void) {
  while (inference.buf_ready == 0) delay(10);
  inference.buf_ready = 0;
  return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
  return 0;
}

static void i2sInit() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = 0,
    .dma_buf_count = 64,
    .dma_buf_len = 1024,
    .use_apll = true,
    .tx_desc_auto_clear = false,
    .fixed_mclk = -1
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif