#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <MPU9250.h>
#include <math.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "sentinel_core.h"

// ================== CONFIG RÉSEAU ==================
const char* WIFI_SSID     = "test";
const char* WIFI_PASSWORD = "123456789";
const char* MQTT_SERVER   = "10.55.236.10";
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "sentinelle/moteur/vibrations";

WiFiClient espClient;
PubSubClient mqttClient(espClient);
MPU9250 mpu;

// ================== TFLite ==================
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  tflite::ErrorReporter* error_reporter = nullptr;

  constexpr int kArenaSize = 12 * 1024;
  uint8_t tensor_arena[kArenaSize] __attribute__((aligned(16)));

  const int WINDOW_SIZE = 15;
  float acc_mag_buf[WINDOW_SIZE];
  float gyro_mag_buf[WINDOW_SIZE];
  int buf_idx = 0;
  bool buffer_full = false;
}

// Scaler (valeurs exactes du dernier entraînement)
const float SCALER_MEAN[6]  = {1.3631f, 3.147927f, 0.810827f, 59.193f, 137.98776f, 33.789809f};
const float SCALER_SCALE[6] = {0.660375f, 2.669315f, 0.892057f, 53.068104f, 160.616163f, 37.930314f};

// ================== Anti-rebond ==================
String current_state = "NORMAL";
String pending_state = "NORMAL";
int state_counter = 0;
const int DEBOUNCE_COUNT = 3;   // Nombre de fenêtres consécutives avant de changer d'état

void setupWiFi() {
  Serial.print("[WiFi] Connexion...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println("\n[WiFi] OK - " + WiFi.localIP().toString());
}

void reconnectMQTT() {
  while (!mqttClient.connected()) {
    String clientId = "ESP32_Sentinelle_" + String(random(0xffff), HEX);
    if (mqttClient.connect(clientId.c_str())) {
      Serial.println("[MQTT] Connecté");
    } else {
      Serial.print("[MQTT] Échec, rc=");
      Serial.println(mqttClient.state());
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  setupWiFi();
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_sentinel_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[-] Schema TFLite incompatible");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[-] AllocateTensors échoué");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Wire.begin();
  if (!mpu.setup(0x68)) {
    Serial.println("[-] MPU9250 non trouvé");
    while (1);
  }

  Serial.println("\n>>> SENTINELLE TinyML (6 features + règles + debounce) READY <<<\n");
}

void loop() {
  if (!mqttClient.connected()) reconnectMQTT();
  mqttClient.loop();

  if (mpu.update()) {
    float ax = mpu.getAccX(), ay = mpu.getAccY(), az = mpu.getAccZ();
    float gx = mpu.getGyroX(), gy = mpu.getGyroY(), gz = mpu.getGyroZ();

    float acc_mag  = sqrtf(ax*ax + ay*ay + az*az);
    float gyro_mag = sqrtf(gx*gx + gy*gy + gz*gz);

    acc_mag_buf[buf_idx]  = acc_mag;
    gyro_mag_buf[buf_idx] = gyro_mag;
    buf_idx = (buf_idx + 1) % WINDOW_SIZE;
    if (buf_idx == 0) buffer_full = true;

    if (buffer_full) {
      // Calcul des 6 features
      float acc_sum = 0, acc_max = 0, acc_sq = 0;
      float gyro_sum = 0, gyro_max = 0, gyro_sq = 0;

      for (int i = 0; i < WINDOW_SIZE; i++) {
        float a = acc_mag_buf[i];
        float g = gyro_mag_buf[i];
        acc_sum += a; gyro_sum += g;
        if (a > acc_max)  acc_max = a;
        if (g > gyro_max) gyro_max = g;
      }

      float acc_mean  = acc_sum  / WINDOW_SIZE;
      float gyro_mean = gyro_sum / WINDOW_SIZE;

      for (int i = 0; i < WINDOW_SIZE; i++) {
        float da = acc_mag_buf[i]  - acc_mean;
        float dg = gyro_mag_buf[i] - gyro_mean;
        acc_sq  += da * da;
        gyro_sq += dg * dg;
      }

      float acc_std  = sqrtf(acc_sq  / WINDOW_SIZE);
      float gyro_std = sqrtf(gyro_sq / WINDOW_SIZE);

      // Scaling
      float features[6] = {
        (acc_mean  - SCALER_MEAN[0]) / SCALER_SCALE[0],
        (acc_max   - SCALER_MEAN[1]) / SCALER_SCALE[1],
        (acc_std   - SCALER_MEAN[2]) / SCALER_SCALE[2],
        (gyro_mean - SCALER_MEAN[3]) / SCALER_SCALE[3],
        (gyro_max  - SCALER_MEAN[4]) / SCALER_SCALE[4],
        (gyro_std  - SCALER_MEAN[5]) / SCALER_SCALE[5]
      };

      for (int i = 0; i < 6; i++) input->data.f[i] = features[i];

      if (interpreter->Invoke() == kTfLiteOk) {
        float p_norm  = output->data.f[0];
        float p_fric  = output->data.f[1];
        float p_shock = output->data.f[2];

        // =============================================
        // 1. DÉCISION BRUTE (seuils ajustés)
        // =============================================
        String raw_state = "NORMAL";

        // Choc (plus strict)
        if (acc_max > 2.8f || gyro_max > 280.0f || (acc_std > 0.65f && gyro_max > 150.0f)) {
          raw_state = "CHOC";
        }
        // Frottement
        else if (gyro_mean > 22.0f || acc_std > 0.11f || gyro_std > 18.0f) {
          raw_state = "FROTTEMENT";
        }
        // Fallback modèle
        else if (p_shock > 0.55f) {
          raw_state = "CHOC";
        }
        else if (p_fric > 0.50f) {
          raw_state = "FROTTEMENT";
        }

        // =============================================
        // 2. FILTRE ANTI-REBOND
        // =============================================
        if (raw_state == pending_state) {
          state_counter++;
        } else {
          pending_state = raw_state;
          state_counter = 1;
        }

        if (state_counter >= DEBOUNCE_COUNT) {
          current_state = pending_state;
        }

        // =============================================
        // 3. MQTT
        // =============================================
        String payload = "{";
        payload += "\"state\":\"" + current_state + "\",";
        payload += "\"acc_mean\":"  + String(acc_mean, 3)  + ",";
        payload += "\"acc_std\":"   + String(acc_std, 3)   + ",";
        payload += "\"gyro_mean\":" + String(gyro_mean, 1) + ",";
        payload += "\"gyro_max\":"  + String(gyro_max, 1)  + ",";
        payload += "\"p_norm\":"    + String(p_norm  * 100, 1) + ",";
        payload += "\"p_fric\":"    + String(p_fric  * 100, 1) + ",";
        payload += "\"p_shock\":"   + String(p_shock * 100, 1);
        payload += "}";

        mqttClient.publish(MQTT_TOPIC, payload.c_str());
        Serial.println("[MQTT] " + payload);
      }
    }
  }
  delay(30);
}