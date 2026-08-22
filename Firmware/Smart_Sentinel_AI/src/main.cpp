#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <MPU9250.h>
#include <math.h>

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "sentinel_core.h"

// ==========================================
// CONFIGURATION RÉSEAU & MQTT
// ==========================================
const char* WIFI_SSID     = "test";
const char* WIFI_PASSWORD = "123456789";

const char* MQTT_SERVER = "10.55.236.10";
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC    = "sentinelle/moteur/vibrations";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// Capteur
MPU9250 mpu;

// Variables Globales TFLite
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  tflite::ErrorReporter* error_reporter = nullptr;

  constexpr int kArenaSize = 10240; 
  uint8_t tensor_arena[kArenaSize] __attribute__((aligned(16)));

  const int WINDOW_SIZE = 15;
  float mag_buffer[WINDOW_SIZE];
  int buffer_index = 0;
}

// Fonction pour maintenir la connexion WiFi
void setupWiFi() {
    delay(10);
    Serial.println();
    Serial.print("[WiFi] Connexion à ");
    Serial.println(WIFI_SSID);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n[WiFi] Connecté ! Adresse IP: " + WiFi.localIP().toString());
}

// Fonction pour reconnecter le MQTT
void reconnectMQTT() {
    while (!mqttClient.connected()) {
        Serial.print("[MQTT] Connexion au broker (" + String(MQTT_SERVER) + ")...");
        String clientId = "ESP32_Sentinelle_" + String(random(0xffff), HEX);
        
        if (mqttClient.connect(clientId.c_str())) {
            Serial.println(" Connecté !");
        } else {
            Serial.print(" Échec, rc=");
            Serial.print(mqttClient.state());
            Serial.println(" Retentative dans 3 secondes...");
            delay(3000);
        }
    }
}

void setup() {
    Serial.begin(115200);
    while(!Serial);

    // 1. Initialisation Réseau
    setupWiFi();
    mqttClient.setServer(MQTT_SERVER, MQTT_PORT);

    // 2. Initialisation TFLite Micro
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(g_sentinel_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[-] Erreur : Schema TFLite incompatible !");
        while(1);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize, error_reporter);

    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[-] Échec AllocateTensors");
        while(1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    // 3. Initialisation Capteur MPU9250
    Wire.begin();
    if (!mpu.setup(0x68)) {
        Serial.println("[-] Erreur MPU9250 : Vérifier les branchements (I2C 0x68)");
        while(1);
    }

    Serial.println("\n=========================================");
    Serial.println(">>> SENTINELLE EDGE AI + MQTT : READY <<<");
    Serial.println("=========================================\n");
}

void loop() {
    if (!mqttClient.connected()) {
        reconnectMQTT();
    }
    mqttClient.loop();

    if (mpu.update()) {
        // Calcul de la magnitude d'accélération
        float magnitude = sqrt(pow(mpu.getAccX(), 2) + pow(mpu.getAccY(), 2) + pow(mpu.getAccZ(), 2));

        // Buffer glissant
        mag_buffer[buffer_index % WINDOW_SIZE] = magnitude;
        buffer_index++;

        if (buffer_index >= WINDOW_SIZE) {
            float sum = 0, max_val = 0;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                sum += mag_buffer[i];
                if (mag_buffer[i] > max_val) max_val = mag_buffer[i];
            }
            float mean_val = sum / WINDOW_SIZE;

            float sq_sum = 0;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                sq_sum += pow(mag_buffer[i] - mean_val, 2);
            }
            float std_val = sqrt(sq_sum / WINDOW_SIZE);

            // Inférence TFLite [max, mean]
            input->data.f[0] = max_val;
            input->data.f[1] = mean_val;

            if (interpreter->Invoke() == kTfLiteOk) {
                float p_norm  = output->data.f[0];
                float p_shock = output->data.f[1];
                float p_fric  = output->data.f[2];

                // Logique Hybride de Classification
                String state_str = "NORMAL";
                
                if (std_val > 0.30 || max_val > 2.5 || p_shock > 0.60) {
                    state_str = "CHOC";
                }
                else if (p_fric > 0.60 || std_val > 0.08) {
                    state_str = "FROTTEMENT";
                }

                // 4. Publication MQTT (Format JSON)
                String payload = "{";
                payload += "\"state\":\"" + state_str + "\",";
                payload += "\"std\":" + String(std_val, 3) + ",";
                payload += "\"max_val\":" + String(max_val, 2) + ",";
                payload += "\"mean_val\":" + String(mean_val, 2) + ",";
                payload += "\"p_norm\":" + String(p_norm * 100, 1) + ",";
                payload += "\"p_fric\":" + String(p_fric * 100, 1) + ",";
                payload += "\"p_shock\":" + String(p_shock * 100, 1);
                payload += "}";

                mqttClient.publish(MQTT_TOPIC, payload.c_str());

                // Affichage Console pour Débogage
                Serial.print("[MQTT Output] ");
                Serial.println(payload);
            }
        }
    }
    delay(30); // ~33 Hz
}