#include <Arduino.h>
#include <Wire.h>
#include <MPU9250.h>
#include <math.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "sentinel_core.h" 

MPU9250 mpu;

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  tflite::ErrorReporter* error_reporter = nullptr;

  // Optimisation de la mémoire pour ESP32
  constexpr int kArenaSize = 10240; // 10KB suffisent pour ce modèle
  uint8_t tensor_arena[kArenaSize] __attribute__((aligned(16)));

  // Fenêtrage statistique (Buffer glissant)
  const int WINDOW_SIZE = 15; 
  float mag_buffer[WINDOW_SIZE];
  int buffer_index = 0;
}

void setup() {
    Serial.begin(115200);
    while(!Serial);
    
    // 1. Initialisation TFLite Micro
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    
    model = tflite::GetModel(g_sentinel_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("[-] Model Schema Mismatch!");
        while(1);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize, error_reporter);
    
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[-] AllocateTensors Failed");
        while(1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    // 2. Initialisation Capteur MPU9250
    Wire.begin();
    if (!mpu.setup(0x68)) {
        Serial.println("[-] MPU9250 Error : Check wiring");
        while(1);
    }
    
    Serial.println("=========================================");
    Serial.println(">>> SMART SENTINEL AI : ONLINE");
    Serial.println(">>> INPUTS: [MAX, MEAN] | OUTPUTS: 3 CLASSES");
    Serial.println("=========================================");
}

void loop() {
    if (mpu.update()) {
        // Calcul Magnitude Accélération
        float magnitude = sqrt(pow(mpu.getAccX(), 2) + pow(mpu.getAccY(), 2) + pow(mpu.getAccZ(), 2));
        
        // Remplissage du buffer
        mag_buffer[buffer_index % WINDOW_SIZE] = magnitude;
        buffer_index++;

        // On attend d'avoir assez de données pour l'analyse
        if (buffer_index >= WINDOW_SIZE) {
            float sum = 0, max_val = 0;
            for (int i = 0; i < WINDOW_SIZE; i++) {
                sum += mag_buffer[i];
                if (mag_buffer[i] > max_val) max_val = mag_buffer[i];
            }
            float mean_val = sum / WINDOW_SIZE;

            // Calcul de l'écart-type (Standard Deviation) pour la logique "Guardrail"
            float sq_sum = 0;
            for (int i = 0; i < WINDOW_SIZE; i++) sq_sum += pow(mag_buffer[i] - mean_val, 2);
            float std_val = sqrt(sq_sum / WINDOW_SIZE);

            // 3. Préparation Input IA (Doit correspondre à l'ordre X = df[['max', 'mean']])
            input->data.f[0] = max_val;  // Feature 1
            input->data.f[1] = mean_val; // Feature 2

            // 4. Inférence
            if (interpreter->Invoke() == kTfLiteOk) {
                float p_norm = output->data.f[0];
                float p_shock = output->data.f[1];
                float p_fric = output->data.f[2];

                // --- LOGIQUE DE DÉCISION HYBRIDE (IA + SEUILS PHYSIQUES) ---
                String status = "STABLE (Normal)";
                
                // Priorité aux seuils critiques (Sécurité industrielle)
                if (std_val > 0.30 || max_val > 2.5) {
                    status = "!!! ALERTE CHOC DETECTÉ !!!";
                } 
                // Sinon on suit la prédiction de l'IA
                else if (p_fric > 0.70 || std_val > 0.08) {
                    status = "ANOMALIE : FROTTEMENT SUSPECT";
                }

                // 5. Monitoring
                Serial.print("STATS [M:" + String(mean_val, 2) + " Max:" + String(max_val, 2) + " Std:" + String(std_val, 2) + "] ");
                Serial.print("IA [N:" + String(p_norm*100, 0) + "% S:" + String(p_shock*100, 0) + "% F:" + String(p_fric*100, 0) + "%] ");
                Serial.println("-> " + status);
            }
        }
    }
    delay(30); // Fréquence d'échantillonnage (~33Hz)
}
