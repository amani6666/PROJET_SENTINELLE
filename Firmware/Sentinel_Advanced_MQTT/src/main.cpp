/* * Sentinel_V1_Final_Test.ino
 * Dedicated for Predictive Maintenance Project (INSAT - IMI)
 * Hardware: ESP32 DevKit + MPU9250 on I2C (Pins 21/22)
 * * Note: Using FreeRTOS is critical here. If I run MQTT on the same
 * core as the sensor sampling, the WiFi stack makes the 100Hz 
 * vibration sampling jitter. Isolated them to separate cores.
 */

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "mpu9250.h"
#include "model.h" 

// --- PINOUT & I2C ---
#define I2C_SDA 21 // Default ESP32 pins, but adding them explicitly just in case
#define I2C_SCL 22
#define MPU_ADDR 0x68

// --- NETWORK CONFIG ---
// INSAT Lab WiFi or my phone hotspot
const char* ssid = "test"; 
const char* password = "123456789";
const char* mqtt_server = "10.47.53.10"; // Local Broker IP

// --- SENSOR CALIBRATION ---
// Measured bias for my specific unit during static tests on my desk.
// Gravity at 9.81 was giving me 0.05 m/s2 when it should be zero.
const float GRAVITY_OFFSET = 0.052; 

MPU9250 mpu(Wire, MPU_ADDR);
WiFiClient espClient;
PubSubClient client(espClient);
Eloquent::ML::Port::RandomForest classifier;

// --- PROCESSING BUFFER ---
const int WINDOW_SIZE = 20; // 50 was crashing the stack memory, 20 is stable
float samples[WINDOW_SIZE];
int head = 0;
volatile int current_state = 0; // 0=OK, 1=SHOCK, 2=FRICTION
volatile float last_rms = 0.0;

// Feature Extraction: Standard logic for vibration analysis (Mean, Peak, StdDev)
void get_features(float* out_features) {
    float sum = 0, peak = 0;
    for(int i=0; i<WINDOW_SIZE; i++) {
        sum += samples[i];
        if(samples[i] > peak) peak = samples[i];
    }
    float avg = sum / WINDOW_SIZE;
    float variance = 0;
    for(int i=0; i<WINDOW_SIZE; i++) variance += pow(samples[i] - avg, 2);
    float dev = sqrt(variance / WINDOW_SIZE);

    out_features[0] = avg; 
    out_features[1] = peak;
    out_features[2] = dev; 
}

// TASK: High-Frequency Sampling & Inference (Pinned to Core 1)
// Priority 5 (High) to ensure we get a deterministic 100Hz sampling.
void SamplerTask(void * arg) {
    for(;;) {
        if (mpu.readSensor() > 0) {
            float x = mpu.getAccelX_mss();
            float y = mpu.getAccelY_mss();
            float z = mpu.getAccelZ_mss();
            
            // Raw Magnitude - standard 9.806m/s2 - My specific offset
            last_rms = abs(sqrt(x*x + y*y + z*z) - 9.806) - GRAVITY_OFFSET;
            if (last_rms < 0) last_rms = 0;

            samples[head++] = last_rms;

            if (head >= WINDOW_SIZE) {
                float vec[3];
                get_features(vec);
                current_state = classifier.predict(vec);
                head = 0;
            }
        } else {
            // Force I2C re-init. The sensor hangs a lot when the motor is running.
            // Added this after a lot of frustrating testing.
            Serial.println("Warning: MPU I2C failure. Re-initiating Wire.");
            Wire.begin(I2C_SDA, I2C_SCL);
        }
        // Fixed 10ms delay (using FreeRTOS tick delay) for ~100Hz sampling.
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }
}

// TASK: Network/MQTT & WiFi management (Pinned to Core 0)
// Priority 1 (Low). Runs on the app core to handle network overhead.
void CommsTask(void * arg) {
    for(;;) {
        if (client.connected()) {
            char buf[128];
            const char* label = (current_state == 0) ? "NORMAL" : (current_state == 1 ? "ANOMALY_SHOCK" : "ANOMALY_FRICTION");
            
            // Clean JSON string for the Node-RED / Grafana dashboard on my laptop
            snprintf(buf, sizeof(buf), "{\"vib\":%.3f,\"state\":%d,\"tag\":\"%s\"}", last_rms, current_state, label);
            client.publish("factory/node_01/telemetry", buf);
        }
        vTaskDelay(pdMS_TO_TICKS(500)); // Telemetry is enough at 2Hz.
    }
}

void setup() {
    Serial.begin(115200);
    Wire.begin(I2C_SDA, I2C_SCL);
    // 400kHz I2C because 100kHz was too slow for 100Hz reading with multithreading
    Wire.setClock(400000); 

    if (mpu.begin() < 0) {
        Serial.println("Hardware Init Failed! Check MPU connections.");
        while(1) delay(10); // Loop forever to prevent weird crashes
    }

    WiFi.begin(ssid, password);
    client.setServer(mqtt_server, 1883);

    // Multithreading Setup. Core 1 for critical math, Core 0 for boring WiFi.
    xTaskCreatePinnedToCore(SamplerTask, "Inference", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(CommsTask, "MQTT", 4096, NULL, 1, NULL, 0);
    
    Serial.println("System Initialized on Dual Cores.");
}

void loop() {
    // Keep the core MQTT loop running in the main loop, pinned to Core 0 by default.
    if (!client.connected() && WiFi.status() == WL_CONNECTED) {
        if (client.connect("ESP32_Sentinel_IMI_Node01")) {
            Serial.println("Broker Connected");
        }
    }
    client.loop();
    delay(200); 
}