# Sentinel-V1: Edge-AI Predictive Maintenance System

**High-Performance Vibration Monitoring and Anomaly Classification using TinyML and ESP32 Dual-Core RTOS**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technical Architecture](#2-technical-architecture)
3. [Fault Classification](#3-fault-classification)
4. [Project Structure](#4-project-structure)
5. [Operational Pipeline](#5-operational-pipeline)
6. [Hardware Requirements](#6-hardware-requirements)
7. [Software Requirements](#7-software-requirements)
8. [Getting Started](#8-getting-started)
9. [Model Training](#9-model-training)
10. [IIoT Integration](#10-iiot-integration)
11. [Performance](#11-performance)
12. [Academic Context](#12-academic-context)

---

## 1. Project Overview

Sentinel-V1 is an end-to-end Industrial IoT (IIoT) solution engineered for the predictive maintenance of rotary machinery. The system bridges the gap between high-frequency raw mechanical signals and real-time actionable intelligence by deploying optimized machine learning models directly on the edge, requiring no cloud inference infrastructure.

The system identifies three critical mechanical failure modes — Shocks, Friction, and Bearing Wear — at the source, enabling maintenance teams to intervene before catastrophic equipment failure occurs.

The project specifically addresses two fundamental challenges in embedded AI:

- **Jitter Management** — ensuring timing determinism in a concurrent multi-task environment.
- **Sampling Determinism** — maintaining a stable 100 Hz acquisition loop regardless of ongoing network activity.

---

## 2. Technical Architecture

### 2.1 Dual-Core RTOS Design

Sentinel-V1 is built on a FreeRTOS-based dual-core architecture on the ESP32, designed to enforce strict real-time constraints through physical task isolation.

**Core 1 — Deterministic Sensing and Inference**

Dedicated exclusively to high-speed I2C communication with the MPU9250 IMU and real-time inference using TensorFlow Lite Micro. This core operates on a fixed 100 Hz interrupt-driven loop and is fully shielded from any network-related latency. Zero sampling gaps are guaranteed during WiFi or MQTT activity.

**Core 0 — Communication and Telemetry**

Manages the WiFi stack, MQTT client, and OTA firmware update handler. All outbound data transmission is handled asynchronously through a thread-safe queue, ensuring the sensing loop on Core 1 is never blocked or delayed.

### 2.2 Memory Architecture

Custom memory planners minimize the SRAM footprint of the TFLite Micro interpreter, enabling complex model execution to coexist with a robust RTOS heap on a resource-constrained device. The final quantized model operates entirely within the ESP32's available SRAM.

### 2.3 Signal Processing Layer

Before inference, raw IMU data is transformed into three time-domain features computed in real time on-device:

| Feature | Purpose |
|---|---|
| Root Mean Square (RMS) | Energy tracking and vibration intensity |
| Peak-to-Peak Amplitude | Mechanical shock and impact detection |
| Standard Deviation | Pattern stability and signal regularity |

---

## 3. Fault Classification

The model is trained to distinguish four operational states of rotary machinery:

| Class | Description |
|---|---|
| Normal | Steady-state baseline — healthy operation |
| Shock | High-amplitude transient impacts |
| Friction | Continuous surface wear and frictional degradation |
| Bearing Wear | Progressive bearing fatigue signatures |

---

## 4. Project Structure

The repository maintains a strict separation between embedded firmware, machine learning artifacts, and data engineering utilities.

**DONNEES\_VIBRATIONS/** — Comprehensive vibration data repository containing raw acquisition files per fault class and the final labeled training dataset (`mondataset_pro.csv`).

**Firmware/Sentinel\_Advanced\_MQTT/** — Production-grade IIoT firmware built with PlatformIO. Contains the RTOS task partitioning logic (`main.cpp`), the static C++ model weights header (`model.h`), the hardware abstraction layer (`sentinel_core.h`), and the optimized TFLite Micro kernel library.

**Firmware/Smart\_Sentinel\_AI/** — Specialized inference development module used for neural network interpreter configuration and standalone validation outside the full IIoT stack.

**models/** — Serialized model artifacts, including the original Keras source model (`sentinel_model.h5`), the standard TFLite export (`model.tflite`), and the final pruned and quantized edge model (`sentinel_v2.tflite`).

**tools/** — Deployment and calibration infrastructure: a TFLite-to-C++ array converter (`convertisseur.py`), an automated firmware deployment script (`export_to_esp32.py`), and the MPU9250 calibration utility implementing static gravity and zero-g bias compensation.

**training/** — Machine learning development pipeline: high-speed data acquisition (`collect_data.py`), a normalization and labeling pipeline (`fusion_data.py`), and the model training and INT8 quantization script (`train_v2.py`).

---

## 5. Operational Pipeline

### Step 1 — Sensor Calibration

Before any data collection, the MPU9250 Calibration utility must be flashed and executed. This step applies zero-g bias compensation to all accelerometer and gyroscope axes, maximizing the Signal-to-Noise Ratio of subsequent acquisitions. Resulting offsets must be written back into `sentinel_core.h`.

### Step 2 — Data Acquisition

The `collect_data.py` script acquires labeled vibration samples from the device via serial or MQTT and writes them to the corresponding CSV files under `DONNEES_VIBRATIONS/`. Each fault class must be recorded separately under controlled and reproducible mechanical conditions.

### Step 3 — Data Preparation

`fusion_data.py` consolidates raw per-class CSV files into a single normalized and labeled training dataset. This step handles class balancing, feature normalization, and train/validation splitting.

### Step 4 — Model Training and Quantization

`train_v2.py` trains the neural network on the prepared dataset, exports the Keras model, converts it to TFLite format, and applies INT8 post-training quantization. The quantized model is validated to confirm inference accuracy remains above 90%.

### Step 5 — Firmware Deployment

`convertisseur.py` converts the `.tflite` binary into a C++ byte array header ready for static inclusion in the firmware. `export_to_esp32.py` then automates the PlatformIO build and flash cycle.

### Step 6 — Live Monitoring

Once deployed, the firmware publishes JSON-formatted classification results to the configured MQTT broker. The Node-RED flow processes incoming messages and feeds the Grafana dashboard for real-time visualization and alert management.

---

## 6. Hardware Requirements

| Component | Specification |
|---|---|
| Microcontroller | ESP32 (dual-core Xtensa LX6, 240 MHz) |
| IMU Sensor | MPU9250 (3-axis accelerometer + 3-axis gyroscope) |
| Communication Interface | I2C (400 kHz Fast Mode) |
| Power Supply | 3.3 V regulated |
| Wireless Connectivity | 2.4 GHz WiFi (802.11 b/g/n) |

---

## 7. Software Requirements

| Tool | Purpose |
|---|---|
| PlatformIO (VS Code extension) | Firmware build system and flash toolchain |
| Python 3.8 or higher | Training pipeline and deployment scripts |
| TensorFlow 2.x / Keras | Model training and TFLite conversion |
| TensorFlow Lite Micro | On-device inference runtime |
| MQTT Broker (e.g., Mosquitto) | Message transport layer |
| Node-RED | IIoT data flow orchestration |
| Grafana | Real-time monitoring and alerting dashboard |

Install Python dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn paho-mqtt pyserial
```

---

## 8. Getting Started

### 8.1 Clone the Repository

```bash
git clone https://github.com/<your-username>/sentinel-v1.git
cd sentinel-v1
```

### 8.2 Calibrate the Sensor

Open the `tools/MPU9250_Calibration/` project in PlatformIO, flash it to the ESP32, and record the reported bias offsets. Update the calibration constants in `Firmware/sentinel_core.h` before proceeding.

### 8.3 Configure the Firmware

Edit `Firmware/Sentinel_Advanced_MQTT/src/main.cpp` and set the network and broker parameters:

```cpp
#define WIFI_SSID       "your_network_ssid"
#define WIFI_PASSWORD   "your_network_password"
#define MQTT_BROKER_IP  "192.168.x.x"
#define MQTT_PORT       1883
#define MQTT_TOPIC      "sentinel/classification"
```

### 8.4 Build and Flash

```bash
cd Firmware/Sentinel_Advanced_MQTT
pio run --target upload
```

Alternatively, use the automated deployment script:

```bash
python tools/export_to_esp32.py
```

---

## 9. Model Training

### 9.1 Collect Training Data

```bash
python training/collect_data.py --class normal   --output DONNEES_VIBRATIONS/mes_vibrations_normales.csv
python training/collect_data.py --class shock    --output DONNEES_VIBRATIONS/chocs.csv
python training/collect_data.py --class friction --output DONNEES_VIBRATIONS/frottement.csv
```

### 9.2 Prepare the Dataset

```bash
python training/fusion_data.py
```

This generates the consolidated file `DONNEES_VIBRATIONS/mondataset_pro.csv`.

### 9.3 Train and Quantize

```bash
python training/train_v2.py
```

Expected outputs:

- `models/sentinel_model.h5` — Keras source model
- `models/model.tflite` — Standard TFLite export
- `models/sentinel_v2.tflite` — INT8 quantized edge-optimized model

### 9.4 Convert to C++ Header

```bash
python tools/convertisseur.py \
  --input  models/sentinel_v2.tflite \
  --output Firmware/Sentinel_Advanced_MQTT/src/model.h
```

---

## 10. IIoT Integration

Sentinel-V1 publishes classification results as JSON payloads over MQTT:

```json
{
  "timestamp": 1720000000,
  "label": "FRICTION",
  "confidence": 0.97,
  "features": {
    "rms": 0.342,
    "peak_to_peak": 1.128,
    "std_dev": 0.089
  }
}
```

The Node-RED flow subscribes to the configured topic, applies threshold-based alerting logic, and forwards processed data to Grafana. Dashboard panels expose real-time fault distribution, confidence trends, and maintenance alert history.

---

## 11. Performance

| Metric | Value |
|---|---|
| Sampling Rate | 100 Hz (deterministic) |
| Inference Accuracy | > 90% on held-out test set |
| Model Size (quantized) | < 20 KB (INT8) |
| Inference Latency | < 10 ms per cycle |
| Core 1 Timing Jitter | < 1 ms under full WiFi load |

---

## 12. Academic Context

| Field | Detail |
|---|---|
| Institution | National Institute of Applied Sciences and Technology (INSAT), Tunisia |
| Specialization | Instrumentation and Industrial Maintenance (IMI) |
| Developer | Amani Liouane |
| Academic Level | 4th-Year Engineering Student |
| Domain | Embedded Systems, TinyML, Industrial IoT |

---

*All rights reserved by the author. This project is developed for academic and research purposes at INSAT, Tunisia. Contact the developer for collaboration or usage inquiries.*
