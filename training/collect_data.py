import paho.mqtt.client as mqtt
import json
import csv
import os
import sys
import time
from datetime import datetime

# --- CONFIGURATION ---
MQTT_BROKER = "10.47.53.10"
MQTT_TOPIC = "industrie/vibrations"
BASE_DIR = r"D:/PROJET_SENTINELLE/DONNEES_VIBRATIONS"

modes = {
    "normal": {"file": "mes_vibrations_normales.csv", "id": 0, "desc": "Steady State (Baseline)"},
    "shock": {"file": "chocs.csv", "id": 1, "desc": "Impact & Anomaly Detection"},
    "friction": {"file": "frottement.csv", "id": 2, "desc": "Continuous Friction/Wear"}
}

# --- HUMAN CHECK ---
selected_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "normal"

if selected_mode not in modes:
    print(f"\n[!] Whoops! '{selected_mode}' is not a valid mode.")
    print(f"[?] Please use: normal, shock, or friction.")
    sys.exit(1)

TARGET_FILE = os.path.join(BASE_DIR, modes[selected_mode]["file"])
LABEL_ID = modes[selected_mode]["id"]

# --- SETUP DIRECTORY ---
os.makedirs(BASE_DIR, exist_ok=True)

# Initialize file with headers if it's the first time
if not os.path.exists(TARGET_FILE):
    with open(TARGET_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "mean", "max", "std", "label"])
        print(f"[+] Created new file: {modes[selected_mode]['file']}")

print(f"\n{'='*40}")
print(f"   SENTINEL DATA COLLECTOR - ACTIVE")
print(f"{'='*40}")
print(f"[*] Target Mode : {selected_mode.upper()} ({modes[selected_mode]['desc']})")
print(f"[*] Saving to   : {TARGET_FILE}")
print(f"[*] Status      : Waiting for ESP32 on {MQTT_BROKER}...")
print(f"{'='*40}\n")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[SUCCESS] MQTT Link Established! Capturing data now...\n")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"[ERROR] Connection failed (Code {rc}). Check your IP or Broker.")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        val_raw = payload.get("vibration", 0.0)
        
        # Fallback stats if ESP32 only sends raw
        val_mean = payload.get("mean", val_raw)
        val_max  = payload.get("max", val_raw)
        val_std  = payload.get("std", 0.0)

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # THE SECURE WRITE (Flush & Sync)
        with open(TARGET_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, val_mean, val_max, val_std, LABEL_ID])
            f.flush()
            os.fsync(f.fileno()) # Forces Windows to write to Disk D immediately
            
        # Human-friendly progress bar
        print(f"[RECORDING] {timestamp} | Value: {val_raw:>6.2f} m/s² | Mode: {selected_mode.upper()}", end="\r")
        
    except Exception as e:
        print(f"\n[!] Data Error: {e}")

# --- MQTT EXECUTION ---
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print(f"\n\n[INFO] Session ended by user.")
    print(f"[OK] Data safely stored in {TARGET_FILE}")
    sys.exit(0)