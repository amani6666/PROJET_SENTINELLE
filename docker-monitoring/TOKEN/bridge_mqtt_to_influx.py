import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

# --- CONFIGURATION ---
MQTT_BROKER = "10.207.229.10"
MQTT_TOPIC = "industrie/vibrations"
INFLUX_URL = "http://localhost:8086"
# Utilise ton token exact trouvé dans l'onglet Tokens
INFLUX_TOKEN = "Y1jkkjfZbho-gbPjTv8OnBYoJRXTcdNXqeVI7d0VBV903DRb6iRb6awFDoneWbGyGy76UPqPRVQUDbOU-prjIw=="
INFLUX_ORG = "SENTINEL_ORG"
INFLUX_BUCKET = "VIBRATIONS_DATA"

client_influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client_influx.write_api(write_options=SYNCHRONOUS)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        valeur = data["vibration"]
        
        
        point = Point("vibrations_moteur") \
            .tag("machine", data["machine"]) \
            .field("vibration", float(valeur))
        
        write_api.write(bucket=INFLUX_BUCKET, record=point)
        print(f"✅ Donnée stockée dans InfluxDB : {valeur}")
    except Exception as e:
        print(f"❌ Erreur de stockage : {e}")

client_mqtt = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client_mqtt.on_message = on_message
client_mqtt.connect(MQTT_BROKER, 1883, 60)
client_mqtt.subscribe(MQTT_TOPIC)

print("🚀 Script actif : En attente des données de l'ESP32...")
client_mqtt.loop_forever()    