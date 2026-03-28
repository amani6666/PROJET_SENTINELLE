import time
import random
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# --- CONFIGURATION ---
# Remplace par TON Token que tu as mis dans tokens.txt
TOKEN = "tdT9zbrAAyMnMglcICjXoE8dlVMEar--gQjsrLZT_Zk9KPaNQgRlAg0n3jLJME6soyaXEtV8591gyE5HoK59Bg==" 
ORG = "sentinel_ai"
BUCKET = "vibrations_data"
# On utilise localhost car le script Python tourne sur Windows, pas dans Docker
URL = "http://localhost:8086" 

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

print("🚀 Simulation Sentinel AI lancée... (Ctrl+C pour stopper)")

try:
    while True:
        # On simule un score d'anomalie (entre 0 et 1)
        # Si > 0.8, c'est une alerte !
        valeur_vibration = random.uniform(0.1, 0.95)
        
        point = Point("vibration_analysis") \
            .tag("machine_id", "moteur_pompe_01") \
            .field("anomaly_score", valeur_vibration)
        
        write_api.write(bucket=BUCKET, org=ORG, record=point)
        
        print(f"📡 Donnée envoyée : Score={valeur_vibration:.2f}")
        time.sleep(1) # Envoi toutes les secondes

except KeyboardInterrupt:
    print("\n🛑 Simulation arrêtée.")
finally:
    client.close()