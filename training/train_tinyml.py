import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================== CONFIG ==================
DATA_DIR = r"D:\PROJET_SENTINELLE\DONNEES_VIBRATIONS"   # ← adapte ton chemin

FILES = {
    "repos_statique.csv": 0,              # Normal
    "deplacement_marche.csv": 1,          # Frottement
    "mouvement_brusque_choc.csv": 2,      # Choc
}

COLUMN_NAMES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE = 15
STEP = 5
# ============================================

def load_file(filepath):
    try:
        # Gestion flexible des séparateurs (; ou ,)
        df = pd.read_csv(filepath, sep=None, engine='python')
        df.columns = [c.strip().lower() for c in df.columns]
        
        if not set(COLUMN_NAMES).issubset(df.columns):
            df = pd.read_csv(filepath, header=None, sep=None, engine='python').iloc[:, :6]
            df.columns = COLUMN_NAMES
            
        df = df[COLUMN_NAMES]
        for col in COLUMN_NAMES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception as e:
        print(f"Erreur {filepath}: {e}")
        return None

# 1. Chargement + extraction des 6 features
X_list, y_list = [], []

for filename, label in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"[!] Fichier manquant : {filename}")
        continue
    
    df = load_file(path)
    if df is None or len(df) == 0:
        continue
    
    data = df[COLUMN_NAMES].values
    
    for start in range(0, len(data) - WINDOW_SIZE + 1, STEP):
        window = data[start:start + WINDOW_SIZE]
        
        # Magnitude Accélération
        acc_mag = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
        # Magnitude Gyroscope
        gyro_mag = np.sqrt(window[:, 3]**2 + window[:, 4]**2 + window[:, 5]**2)
        
        features = [
            np.mean(acc_mag),   # 0
            np.max(acc_mag),    # 1
            np.std(acc_mag),    # 2
            np.mean(gyro_mag),  # 3
            np.max(gyro_mag),   # 4
            np.std(gyro_mag),   # 5
        ]
        X_list.append(features)
        y_list.append(label)
    
    print(f"Chargé : {filename} → {len(data)} lignes brutes → label {label}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

print(f"\nTotal fenêtres : {len(X)}")
print(f"Répartition classes : {np.bincount(y)}")

# 2. Split + Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\n=== PARAMÈTRES DU SCALER (à noter) ===")
print("mean  =", [round(float(x), 6) for x in scaler.mean_])
print("scale =", [round(float(x), 6) for x in scaler.scale_])

# 3. Petit réseau de neurones (6 → 16 → 8 → 3)
model = keras.Sequential([
    layers.Input(shape=(6,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_s, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# 4. Évaluation
y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
print("\n=== RAPPORT DE CLASSIFICATION ===")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Frottement', 'Choc'], zero_division=0))
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# 5. Sauvegarde
model.save("sentinel_model.keras")
np.savez("scaler_params.npz", mean=scaler.mean_, scale=scaler.scale_)

# 6. Conversion TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("sentinel_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"\nTaille modèle TFLite : {os.path.getsize('sentinel_model.tflite')} octets")

# 7. Génération du header C++
def convert_to_c_array(tflite_path, header_path="sentinel_core.h"):
    with open(tflite_path, "rb") as f:
        data = f.read()
    
    with open(header_path, "w") as f:
        f.write("#ifndef SENTINEL_CORE_H\n")
        f.write("#define SENTINEL_CORE_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int g_sentinel_model_data_len = {len(data)};\n")
        f.write("alignas(16) const unsigned char g_sentinel_model_data[] = {\n")
        
        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("\n  ")
            f.write(f"0x{b:02x}, ")
        f.write("\n};\n\n#endif\n")
    
    print(f"Header généré : {header_path}")

convert_to_c_array("sentinel_model.tflite")
print("\n✅ Pipeline TinyML (6 features) terminé avec succès.")