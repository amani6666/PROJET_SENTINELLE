import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- configuration Dynamique des Chemins (GSoC Style) ---
# On détecte où se trouve le script (D:/PROJET_SENTINELLE/training)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# On remonte d'un cran pour atteindre la racine du projet
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# On définit les dossiers cibles de manière relative
DATA_DIR = os.path.join(BASE_DIR, "DONNEES_VIBRATIONS")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Sécurité : Créer le dossier models s'il n'existe pas
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# 1. Professional Data Loading & Normalization
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
data_list = []

if not all_files:
    print(f"[-] Erreur : Aucun fichier CSV trouvé dans {DATA_DIR}")
    exit()

for f in all_files:
    temp_df = pd.read_csv(f)
    temp_df.columns = temp_df.columns.str.strip().str.lower()
    data_list.append(temp_df)

df = pd.concat(data_list, ignore_index=True).fillna(0)
print(f"[+] Fusion Success. Dataset Size: {len(df)} rows.")

# 2. Map Features
try:
    X = df[['max', 'mean']].values
    raw_labels = df['label'].values.astype(int)
    y = tf.keras.utils.to_categorical(raw_labels, num_classes=3)
except KeyError as e:
    print(f"[-] Critical Mapping Error: {e}")
    print(f"[?] Current Headers: {list(df.columns)}")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 4. High-Precision Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[+] Training Kernel: Active...")
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# 5. Export to models/ folder
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# On enregistre dans le nouveau dossier 'models'
output_path = os.path.join(MODELS_DIR, 'sentinel_v2.tflite')
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"[SUCCESS] Model exported to: {output_path}")