import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

DATA_DIR = r"D:\PROJET_SENTINELLE\DONNEES_VIBRATIONS"

# Regroupement des fichiers par état physique réel (3 classes)
CLASS_GROUPS = {
    0: {'name': 'Normal',     'files': ['repos_statique.csv', 'mes_vibrations_normales.csv']},
    1: {'name': 'Frottement', 'files': ['deplacement_marche.csv', 'frottement.csv']},
    2: {'name': 'Choc',       'files': ['mouvement_brusque_choc.csv', 'chocs.csv']}
}

COLUMN_NAMES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

def load_file(filepath):
    """Charge un CSV de manière flexible quel que soit le format."""
    try:
        df = pd.read_csv(filepath)
        if not set(COLUMN_NAMES).issubset(df.columns):
            df = pd.read_csv(filepath, sep=None, engine='python')
        if not set(COLUMN_NAMES).issubset(df.columns):
            df = pd.read_csv(filepath, header=None, sep=None, engine='python').iloc[:, :6]
            df.columns = COLUMN_NAMES
            
        df = df[COLUMN_NAMES]
        for col in COLUMN_NAMES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    except Exception:
        return None

# 1. Chargement et étiquetage
dataframes = []
target_names = []

for label, group in CLASS_GROUPS.items():
    target_names.append(group['name'])
    for filename in group['files']:
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            df_temp = load_file(file_path)
            if df_temp is not None and len(df_temp) > 0:
                df_temp['label'] = label
                dataframes.append(df_temp)
                print(f"Chargé : {filename} ({len(df_temp)} lignes -> Classe {label}: {group['name']})")

df_all = pd.concat(dataframes, ignore_index=True)

# 2. Fenêtrage temporel et extraction de features
def extract_features(df, window_size=10, overlap=5):
    features, labels = [], []
    for label_val in df['label'].unique():
        sub_df = df[df['label'] == label_val][COLUMN_NAMES].values
        for start in range(0, len(sub_df) - window_size + 1, window_size - overlap):
            window = sub_df[start:start + window_size]
            
            means = np.mean(window, axis=0)
            stds = np.std(window, axis=0)
            mins = np.min(window, axis=0)
            maxs = np.max(window, axis=0)
            
            acc_mag = np.mean(np.linalg.norm(window[:, :3], axis=1))
            gyro_mag = np.mean(np.linalg.norm(window[:, 3:], axis=1))
            
            features.append(np.hstack([means, stds, mins, maxs, acc_mag, gyro_mag]))
            labels.append(label_val)
            
    return np.array(features), np.array(labels)

X, y = extract_features(df_all, window_size=10, overlap=5)

# 3. Entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Évaluation
y_pred = model.predict(X_test_scaled)

print("\n=== RAPPORT DE CLASSIFICATION ===")
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Sauvegarde
joblib.dump(model, os.path.join(DATA_DIR, 'model_sentinelle.joblib'))
joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler_sentinelle.joblib'))
print(f"\nModèle et Scaler sauvegardés dans : {DATA_DIR}")