import os

# --- SECTION MISE À JOUR ---
# On localise le dossier actuel (D:/PROJET_SENTINELLE/tools)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# On définit les nouveaux chemins relatifs
# On sort de 'tools' pour aller dans 'models'
tflite_model = os.path.join(BASE_DIR, "..", "models", "sentinel_v2.tflite")

# On sort de 'tools' pour aller dans 'firmware' (ou include)
output_header = os.path.join(BASE_DIR, "..", "firmware", "sentinel_core.h")
# ---------------------------

try:
    with open(tflite_model, 'rb') as f:
        bytes_data = f.read()

    with open(output_header, 'w') as f:
        f.write('#ifndef SENTINEL_CORE_H\n#define SENTINEL_CORE_H\n\n')
        f.write('const unsigned char g_sentinel_model_data[] __attribute__((aligned(16))) = {\n')
        
        for i, b in enumerate(bytes_data):
            f.write(f'0x{b:02x}, ' + ('\n' if (i + 1) % 12 == 0 else ''))
            
        f.write('\n};\n\n')
        f.write(f'const int g_sentinel_model_data_len = {len(bytes_data)};\n')
        f.write('#endif\n')

    print(f"SUCCÈS : Le header a été exporté vers {output_header}")

except FileNotFoundError:
    print(f"ERREUR : Le fichier {tflite_model} est introuvable. Vérifie l'emplacement !")