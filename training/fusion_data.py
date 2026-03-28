import pandas as pd
import os

# --- PATH CONFIGURATION (Disk D) ---
DATA_DIR = r'D:\PROJET_SENTINELLE\DONNEES_VIBRATIONS'
OUTPUT_CSV = r'D:\PROJET_SENTINELLE\mondataset_pro.csv'

# Mapping files to labels: 0=Normal, 1=Shock, 2=Friction
FILE_MAPPING = {
    "mes_vibrations_normales.csv": 0,
    "chocs.csv": 1,
    "frottement.csv": 2
}

def merge_vibration_data():
    data_frames = []
    print("--- Sentinel Data Fusion Tool ---")

    for file_name, label in FILE_MAPPING.items():
        file_path = os.path.join(DATA_DIR, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Assign the target label for supervised learning
            df['label'] = label
            data_frames.append(df)
            print(f" [+] Successfully loaded: {file_name} (Label {label})")
        else:
            print(f" [!] Warning: {file_name} not found in {DATA_DIR}")

    if data_frames:
        final_dataset = pd.concat(data_frames, ignore_index=True)
        # Save to Disk D for the training script
        final_dataset.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[SUCCESS] Master dataset saved: {OUTPUT_CSV}")
        print(f"Total samples merged: {len(final_dataset)}")
    else:
        print("\n[ERROR] No data files were found. Check your directory paths.")

if __name__ == "__main__":
    merge_vibration_data()