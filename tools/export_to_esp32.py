import tensorflow as tf
import os

# 1. Load the trained model from Disk D
model_path = 'D:/PROJET_SENTINELLE/sentinel_model.h5'
model = tf.keras.models.load_model(model_path)

# 2. Convert to TFLite (TinyML format)
print("[*] Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save the binary file
tflite_path = 'D:/PROJET_SENTINELLE/model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# 4. Generate the C++ Header (The "model.h")
print("[*] Generating C++ header for ESP32...")
hex_array = [f"0x{b:02x}" for b in tflite_model]
header_content = (
    "// Sentinel Edge AI - Generated Model\n"
    "// Move this file to your Arduino/PlatformIO project\n\n"
    f"const unsigned char sentinel_model_data[] = {{\n  "
    + ", ".join(hex_array) +
    "\n};\n"
    f"const unsigned int sentinel_model_data_len = {len(tflite_model)};\n"
)

header_path = 'D:/PROJET_SENTINELLE/model.h'
with open(header_path, 'w') as f:
    f.write(header_content)

print(f"\n[SUCCESS] Final model ready at: {header_path}")
print("[TIP] You can now include this in your ESP32 code!")