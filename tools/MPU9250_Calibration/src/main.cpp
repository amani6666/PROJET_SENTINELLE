#include "mpu9250.h"

MPU9250 mpu(Wire, 0x68);

void setup() {
    Serial.begin(115200);
    Wire.begin(21, 22);
    delay(2000);

    Serial.println(">>> CALIBRAGE DU MPU9250...");
    Serial.println("Laissez le capteur parfaitement immobile à plat.");
    delay(5000);

    if (mpu.begin() < 0) {
        Serial.println("Erreur capteur");
        while(1);
    }

    // Calibrage des accelerometres et gyroscopes
    mpu.calibrateAccel();
    mpu.calibrateGyro();

    Serial.println(">>> CALIBRAGE TERMINÉ !");
    Serial.println("Recopiez ces valeurs si nécessaire :");
    Serial.print("Accel Bias X: "); Serial.println(mpu.getAccelBiasX_mss());
    Serial.print("Accel Bias Y: "); Serial.println(mpu.getAccelBiasY_mss());
    Serial.print("Accel Bias Z: "); Serial.println(mpu.getAccelBiasZ_mss());
}

void loop() {
    // Rien ici
}