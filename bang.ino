#include <Servo.h>

Servo servoX;
Servo servoY;

int angleX = 90;
int angleY = 90;

void setup() {
    Serial.begin(9600);
    servoX.attach(9);
    servoY.attach(10); 
    servoX.write(angleX);
    servoY.write(angleY);
}

void loop() {
    if (Serial.available() > 0) {
        String data = Serial.readStringUntil('\n');
        int commaIndex = data.indexOf(',');
        if (commaIndex > 0) {
            int newAngleX = data.substring(0, commaIndex).toInt();
            int newAngleY = data.substring(commaIndex + 1).toInt();
            if (newAngleX >= 0 && newAngleX <= 180) {
                servoX.write(newAngleX);
                Serial.print("X: ");
                Serial.println(newAngleX);
            }
            if (newAngleY >= 0 && newAngleY <= 180) {
                servoY.write(newAngleY);
                Serial.print("Y: ");
                Serial.println(newAngleY);
            }
        }
    }
}
