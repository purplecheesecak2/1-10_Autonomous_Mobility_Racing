#include <Arduino.h>

const int pinENA = 14;
const int pinIN1 = 27;
const int pinIN2 = 26;

int pwm_value = 0;
bool forward = true;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  pinMode(pinENA, OUTPUT);
  pinMode(pinIN1, OUTPUT);
  pinMode(pinIN2, OUTPUT);

  digitalWrite(pinIN1, LOW);
  digitalWrite(pinIN2, LOW);
  analogWrite(pinENA, 0);

  Serial.println("ESP32 Motor Control Ready");
}

void loop() {
  // Serial 명령 체크
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    int comma = line.indexOf(',');
    if (comma > 0) {
      float speed = line.substring(comma + 1).toFloat();
      pwm_value = (int)(abs(speed) * 170.0);
      if (pwm_value > 255) pwm_value = 255;
      forward = (speed >= 0);

      Serial.print("Received: steering=");
      Serial.print(line.substring(0, comma));
      Serial.print(" speed=");
      Serial.print(speed);
      Serial.print(" PWM=");
      Serial.println(pwm_value);
    }
  }

  // 모터 제어 (항상 실행)
  if (pwm_value > 5) {
    if (forward) {
      digitalWrite(pinIN1, HIGH);
      digitalWrite(pinIN2, LOW);
    } else {
      digitalWrite(pinIN1, LOW);
      digitalWrite(pinIN2, HIGH);
    }
    analogWrite(pinENA, pwm_value);
  } else {
    digitalWrite(pinIN1, LOW);
    digitalWrite(pinIN2, LOW);
    analogWrite(pinENA, 0);
  }
  delay(50);
}
