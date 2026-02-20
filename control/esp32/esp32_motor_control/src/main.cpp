/*
 * ESP32 Motor Control with Adaptive PID
 * Receives commands from Jetson via Serial
 * Controls motor driver with adaptive PID speed control
 *
 * [수정사항]
 * 1. SPEED 피드백 전송 추가 (readSerialFeedback 대응)
 * 2. 속도 감쇄 제거 → Jetson에서 일괄 처리
 * 3. Serial.readStringUntil() 블로킹 → 논블로킹 방식으로 교체
 */

#include <ESP32Servo.h>

// ========== PIN CONFIGURATION ==========
#define MOTOR_PWM_PIN       25
#define MOTOR_DIR_PIN1      27
#define MOTOR_DIR_PIN2      26
#define SERVO_PIN           18
#define ENCODER_PIN_A       34
#define ENCODER_PIN_B       35

// ========== PWM CONFIGURATION ==========
#define PWM_CHANNEL         4
#define PWM_FREQ            1000
#define PWM_RESOLUTION      8      // 8-bit (0-255)

// ========== PARAMETERS ==========
#define SERIAL_BAUD         115200
#define TIMEOUT_MS          500
#define CONTROL_RATE_HZ     50
#define FEEDBACK_RATE_HZ    10     // 피드백 전송 주기 (10Hz로 충분)

// Steering limits
#define SERVO_MIN_ANGLE     60
#define SERVO_MAX_ANGLE     120
#define SERVO_CENTER        90

// Speed limits
#define MAX_SPEED_MPS       3.0
#define MIN_SPEED_MPS       -3.0
#define MAX_PWM             255
#define MIN_PWM             0

// ========== ADAPTIVE PID CONFIGURATION ==========
#define BASE_KP             50.0
#define BASE_KI             10.0
#define BASE_KD             5.0

#define LOW_SPEED_THRESHOLD   1.0
#define HIGH_SPEED_THRESHOLD  2.5

#define LOW_SPEED_KP_SCALE    1.3
#define LOW_SPEED_KI_SCALE    1.2
#define LOW_SPEED_KD_SCALE    0.8

#define HIGH_SPEED_KP_SCALE   0.7
#define HIGH_SPEED_KI_SCALE   0.5
#define HIGH_SPEED_KD_SCALE   1.2

// [수정] 조향 기반 게인 감쇄만 유지 (속도 감쇄는 Jetson으로 이관)
#define STEERING_THRESHOLD    15.0
#define MAX_STEERING_ANGLE    45.0

// ========== SOFT START CONFIGURATION ==========
#define SOFTSTART_INITIAL_PWM   50
#define SOFTSTART_STEP          10
#define SOFTSTART_DELAY_MS      10

// ========== GLOBAL VARIABLES ==========
Servo steeringServo;

float target_steering = 0.0;
float target_speed = 0.0;
unsigned long last_command_time = 0;

float current_kp = BASE_KP;
float current_ki = BASE_KI;
float current_kd = BASE_KD;

float speed_error = 0.0;
float speed_error_sum = 0.0;
float speed_error_prev = 0.0;
float pid_output = 0.0;

volatile long encoder_count = 0;
float measured_speed = 0.0;

// [수정] 논블로킹 시리얼 파싱용 버퍼
String serial_buffer = "";

// [소프트스타트] 현재 PWM 상태 추적
int current_pwm_state = 0;

// ========== FUNCTION PROTOTYPES ==========
void readSerialCommands();
void applySteering(float steering_deg);
void applyAdaptiveSpeedControl(float target_mps);
void updateAdaptiveGains(float speed, float steering_deg);
void setMotorPWM(float pwm_value);
void stopMotor();
void updateSpeedMeasurement();
float speedToPWM(float speed_mps);
void resetPID();
void sendSpeedFeedback();           // [추가]
void IRAM_ATTR encoderISR();
void printAdaptiveStatus();

// ========== SETUP ==========
void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println("ESP32 Adaptive Motor Control Started");

  pinMode(MOTOR_DIR_PIN1, OUTPUT);
  pinMode(MOTOR_DIR_PIN2, OUTPUT);

  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_PWM_PIN, PWM_CHANNEL);

  steeringServo.attach(SERVO_PIN);
  steeringServo.write(SERVO_CENTER);

  pinMode(ENCODER_PIN_A, INPUT_PULLUP);
  pinMode(ENCODER_PIN_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_A), encoderISR, RISING);

  stopMotor();

  Serial.println("Initialization complete. Waiting for commands...");
}

// ========== MAIN LOOP ==========
void loop() {
  static unsigned long last_control_time = 0;
  static unsigned long last_feedback_time = 0;
  static unsigned long last_print_time = 0;
  unsigned long current_time = millis();

  // [수정] 논블로킹 시리얼 읽기
  readSerialCommands();

  // Timeout 체크
  if (current_time - last_command_time > TIMEOUT_MS) {
    stopMotor();
    steeringServo.write(SERVO_CENTER);
    resetPID();
    return;
  }

  // 제어 루프 (50Hz)
  if (current_time - last_control_time >= (1000 / CONTROL_RATE_HZ)) {
    last_control_time = current_time;

    updateSpeedMeasurement();
    applySteering(target_steering);
    applyAdaptiveSpeedControl(target_speed);  // [수정] 조향 인자 제거
  }

  // [추가] 속도 피드백 전송 (10Hz)
  if (current_time - last_feedback_time >= (1000 / FEEDBACK_RATE_HZ)) {
    last_feedback_time = current_time;
    sendSpeedFeedback();
  }

  // 디버그 출력 (1Hz)
  if (current_time - last_print_time >= 1000) {
    last_print_time = current_time;
    printAdaptiveStatus();
  }
}

// ========== SERIAL COMMUNICATION ==========
// [수정] 논블로킹 방식: available() 로 1바이트씩 읽어 \n 감지 시 파싱
void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      serial_buffer.trim();

      int comma_index = serial_buffer.indexOf(',');
      if (comma_index > 0) {
        float new_steering = serial_buffer.substring(0, comma_index).toFloat();
        float new_speed    = serial_buffer.substring(comma_index + 1).toFloat();

        if (!isnan(new_steering) && !isnan(new_speed)) {
          target_steering = constrain(new_steering, -45.0, 45.0);
          target_speed    = constrain(new_speed, MIN_SPEED_MPS, MAX_SPEED_MPS);
          last_command_time = millis();
        }
      }

      serial_buffer = "";  // 버퍼 초기화
    } else {
      serial_buffer += c;

      // 버퍼 오버플로 방지
      if (serial_buffer.length() > 64) {
        serial_buffer = "";
      }
    }
  }
}

// ========== SPEED FEEDBACK ==========
// [추가] Jetson readSerialFeedback()의 "SPEED:x.xx\n" 포맷에 대응
void sendSpeedFeedback() {
  Serial.print("SPEED:");
  Serial.println(measured_speed, 2);
}

// ========== STEERING CONTROL ==========
void applySteering(float steering_deg) {
  float servo_angle = map(steering_deg * 100, -4500, 4500,
                          SERVO_MIN_ANGLE * 100, SERVO_MAX_ANGLE * 100) / 100.0;
  servo_angle = constrain(servo_angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
  steeringServo.write((int)servo_angle);
}

// ========== ADAPTIVE SPEED CONTROL ==========
// [수정] adjustSpeedForSteering 제거 → Jetson이 이미 처리한 속도를 그대로 사용
void applyAdaptiveSpeedControl(float target_mps) {
  updateAdaptiveGains(measured_speed, target_steering);

  #ifdef USE_ENCODER_FEEDBACK
    speed_error = target_mps - measured_speed;
    speed_error_sum += speed_error * (1.0 / CONTROL_RATE_HZ);
    float speed_error_derivative = (speed_error - speed_error_prev) * CONTROL_RATE_HZ;
    speed_error_prev = speed_error;

    speed_error_sum = constrain(speed_error_sum, -10.0, 10.0);

    pid_output = current_kp * speed_error +
                 current_ki * speed_error_sum +
                 current_kd * speed_error_derivative;
  #else
    pid_output = speedToPWM(target_mps);
  #endif

  setMotorPWM(pid_output);
}

// ========== ADAPTIVE GAIN ADJUSTMENT ==========
// [수정] 속도 감쇄 로직 제거, 게인 조정만 담당
void updateAdaptiveGains(float speed, float steering_deg) {
  float abs_speed    = abs(speed);
  float abs_steering = abs(steering_deg);

  // 속도 기반 게인 스케일링
  float speed_scale_kp, speed_scale_ki, speed_scale_kd;

  if (abs_speed < LOW_SPEED_THRESHOLD) {
    speed_scale_kp = LOW_SPEED_KP_SCALE;
    speed_scale_ki = LOW_SPEED_KI_SCALE;
    speed_scale_kd = LOW_SPEED_KD_SCALE;
  } else if (abs_speed > HIGH_SPEED_THRESHOLD) {
    speed_scale_kp = HIGH_SPEED_KP_SCALE;
    speed_scale_ki = HIGH_SPEED_KI_SCALE;
    speed_scale_kd = HIGH_SPEED_KD_SCALE;
  } else {
    float ratio = (abs_speed - LOW_SPEED_THRESHOLD) /
                  (HIGH_SPEED_THRESHOLD - LOW_SPEED_THRESHOLD);
    speed_scale_kp = LOW_SPEED_KP_SCALE + ratio * (HIGH_SPEED_KP_SCALE - LOW_SPEED_KP_SCALE);
    speed_scale_ki = LOW_SPEED_KI_SCALE + ratio * (HIGH_SPEED_KI_SCALE - LOW_SPEED_KI_SCALE);
    speed_scale_kd = LOW_SPEED_KD_SCALE + ratio * (HIGH_SPEED_KD_SCALE - LOW_SPEED_KD_SCALE);
  }

  // 조향각 기반 게인 감쇄 (큰 조향 시 부드럽게)
  float steering_factor = 1.0;
  if (abs_steering > STEERING_THRESHOLD) {
    float steering_ratio = (abs_steering - STEERING_THRESHOLD) /
                           (MAX_STEERING_ANGLE - STEERING_THRESHOLD);
    steering_ratio  = constrain(steering_ratio, 0.0, 1.0);
    steering_factor = 1.0 - (steering_ratio * 0.3);
  }

  current_kp = BASE_KP * speed_scale_kp * steering_factor;
  current_ki = BASE_KI * speed_scale_ki * steering_factor;
  current_kd = BASE_KD * speed_scale_kd * steering_factor;
}

// ========== MOTOR DRIVER ==========
void setMotorPWM(float pwm_value) {
  bool forward = (pwm_value >= 0);
  int pwm_abs = (int)constrain(abs(pwm_value), MIN_PWM, MAX_PWM);

  if (pwm_abs < 10) {
    stopMotor();
    return;
  }

  digitalWrite(MOTOR_DIR_PIN1, forward ? HIGH : LOW);
  digitalWrite(MOTOR_DIR_PIN2, forward ? LOW  : HIGH);

  // [소프트스타트] 정지 상태에서 시작할 때 220까지 킥 후 목표 PWM으로 안정화
  if (current_pwm_state == 0) {
    int kick_pwm = max(pwm_abs, 220);
    for (int pwm = SOFTSTART_INITIAL_PWM; pwm <= kick_pwm; pwm += SOFTSTART_STEP) {
      ledcWrite(PWM_CHANNEL, pwm);
      delay(SOFTSTART_DELAY_MS);
    }
  }

  ledcWrite(PWM_CHANNEL, pwm_abs);
  current_pwm_state = pwm_abs;
}

void stopMotor() {
  digitalWrite(MOTOR_DIR_PIN1, LOW);
  digitalWrite(MOTOR_DIR_PIN2, LOW);
  ledcWrite(PWM_CHANNEL, 0);
  current_pwm_state = 0;  // [소프트스타트] 정지 시 상태 초기화
}

// ========== SPEED MEASUREMENT ==========
void updateSpeedMeasurement() {
  #ifdef USE_ENCODER_FEEDBACK
    static unsigned long last_speed_update = 0;
    unsigned long now = millis();
    float dt = (now - last_speed_update) / 1000.0;

    if (dt > 0.1) {
      // measured_speed = (encoder_count * wheel_circumference) / (encoder_ppr * dt);
      encoder_count = 0;
      last_speed_update = now;
    }
  #else
    measured_speed = target_speed;
  #endif
}

void IRAM_ATTR encoderISR() {
  encoder_count++;
}

// ========== UTILITY ==========
float speedToPWM(float speed_mps) {
  if (abs(speed_mps) < 0.01) return 0;
  float sign = (speed_mps > 0) ? 1.0 : -1.0;
  float pwm = 150.0 + (abs(speed_mps) / MAX_SPEED_MPS) * 70.0;
  return constrain(sign * pwm, -MAX_PWM, MAX_PWM);
}

void resetPID() {
  speed_error      = 0.0;
  speed_error_sum  = 0.0;
  speed_error_prev = 0.0;
  pid_output       = 0.0;
}

// ========== DEBUG ==========
void printAdaptiveStatus() {
  Serial.println("=== Adaptive Control Status ===");
  Serial.print("Target Speed: ");   Serial.println(target_speed, 2);
  Serial.print("Measured Speed: "); Serial.println(measured_speed, 2);
  Serial.print("Steering: ");       Serial.print(target_steering, 1); Serial.println(" deg");
  Serial.print("Kp: "); Serial.print(current_kp, 1);
  Serial.print(" Ki: "); Serial.print(current_ki, 1);
  Serial.print(" Kd: "); Serial.println(current_kd, 1);
  Serial.print("PWM Output: ");     Serial.println(pid_output, 1);
  Serial.println();
}
