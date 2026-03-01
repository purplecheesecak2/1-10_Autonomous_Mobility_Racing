/*
 * ESP32 Motor Control with Adaptive PID
 * Receives commands from Jetson via Serial
 * Controls motor driver with adaptive PID speed control
 */

#include <ESP32Servo.h>

// ========== PIN CONFIGURATION ==========
#define MOTOR_PWM_PIN       25    // DC motor PWM
#define MOTOR_DIR_PIN1      27    // Motor direction 1
#define MOTOR_DIR_PIN2      26    // Motor direction 2
#define SERVO_PIN           18    // Steering servo
#define ENCODER_PIN_A       34    // Speed encoder (optional)
#define ENCODER_PIN_B       35    // Speed encoder (optional)

// ========== PWM CONFIGURATION ==========
#define PWM_CHANNEL         4
#define PWM_FREQ            1000   // 1kHz
#define PWM_RESOLUTION      8      // 8-bit (0-255)

// ========== PARAMETERS ==========
#define SERIAL_BAUD         115200
#define TIMEOUT_MS          500    // Stop if no command for 500ms
#define CONTROL_RATE_HZ     50     // 50Hz control loop

// Steering limits (degrees)
#define SERVO_MIN_ANGLE     60     // Adjust to your servo
#define SERVO_MAX_ANGLE     120
#define SERVO_CENTER        85

// Speed limits
#define MAX_SPEED_MPS       3.0
#define MIN_SPEED_MPS       -3.0   // Negative = reverse
#define MAX_PWM             255
#define MIN_PWM             0

// ========== ENCODER CONFIGURATION ==========
// TODO: 캘리브레이션 필요
// 측정 방법: 바퀴를 정확히 1회전시키고 시리얼 모니터의 "ENC_RAW:" 값을 확인
// ENCODER_PPR = 그 값으로 교체
#define ENCODER_PPR           139        // 캘리브레이션 완료 (5회전 측정: 695펄스 ÷ 5)
#define WHEEL_CIRCUMFERENCE_M 0.3456f   // π × 0.11m (바퀴 지름 11cm)

// ========== ADAPTIVE PID CONFIGURATION ==========
// Base PID gains (tune these first!)
#define BASE_KP             50.0
#define BASE_KI             10.0
#define BASE_KD             5.0

// Speed-based gain scaling thresholds
#define LOW_SPEED_THRESHOLD   1.0   // m/s - 이 속도 이하는 저속
#define HIGH_SPEED_THRESHOLD  2.5   // m/s - 이 속도 이상은 고속

// Speed-based gain multipliers (실차 테스트로 조정)
#define LOW_SPEED_KP_SCALE    1.3   // 저속: 빠른 응답
#define LOW_SPEED_KI_SCALE    1.2
#define LOW_SPEED_KD_SCALE    0.8

#define HIGH_SPEED_KP_SCALE   0.7   // 고속: 안정성 우선
#define HIGH_SPEED_KI_SCALE   0.5
#define HIGH_SPEED_KD_SCALE   1.2

// Steering-based speed limiting (ESP32에서 처리)
#define STEERING_THRESHOLD    25.0  // degrees - 이 각도 이상이면 감쇄 시작
#define MAX_STEERING_ANGLE    45.0  // degrees
#define SPEED_REDUCTION_FACTOR 0.8  // 최대 조향 시 속도를 80%까지 제한

// ========== GLOBAL VARIABLES ==========
Servo steeringServo;

// Command from Jetson
float target_steering = 0.0;  // degrees
float target_speed = 0.0;     // m/s
float adjusted_target_speed = 0.0;  // 조향각 보정 후 목표 속도
unsigned long last_command_time = 0;

// Adaptive PID variables
float current_kp = BASE_KP;
float current_ki = BASE_KI;
float current_kd = BASE_KD;

// PID variables
float current_speed = 0.0;
float speed_error = 0.0;
float speed_error_sum = 0.0;
float speed_error_prev = 0.0;
float pid_output = 0.0;

// Encoder (if available)
volatile long encoder_count = 0;
float measured_speed = 0.0;
bool motor_going_forward = true;  // 방향 추적 (속도 부호 결정용)

// ========== FUNCTION PROTOTYPES ==========
void readSerialCommands();
void applySteering(float steering_deg);
void applyAdaptiveSpeedControl(float target_mps, float steering_deg);
void updateAdaptiveGains(float speed, float steering_deg);
float adjustSpeedForSteering(float target_mps, float steering_deg);
void setMotorPWM(float pwm_value);
void stopMotor();
void updateSpeedMeasurement();
float speedToPWM(float speed_mps);
void resetPID();
void IRAM_ATTR encoderISR();
void printAdaptiveStatus();

// ========== SETUP ==========
void setup() {
  // Serial communication
  Serial.begin(SERIAL_BAUD);
  Serial.println("ESP32 Adaptive Motor Control Started");

  // Motor pins
  pinMode(MOTOR_DIR_PIN1, OUTPUT);
  pinMode(MOTOR_DIR_PIN2, OUTPUT);

  // PWM setup
  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_PWM_PIN, PWM_CHANNEL);

  // Servo setup
  steeringServo.attach(SERVO_PIN);
  steeringServo.write(SERVO_CENTER);

  // Encoder pins (optional)
  pinMode(ENCODER_PIN_A, INPUT_PULLUP);
  pinMode(ENCODER_PIN_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_A), encoderISR, RISING);

  // Initialize
  stopMotor();

  Serial.println("Initialization complete. Waiting for commands...");
  Serial.println("Adaptive PID enabled:");
  Serial.println("  - Speed-based gain scaling");
  Serial.println("  - Steering-based speed limiting");
}

// ========== MAIN LOOP ==========
void loop() {
  static unsigned long last_control_time = 0;
  static unsigned long last_print_time = 0;
  unsigned long current_time = millis();

  // Read commands from Serial
  readSerialCommands();

  // Debug print (1Hz) - timeout과 무관하게 항상 출력
  if (current_time - last_print_time >= 1000) {
    last_print_time = current_time;
    printAdaptiveStatus();
  }

  // Check timeout
  if (current_time - last_command_time > TIMEOUT_MS) {
    // Timeout: stop motor
    stopMotor();
    steeringServo.write(SERVO_CENTER);
    resetPID();
    return;
  }

  // Control loop at fixed rate
  if (current_time - last_control_time >= (1000 / CONTROL_RATE_HZ)) {
    last_control_time = current_time;

    // Update speed measurement (from encoder or estimate)
    updateSpeedMeasurement();

    // Apply steering
    applySteering(target_steering);

    // Apply adaptive speed control
    applyAdaptiveSpeedControl(target_speed, target_steering);
  }
}

// ========== SERIAL COMMUNICATION ==========
void readSerialCommands() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    // Parse CSV: "steering,speed"
    int comma_index = line.indexOf(',');
    if (comma_index > 0) {
      String steering_str = line.substring(0, comma_index);
      String speed_str = line.substring(comma_index + 1);

      float new_steering = steering_str.toFloat();
      float new_speed = speed_str.toFloat();

      // Validate
      if (!isnan(new_steering) && !isnan(new_speed)) {
        target_steering = constrain(new_steering, -45.0, 45.0);
        target_speed = constrain(new_speed, MIN_SPEED_MPS, MAX_SPEED_MPS);
        last_command_time = millis();
      }
    }
  }
}

// ========== STEERING CONTROL ==========
void applySteering(float steering_deg) {
  // Convert steering angle to servo angle
  // steering_deg: -45 to +45 (left to right)
  // Symmetric mapping around SERVO_CENTER: ±SERVO_RANGE degrees
  #define SERVO_RANGE 30  // degrees each side from center

  float servo_angle = SERVO_CENTER + (steering_deg / 45.0) * SERVO_RANGE;
  servo_angle = constrain(servo_angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
  steeringServo.write((int)servo_angle);
}

// ========== ADAPTIVE SPEED CONTROL ==========
void applyAdaptiveSpeedControl(float target_mps, float steering_deg) {
  // 1. 조향각에 따라 목표 속도 조정
  adjusted_target_speed = adjustSpeedForSteering(target_mps, steering_deg);
  
  // 2. 현재 속도와 조향각에 따라 PID 게인 조정
  updateAdaptiveGains(measured_speed, steering_deg);

  // 3. PID 계산
  #ifdef USE_ENCODER_FEEDBACK
    // PID control with encoder feedback
    speed_error = adjusted_target_speed - measured_speed;
    speed_error_sum += speed_error * (1.0 / CONTROL_RATE_HZ);
    float speed_error_derivative = (speed_error - speed_error_prev) * CONTROL_RATE_HZ;
    speed_error_prev = speed_error;

    // Anti-windup
    speed_error_sum = constrain(speed_error_sum, -20.0, 20.0);

    // 피드포워드: 목표 속도를 PWM으로 직접 매핑 (즉각 응답)
    // PID는 실제 오차만 보정
    float feedforward = speedToPWM(adjusted_target_speed);

    // Adaptive PID + Feedforward
    pid_output = feedforward +
                 current_kp * speed_error +
                 current_ki * speed_error_sum +
                 current_kd * speed_error_derivative;
  #else
    // Open-loop: direct mapping from speed to PWM
    pid_output = speedToPWM(adjusted_target_speed);
  #endif

  // 4. 모터에 적용
  setMotorPWM(pid_output);
}

// ========== ADAPTIVE GAIN ADJUSTMENT ==========
void updateAdaptiveGains(float speed, float steering_deg) {
  float abs_speed = abs(speed);
  float abs_steering = abs(steering_deg);
  
  // 속도 기반 게인 스케일링
  float speed_scale_kp = 1.0;
  float speed_scale_ki = 1.0;
  float speed_scale_kd = 1.0;
  
  if (abs_speed < LOW_SPEED_THRESHOLD) {
    // 저속: 공격적인 게인
    speed_scale_kp = LOW_SPEED_KP_SCALE;
    speed_scale_ki = LOW_SPEED_KI_SCALE;
    speed_scale_kd = LOW_SPEED_KD_SCALE;
  } 
  else if (abs_speed > HIGH_SPEED_THRESHOLD) {
    // 고속: 보수적인 게인
    speed_scale_kp = HIGH_SPEED_KP_SCALE;
    speed_scale_ki = HIGH_SPEED_KI_SCALE;
    speed_scale_kd = HIGH_SPEED_KD_SCALE;
  }
  else {
    // 중속: 선형 보간
    float ratio = (abs_speed - LOW_SPEED_THRESHOLD) / 
                  (HIGH_SPEED_THRESHOLD - LOW_SPEED_THRESHOLD);
    speed_scale_kp = LOW_SPEED_KP_SCALE + 
                     ratio * (HIGH_SPEED_KP_SCALE - LOW_SPEED_KP_SCALE);
    speed_scale_ki = LOW_SPEED_KI_SCALE + 
                     ratio * (HIGH_SPEED_KI_SCALE - LOW_SPEED_KI_SCALE);
    speed_scale_kd = LOW_SPEED_KD_SCALE + 
                     ratio * (HIGH_SPEED_KD_SCALE - LOW_SPEED_KD_SCALE);
  }
  
  // 조향각 기반 게인 조정 (큰 조향각일 때 부드럽게)
  float steering_factor = 1.0;
  if (abs_steering > STEERING_THRESHOLD) {
    // 조향각이 클수록 게인을 더 낮춤 (0.7 ~ 1.0 범위)
    float steering_ratio = (abs_steering - STEERING_THRESHOLD) / 
                           (MAX_STEERING_ANGLE - STEERING_THRESHOLD);
    steering_ratio = constrain(steering_ratio, 0.0, 1.0);
    steering_factor = 1.0 - (steering_ratio * 0.1);  // 최대 10% 감소
  }
  
  // 최종 게인 계산
  current_kp = BASE_KP * speed_scale_kp * steering_factor;
  current_ki = BASE_KI * speed_scale_ki * steering_factor;
  current_kd = BASE_KD * speed_scale_kd * steering_factor;
}

// ========== STEERING-BASED SPEED LIMITING ==========
float adjustSpeedForSteering(float target_mps, float steering_deg) {
  float abs_steering = abs(steering_deg);
  
  // 조향각이 작으면 속도 제한 없음
  if (abs_steering < STEERING_THRESHOLD) {
    return target_mps;
  }
  
  // 조향각이 클수록 속도 제한 (선형 감소)
  float steering_ratio = (abs_steering - STEERING_THRESHOLD) / 
                         (MAX_STEERING_ANGLE - STEERING_THRESHOLD);
  steering_ratio = constrain(steering_ratio, 0.0, 1.0);
  
  // 최대 조향각일 때 SPEED_REDUCTION_FACTOR까지 감소
  float speed_limit_factor = 1.0 - (steering_ratio * (1.0 - SPEED_REDUCTION_FACTOR));
  
  float adjusted_speed = target_mps * speed_limit_factor;
  
  return adjusted_speed;
}

// ========== MOTOR DRIVER ==========
void setMotorPWM(float pwm_value) {
  // Determine direction
  bool forward = (pwm_value >= 0);
  int pwm_abs = (int)constrain(abs(pwm_value), MIN_PWM, MAX_PWM);

  if (pwm_abs < 10) {
    // Dead zone: stop
    stopMotor();
    return;
  }

  motor_going_forward = forward;

  if (forward) {
    digitalWrite(MOTOR_DIR_PIN1, HIGH);
    digitalWrite(MOTOR_DIR_PIN2, LOW);
  } else {
    digitalWrite(MOTOR_DIR_PIN1, LOW);
    digitalWrite(MOTOR_DIR_PIN2, HIGH);
  }

  ledcWrite(PWM_CHANNEL, pwm_abs);
}

void stopMotor() {
  digitalWrite(MOTOR_DIR_PIN1, LOW);
  digitalWrite(MOTOR_DIR_PIN2, LOW);
  ledcWrite(PWM_CHANNEL, 0);
}

// ========== SPEED MEASUREMENT ==========
void updateSpeedMeasurement() {
  // If encoder available, calculate speed from encoder counts
  // For now, just estimate or use open-loop

  #ifdef USE_ENCODER_FEEDBACK
    static unsigned long last_speed_update = 0;
    static long last_enc_raw = 0;
    unsigned long now = millis();
    float dt = (now - last_speed_update) / 1000.0;

    if (dt > 0.1) {  // 100ms마다 갱신
      long count = encoder_count;
      encoder_count = 0;
      last_speed_update = now;
      last_enc_raw = count;  // 캘리브레이션용

      float speed_magnitude = (float)abs(count) * WHEEL_CIRCUMFERENCE_M
                              / ((float)ENCODER_PPR * dt);
      measured_speed = motor_going_forward ? speed_magnitude : -speed_magnitude;
    }
  #else
    // Open-loop: assume we're at target speed (no feedback)
    measured_speed = target_speed;
  #endif
}

// Encoder interrupt
void IRAM_ATTR encoderISR() {
  encoder_count++;
}

// ========== UTILITY FUNCTIONS ==========
float speedToPWM(float speed_mps) {
  // Simple linear mapping (tune this for your motor!)
  // Assumes: 1 m/s ≈ 85 PWM (example, adjust for your setup)
  float pwm = speed_mps * 85.0;
  return constrain(pwm, -MAX_PWM, MAX_PWM);
}

void resetPID() {
  speed_error = 0.0;
  speed_error_sum = 0.0;
  speed_error_prev = 0.0;
  pid_output = 0.0;
}

// ========== DEBUG ==========
void printAdaptiveStatus() {
  Serial.println("=== Adaptive Control Status ===");
  Serial.print("Target Speed: ");
  Serial.print(target_speed, 2);
  Serial.print(" -> Adjusted: ");
  Serial.println(adjusted_target_speed, 2);
  
  Serial.print("Steering: ");
  Serial.print(target_steering, 1);
  Serial.println(" deg");
  
  Serial.print("Adaptive Gains - Kp: ");
  Serial.print(current_kp, 1);
  Serial.print(" Ki: ");
  Serial.print(current_ki, 1);
  Serial.print(" Kd: ");
  Serial.println(current_kd, 1);
  
  Serial.print("PWM Output: ");
  Serial.println(pid_output, 1);

  Serial.print("Measured Speed: ");
  Serial.print(measured_speed, 2);
  Serial.println(" m/s");
  Serial.println();

  // Jetson parseFeedbackLine() 이 파싱하는 형식
  Serial.print("SPEED:");
  Serial.println(measured_speed, 2);

  // 캘리브레이션용 원시 엔코더 카운트 출력 (캘리브레이션 완료 후 주석 처리)
  // Serial.print("ENC_RAW:");
  // Serial.println(encoder_count);
}
