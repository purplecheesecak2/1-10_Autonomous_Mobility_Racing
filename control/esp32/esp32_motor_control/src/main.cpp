/*
 * ESP32 Motor Control
 * Receives commands from Jetson via Serial
 * Controls motor driver with PID speed control
 */

#include <ESP32Servo.h>

// ========== PIN CONFIGURATION ==========
#define MOTOR_PWM_PIN       25    // DC motor PWM
#define MOTOR_DIR_PIN1      26    // Motor direction 1
#define MOTOR_DIR_PIN2      27    // Motor direction 2
#define SERVO_PIN           18    // Steering servo
#define ENCODER_PIN_A       34    // Speed encoder (optional)
#define ENCODER_PIN_B       35    // Speed encoder (optional)

// ========== PWM CONFIGURATION ==========
#define PWM_CHANNEL         0
#define PWM_FREQ            1000   // 1kHz
#define PWM_RESOLUTION      8      // 8-bit (0-255)

// ========== PARAMETERS ==========
#define SERIAL_BAUD         115200
#define TIMEOUT_MS          500    // Stop if no command for 500ms
#define CONTROL_RATE_HZ     50     // 50Hz control loop

// Steering limits (degrees)
#define SERVO_MIN_ANGLE     60     // Adjust to your servo
#define SERVO_MAX_ANGLE     120
#define SERVO_CENTER        90

// Speed limits
#define MAX_SPEED_MPS       3.0
#define MIN_SPEED_MPS       -3.0   // Negative = reverse
#define MAX_PWM             255
#define MIN_PWM             0

// PID gains (tune these!)
#define KP                  50.0
#define KI                  10.0
#define KD                  5.0

// ========== GLOBAL VARIABLES ==========
Servo steeringServo;

// Command from Jetson
float target_steering = 0.0;  // degrees
float target_speed = 0.0;     // m/s
unsigned long last_command_time = 0;

// PID variables
float current_speed = 0.0;
float speed_error = 0.0;
float speed_error_sum = 0.0;
float speed_error_prev = 0.0;
float pid_output = 0.0;

// Encoder (if available)
volatile long encoder_count = 0;
float measured_speed = 0.0;

// ========== FUNCTION PROTOTYPES ==========
void readSerialCommands();
void applySteering(float steering_deg);
void applySpeedControl(float target_mps);
void setMotorPWM(float pwm_value);
void stopMotor();
void updateSpeedMeasurement();
float speedToPWM(float speed_mps);
void resetPID();
void IRAM_ATTR encoderISR();

// ========== SETUP ==========
void setup() {
  // Serial communication
  Serial.begin(SERIAL_BAUD);
  Serial.println("ESP32 Motor Control Started");

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
}

// ========== MAIN LOOP ==========
void loop() {
  static unsigned long last_control_time = 0;
  unsigned long current_time = millis();

  // Read commands from Serial
  readSerialCommands();

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

    // Apply speed control with PID
    applySpeedControl(target_speed);
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

        // Debug
        Serial.print("Received: steering=");
        Serial.print(target_steering);
        Serial.print(" speed=");
        Serial.println(target_speed);
      }
    }
  }
}

// ========== STEERING CONTROL ==========
void applySteering(float steering_deg) {
  // Convert steering angle to servo angle
  // steering_deg: -45 to +45 (left to right)
  // servo: SERVO_MIN_ANGLE to SERVO_MAX_ANGLE

  float servo_angle = map(steering_deg * 100, -4500, 4500,
                          SERVO_MIN_ANGLE * 100, SERVO_MAX_ANGLE * 100) / 100.0;

  servo_angle = constrain(servo_angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
  steeringServo.write((int)servo_angle);
}

// ========== SPEED CONTROL (PID) ==========
void applySpeedControl(float target_mps) {
  // Simple open-loop control (no encoder feedback)
  // If encoder available, use PID with feedback

  #ifdef USE_ENCODER_FEEDBACK
    // PID control with encoder
    speed_error = target_mps - measured_speed;
    speed_error_sum += speed_error * (1.0 / CONTROL_RATE_HZ);
    float speed_error_derivative = (speed_error - speed_error_prev) * CONTROL_RATE_HZ;
    speed_error_prev = speed_error;

    // Anti-windup
    speed_error_sum = constrain(speed_error_sum, -10.0, 10.0);

    pid_output = KP * speed_error + KI * speed_error_sum + KD * speed_error_derivative;
  #else
    // Open-loop: direct mapping from speed to PWM
    pid_output = speedToPWM(target_mps);
  #endif

  // Apply to motor
  setMotorPWM(pid_output);
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
    unsigned long now = millis();
    float dt = (now - last_speed_update) / 1000.0;

    if (dt > 0.1) {  // Update every 100ms
      // Calculate speed from encoder
      // measured_speed = (encoder_count * wheel_circumference) / (encoder_ppr * dt);
      encoder_count = 0;  // Reset
      last_speed_update = now;
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

// ========== DEBUG (optional) ==========
void printStatus() {
  Serial.print("Target: S=");
  Serial.print(target_steering);
  Serial.print(" V=");
  Serial.print(target_speed);
  Serial.print(" | Measured: V=");
  Serial.print(measured_speed);
  Serial.print(" | PWM=");
  Serial.println(pid_output);
}
