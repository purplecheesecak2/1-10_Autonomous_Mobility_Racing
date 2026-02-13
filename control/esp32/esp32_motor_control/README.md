# ESP32 Motor Control

Jetson으로부터 Serial 명령을 받아 모터를 제어하는 ESP32 코드

## 시스템 구조

```
Jetson (ROS2) → Serial → ESP32 → Motor Driver → DC Motor
                                → Servo → Steering
```

## 기능

- ✅ Serial 통신 (115200 baud, CSV 형식)
- ✅ 조향 제어 (Servo)
- ✅ 속도 제어 (DC Motor PWM)
- ✅ PID 제어 (옵션: 엔코더 피드백)
- ✅ Timeout 안전 기능
- ✅ 범위 제한

## 하드웨어 연결

### 필수 연결

| ESP32 핀 | 연결 대상 | 설명 |
|---------|----------|------|
| GPIO 25 | Motor Driver PWM | 속도 제어 |
| GPIO 26 | Motor Driver IN1 | 방향 제어 1 |
| GPIO 27 | Motor Driver IN2 | 방향 제어 2 |
| GPIO 18 | Servo Signal | 조향 제어 |
| GND | Common Ground | 공통 접지 |
| VIN/5V | 전원 | ESP32 전원 |

### 옵션 (엔코더 사용 시)

| ESP32 핀 | 연결 대상 | 설명 |
|---------|----------|------|
| GPIO 34 | Encoder A | 속도 측정 |
| GPIO 35 | Encoder B | 속도 측정 |

## 소프트웨어 설정

### PlatformIO로 업로드

```bash
cd esp32_motor_control

# 빌드
pio run

# 업로드
pio run --target upload

# 시리얼 모니터
pio device monitor
```

### Arduino IDE로 업로드

1. `src/main.cpp`를 `esp32_motor_control.ino`로 복사
2. Arduino IDE에서 열기
3. 보드 선택: ESP32 Dev Module
4. 라이브러리 설치: ESP32Servo
5. 업로드

## 설정 (코드 수정)

### 핀 번호 변경 (main.cpp 상단)

```cpp
#define MOTOR_PWM_PIN       25    // 모터 PWM 핀
#define MOTOR_DIR_PIN1      26    // 방향 1
#define MOTOR_DIR_PIN2      27    // 방향 2
#define SERVO_PIN           18    // 서보 핀
```

### Servo 범위 조정

```cpp
#define SERVO_MIN_ANGLE     60     // 최소 각도
#define SERVO_MAX_ANGLE     120    // 최대 각도
#define SERVO_CENTER        90     // 중앙 (직진)
```

### PID 튜닝

```cpp
#define KP                  50.0   // 비례 게인
#define KI                  10.0   // 적분 게인
#define KD                  5.0    // 미분 게인
```

### 속도-PWM 매핑

```cpp
float speedToPWM(float speed_mps) {
    // 1 m/s ≈ 85 PWM (실제 차량에 맞게 조정!)
    float pwm = speed_mps * 85.0;
    return constrain(pwm, -MAX_PWM, MAX_PWM);
}
```

## 입력 프로토콜

CSV 형식: `"steering,speed\n"`

### 예시

```
10.5,0.5    # 조향 10.5도, 속도 0.5 m/s
-15.0,1.2   # 조향 -15도 (왼쪽), 속도 1.2 m/s
0.0,0.0     # 직진, 정지
```

## 안전 기능

1. **Timeout**: 500ms 이상 명령 없으면 자동 정지
2. **범위 제한**: 조향각, 속도 제한
3. **Dead Zone**: 낮은 PWM 값 무시 (모터 보호)

## 테스트

### Serial 모니터로 테스트

```bash
# PlatformIO 모니터
pio device monitor

# 또는 screen
screen /dev/ttyUSB0 115200

# 명령 입력
10.5,0.5
-15.0,1.0
0.0,0.0
```

## 엔코더 피드백 활성화

1. `platformio.ini` 수정:
```ini
build_flags =
    -D USE_ENCODER_FEEDBACK
```

2. 엔코더 연결 (GPIO 34, 35)

3. `updateSpeedMeasurement()` 함수 수정 (실제 엔코더 스펙에 맞춰)

## 튜닝 가이드

### 1. Servo 중앙 찾기

```cpp
steeringServo.write(90);  // 값 조정하여 직진 찾기
```

### 2. 속도 매핑 조정

실제 주행 테스트로 `speedToPWM()` 함수 조정

### 3. PID 튜닝 (엔코더 사용 시)

1. KI, KD = 0으로 설정
2. KP만 증가시켜가며 진동 발생 지점 찾기
3. KP를 60%로 줄이고 KI 추가
4. KD 추가 (진동 감쇄)

## 문제 해결

### 모터가 안 돌아가요
- PWM 연결 확인
- Motor Driver 전원 확인
- `speedToPWM()` 값 확인 (Serial 출력)

### 조향이 반대로 움직여요
- Servo 연결 확인
- `map()` 함수 파라미터 반대로 변경

### Timeout이 계속 발생해요
- Jetson과 Serial 연결 확인
- Baud rate 일치 확인 (115200)
- 케이블 불량 확인
