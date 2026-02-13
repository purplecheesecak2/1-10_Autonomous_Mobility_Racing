# 하드웨어 세팅 가이드

## 필요한 부품

### 필수
- [ ] Jetson (Xavier, Nano 등)
- [ ] ESP32 개발보드
- [ ] DC Motor Driver (L298N, BTS7960, VNH5019 등)
- [ ] DC Motor (구동용)
- [ ] Servo Motor (조향용)
- [ ] 점퍼 케이블
- [ ] USB to Serial 케이블 (Jetson ↔ ESP32)
- [ ] 전원 (배터리 또는 전원 공급기)
  - ESP32: 5V
  - Motor: 12V (모터 스펙에 따라)
  - Servo: 5V

### 선택 (고급)
- [ ] Rotary Encoder (속도 측정용)
- [ ] 스위치/릴레이 (비상 정지)
- [ ] LED (상태 표시)

---

## 1. 전원 연결

### 전원 구성
```
배터리 (12V)
├─ Buck Converter (12V → 5V) → ESP32, Servo
└─ Motor Driver (12V) → DC Motor
```

### 주의사항 ⚠️
- **공통 접지 필수!** 모든 GND를 연결해야 합니다
- ESP32와 모터는 전원을 분리하되 GND는 공통
- Servo는 ESP32와 전원 공유 가능 (전류 확인 필요)

---

## 2. ESP32 연결

### 핀 배치 (코드 기준)

| 기능 | ESP32 핀 | 연결 대상 | 설명 |
|------|---------|----------|------|
| **모터 제어** |
| PWM | GPIO 25 | Motor Driver ENA/PWM | 속도 제어 |
| DIR1 | GPIO 26 | Motor Driver IN1 | 방향 제어 |
| DIR2 | GPIO 27 | Motor Driver IN2 | 방향 제어 |
| **조향 제어** |
| Servo | GPIO 18 | Servo Signal (Orange/Yellow) | 조향각 제어 |
| **Serial 통신** |
| TX | GPIO 1 (TX0) | Jetson RX | 데이터 송신 (옵션) |
| RX | GPIO 3 (RX0) | Jetson TX | 명령 수신 |
| **엔코더 (옵션)** |
| ENC_A | GPIO 34 | Encoder A | 속도 측정 |
| ENC_B | GPIO 35 | Encoder B | 속도 측정 |
| **전원** |
| VIN | 5V | Buck Converter 출력 | ESP32 전원 |
| GND | GND | 공통 접지 | 모든 GND 연결 |

---

## 3. Motor Driver 연결

### L298N 예시

| L298N 핀 | 연결 |
|---------|------|
| ENA | ESP32 GPIO 25 (PWM) |
| IN1 | ESP32 GPIO 26 |
| IN2 | ESP32 GPIO 27 |
| OUT1 | DC Motor + |
| OUT2 | DC Motor - |
| 12V | 배터리 + |
| GND | 공통 GND |
| 5V | (사용 안 함) |

### BTS7960 예시 (고출력)

| BTS7960 핀 | 연결 |
|-----------|------|
| RPWM | ESP32 GPIO 25 |
| LPWM | ESP32 GPIO 26 (또는 27과 연결) |
| R_EN | 5V (항상 활성) |
| L_EN | 5V (항상 활성) |
| R_IS / L_IS | NC (전류 감지, 옵션) |
| VCC | 5V (로직 전원) |
| GND | 공통 GND |
| B+ / B- | 배터리 + / - |
| M+ / M- | DC Motor + / - |

---

## 4. Servo Motor 연결

| Servo 선 | 색상 | 연결 |
|---------|------|------|
| Signal | Orange/Yellow | ESP32 GPIO 18 |
| VCC | Red | 5V |
| GND | Brown/Black | 공통 GND |

### Servo 전원 주의사항
- 큰 서보는 별도 5V 전원 필요 (ESP32 직접 연결 시 과전류)
- BEC (배터리 제거 회로) 사용 권장

---

## 5. Jetson ↔ ESP32 Serial 연결

### USB to Serial 방식 (권장)

```
Jetson USB ─ USB Cable ─ ESP32 USB
```

가장 간단! ESP32를 Jetson USB에 연결

**Jetson에서 확인:**
```bash
ls /dev/ttyUSB*  # 또는 /dev/ttyACM*
```

### UART 직접 연결 (고급)

```
Jetson TX (GPIO 14) ─────── ESP32 RX (GPIO 3)
Jetson RX (GPIO 15) ─────── ESP32 TX (GPIO 1)
Jetson GND ────────────────── ESP32 GND
```

**주의:** Jetson과 ESP32 전압 레벨 확인 (3.3V vs 5V)

---

## 6. 전체 연결도

```
                    ┌─────────────┐
                    │   Jetson    │
                    │  (제어 PC)   │
                    └──────┬──────┘
                           │ USB/Serial
                           │
                    ┌──────▼──────┐
                    │    ESP32    │
                    │             │
                    └─┬─────────┬─┘
                      │         │
              GPIO 25,26,27   GPIO 18
                      │         │
            ┌─────────▼─┐   ┌──▼───┐
            │   Motor   │   │ Servo│
            │  Driver   │   │      │
            └─────┬─────┘   └──────┘
                  │
            ┌─────▼─────┐
            │ DC Motor  │
            │  (구동)    │
            └───────────┘

         ┌──────────────┐
         │   Battery    │
         │    (12V)     │
         └──┬───────┬───┘
            │       │
         12V│    5V│(Buck Conv.)
            │       │
        Motor   ESP32/Servo
```

---

## 7. 테스트 순서

### Step 1: ESP32 단독 테스트
```bash
cd /home/parkbc/esp32_motor_control
pio run --target upload
pio device monitor
```

Serial 모니터에서 수동 입력:
```
0.0,0.5   # 직진, 0.5 m/s
10.0,0.3  # 우회전 10도, 0.3 m/s
```

**확인:**
- ✅ Servo 움직임
- ✅ Motor 회전

### Step 2: Serial 통신 테스트

**Jetson에서:**
```bash
echo "10.5,0.5" > /dev/ttyUSB0
```

**ESP32 모니터 확인:**
```
Received: steering=10.5 speed=0.5
```

### Step 3: 전체 시스템 테스트

```bash
cd /home/parkbc/MIRU/IEVE
source install/setup.bash
ros2 launch test3 test.launch.py
```

---

## 8. 핀 변경 방법

코드에서 핀 번호 수정:

**파일:** `/home/parkbc/esp32_motor_control/src/main.cpp`

```cpp
// 상단에서 핀 번호 변경
#define MOTOR_PWM_PIN       25    // 원하는 핀으로 변경
#define MOTOR_DIR_PIN1      26
#define MOTOR_DIR_PIN2      27
#define SERVO_PIN           18
```

수정 후 재업로드:
```bash
pio run --target upload
```

---

## 9. 문제 해결

### 모터가 안 돌아가요
1. 전원 확인 (배터리 전압, 연결)
2. GND 공통 연결 확인
3. Motor Driver LED 확인
4. PWM 신호 확인 (오실로스코프 또는 LED)

### Servo가 안 움직여요
1. 5V 전원 확인
2. Signal 선 연결 확인
3. Servo 중앙값 조정 (코드에서 `SERVO_CENTER` 변경)

### Serial 통신 안 돼요
1. USB 케이블 확인 (데이터 케이블인지, 충전 전용은 안 됨)
2. `/dev/ttyUSB*` 확인
3. 권한 확인: `sudo chmod 666 /dev/ttyUSB0`
4. Baud rate 일치 확인 (115200)

### ESP32가 리셋돼요
1. 전원 부족 → 더 큰 전원 공급기
2. 모터 노이즈 → 캐패시터 추가 (모터 단자에 0.1uF)
3. GND 분리 → 공통 GND 연결

---

## 10. 안전 수칙 ⚠️

- [ ] 비상 정지 스위치 설치
- [ ] 모터 테스트는 바퀴를 든 상태에서
- [ ] 배터리 역극성 방지 (다이오드 추가)
- [ ] 과전류 방지 (퓨즈 추가)
- [ ] 절연 테이프로 노출 전선 감싸기

---

## 참고: 추천 부품

| 부품 | 추천 모델 | 비고 |
|------|----------|------|
| Motor Driver | BTS7960 | 고출력 (43A) |
|              | L298N | 저가형 (2A) |
| Servo | MG996R | 금속 기어 |
|       | SG90 | 소형 (테스트용) |
| Buck Converter | LM2596 | 12V → 5V |
| ESP32 | ESP32-DevKit-C | 표준 |
|       | ESP32-WROOM-32 | |

---

질문이 있으면 언제든지 물어보세요!
