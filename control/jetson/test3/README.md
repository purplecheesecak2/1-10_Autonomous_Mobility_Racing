# Jetson ESP32 Control Package (Python)

Jetson에서 자율주행 제어 명령을 받아 ESP32로 전송하는 ROS2 Python 패키지

## 시스템 구조

```
Planning Node → Jetson Control Node → ESP32 → Motor Driver
                (이 패키지)          (Serial)
```

## 기능

- 플래닝 명령 수신 (조향각, 속도)
- 입력 검증 (NaN/Inf 필터링)
- 조향각/속도 제한
- 조향각 필터링 (부드러운 제어)
- 급격한 조향 시 자동 감속
- Timeout 안전 기능
- ESP32로 Serial 통신 (CSV 형식)

## 설치 방법

### Python 의존성 설치
```bash
pip3 install pyserial
# 또는
pip3 install -r requirements.txt
```

### ROS2 패키지 빌드
```bash
# ROS2 workspace에 복사
cp -r jetson_esp32_control_pkg ~/ros2_ws/src/

# 빌드
cd ~/ros2_ws
colcon build --packages-select jetson_esp32_control_pkg --symlink-install

# Source
source install/setup.bash
```

**참고**: `--symlink-install` 사용 시 Python 코드 수정 후 재빌드 없이 바로 반영됩니다!

## 실행 방법

### 기본 실행
```bash
ros2 launch jetson_esp32_control_pkg control.launch.py
```

### Serial 포트 지정
```bash
ros2 launch jetson_esp32_control_pkg control.launch.py serial_port:=/dev/ttyACM0
```

### 파라미터 수정
`config/params.yaml` 파일 편집

## 입력 Topic

- `/desired_steering_angle` (std_msgs/Float32) - 목표 조향각 (degrees)
- `/target_speed` (std_msgs/Float32) - 목표 속도 (m/s)
- `/odom` (nav_msgs/Odometry) - 현재 오도메트리

## 출력 (Serial)

CSV 형식: `"steering,speed\n"`

예시: `"10.50,0.50\n"` (조향각 10.5도, 속도 0.5 m/s)

## Serial 포트 확인

```bash
# 연결된 포트 확인
ls /dev/ttyUSB* /dev/ttyACM*

# 권한 설정 (필요시)
sudo chmod 666 /dev/ttyUSB0
```

## 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| serial_port | /dev/ttyUSB0 | ESP32 Serial 포트 |
| baud_rate | 115200 | Serial 통신 속도 |
| max_steering | 30.0 | 최대 조향각 (degrees) |
| max_speed | 3.0 | 최대 속도 (m/s) |
| steering_filter_alpha | 0.3 | 조향 필터 강도 |
| timeout_threshold | 0.5 | Timeout 시간 (seconds) |

## 안전 기능

1. **Timeout 처리**: 명령이 0.5초 이상 없으면 ESP32로 아무것도 보내지 않음
2. **수동 제어 가능**: Planning 명령이 없을 때는 수동 조이스틱 제어 가능
3. **급격한 조향 감속**: 조향각 20도 이상 시 속도 50% 감소
