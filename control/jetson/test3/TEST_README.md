# Test3 테스트 가이드

## 빌드

```bash
cd /home/parkbc/MIRU/IEVE
colcon build --packages-select test3 --symlink-install
source install/setup.bash
```

## 테스트 방법

### 방법 1: 자동 패턴 테스트 (추천)

5가지 패턴을 자동으로 테스트합니다:
1. 직진 (0도, 0.5 m/s) - 3초
2. 좌회전 (-15도, 0.3 m/s) - 3초
3. 우회전 (+15도, 0.3 m/s) - 3초
4. 급회전 (-25도, 0.8 m/s → 자동 감속) - 3초
5. 정지 (0도, 0.0 m/s) - 3초

```bash
ros2 launch test3 test.launch.py
```

**확인 사항:**
- ✅ 3초마다 패턴 변경
- ✅ ESP32에 명령 전송 확인
- ✅ 서보/모터 동작 확인
- ✅ 급회전 시 자동 감속 확인

---

### 방법 2: 단순 고정 명령 테스트

고정된 steering과 speed로 테스트:

```bash
# 직진 0.5 m/s
ros2 launch test3 test.launch.py simple_mode:=true test_steering:=0.0 test_speed:=0.5

# 우회전 10도, 0.3 m/s
ros2 launch test3 test.launch.py simple_mode:=true test_steering:=10.0 test_speed:=0.3

# 좌회전 -15도, 0.4 m/s
ros2 launch test3 test.launch.py simple_mode:=true test_steering:=-15.0 test_speed:=0.4
```

---

### 방법 3: 별도 터미널에서 테스트

Control 노드만 실행하고 수동으로 명령 보내기:

**Terminal 1: Control 노드**
```bash
cd /home/parkbc/MIRU/IEVE
source install/setup.bash
ros2 launch test3 control.launch.py
```

**Terminal 2: Test Publisher**
```bash
source install/setup.bash
ros2 run test3 test_publisher
```

**Terminal 3: 수동 명령 (옵션)**
```bash
source install/setup.bash
ros2 topic pub /desired_steering_angle std_msgs/Float32 "data: 10.0" -r 10
ros2 topic pub /target_speed std_msgs/Float32 "data: 0.5" -r 10
```

---

## ESP32 모니터링

ESP32가 명령을 받는지 확인:

```bash
cd /home/parkbc/esp32_motor_control
pio device monitor

# 또는
screen /dev/ttyUSB0 115200
```

**예상 출력:**
```
Received: steering=0.00 speed=0.50
Received: steering=-15.00 speed=0.30
Received: steering=15.00 speed=0.30
...
```

---

## Timeout 테스트

Test publisher를 중단하고 0.5초 후 자동 정지 확인:

```bash
# Ctrl+C로 test_publisher 중단
# → Control 노드가 "Planning timeout" 출력
# → ESP32로 명령 전송 중단
# → 수동 조이스틱 제어 가능
```

---

## 문제 해결

### Serial 포트 오류
```bash
ls /dev/ttyUSB* /dev/ttyACM*
sudo chmod 666 /dev/ttyUSB0
```

### 빌드 오류
```bash
pip3 install pyserial
colcon build --packages-select test3 --symlink-install
```

### Topic 확인
```bash
ros2 topic list
ros2 topic echo /desired_steering_angle
ros2 topic echo /target_speed
```
