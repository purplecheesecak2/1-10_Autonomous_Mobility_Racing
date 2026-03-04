import cv2
import os

# ── 설정 ──────────────────────────────────────────────────
CAMERA_ID   = 0              # USB 카메라 인덱스 (보통 0, 안되면 1 시도)
OUTPUT_PATH = "calib_record.mp4"
RESOLUTION  = (1280, 720)    # 실제 차량 전방 카메라와 동일한 해상도로 맞출 것
FPS         = 30
# ──────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)  # Jetson Linux는 V4L2 백엔드 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError(f"카메라를 열 수 없습니다. CAMERA_ID={CAMERA_ID}")

fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
writer  = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, RESOLUTION)

print("촬영 시작! 종료하려면 'q'를 누르세요.")
print("체커보드를 다양한 위치/각도로 천천히 이동해주세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    writer.write(frame)
    cv2.imshow("Recording (q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"저장 완료: {OUTPUT_PATH}")


# ── 프레임 자동 추출 ──────────────────────────────────────
EXTRACT_INTERVAL = 2.0        # 추출 간격 (초)
OUTPUT_DIR       = "calib_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap2   = cv2.VideoCapture(OUTPUT_PATH)
fps    = cap2.get(cv2.CAP_PROP_FPS)
step   = int(fps * EXTRACT_INTERVAL)  # N초마다 몇 프레임인지
count  = 0
saved  = 0

print(f"\n프레임 추출 시작 (간격: {EXTRACT_INTERVAL}초, step: {step}프레임)")

while True:
    ret, frame = cap2.read()
    if not ret:
        break

    if count % step == 0:
        path = os.path.join(OUTPUT_DIR, f"frame_{saved:04d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1

    count += 1

cap2.release()
print(f"추출 완료: {saved}장 → '{OUTPUT_DIR}/' 폴더")