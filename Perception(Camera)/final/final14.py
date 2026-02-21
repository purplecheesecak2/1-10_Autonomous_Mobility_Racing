#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge

class LaneDetectionFinal(Node):
    def __init__(self):
        super().__init__('lane_detection_final_node')
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다!")
            return
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.W, self.H = 640, 480
        self.roi_start = int(self.H * 0.65)

        self.canny_t1 = 50
        self.canny_t2 = 150

        self.target_offset_dir = 0
        self.current_lane_x    = self.W / 2

        # ★ 노란선 경계 이전값 (초기엔 전체 허용)
        self.prev_x_min = 0
        self.prev_x_max = self.W
        # 노란선 미감지 프레임 카운터 (30프레임 = 약 1초)
        self.yellow_lost_min = 0
        self.yellow_lost_max = 0
        self.YELLOW_RESET_FRAMES = 30

        cv2.namedWindow('Lane Detection')
        cv2.createTrackbar('Canny T1',  'Lane Detection', 50,  255,  self._noop)
        cv2.createTrackbar('Canny T2',  'Lane Detection', 150, 255,  self._noop)
        cv2.createTrackbar('ROI Start', 'Lane Detection', 65,  90,   self._noop)
        cv2.createTrackbar('White V',   'Lane Detection', 200, 255,  self._noop)
        cv2.createTrackbar('White S',   'Lane Detection', 30,  100,  self._noop)
        cv2.createTrackbar('Min Area',  'Lane Detection', 300, 3000, self._noop)
        cv2.createTrackbar('Fill W',    'Lane Detection', 15,  60,   self._noop)

        self.color_pub  = self.create_publisher(Image,   '/camera/color/image_raw', 1)
        self.lane_pub   = self.create_publisher(Image,   '/camera/lane/image_raw',  1)
        self.offset_pub = self.create_publisher(Float32, '/lane/offset',             1)
        self.create_subscription(Int32, '/lane/target', self._target_cb, 1)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("Lane Detection 시작")

    def _noop(self, _): pass
    def _target_cb(self, msg): self.target_offset_dir = int(msg.data)

    def _build_mask(self, frame, white_v, white_s, fill_w):
        roi = frame[self.roi_start:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        k   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        white = cv2.inRange(hsv, np.array([0, 0, white_v]), np.array([180, white_s, 255]))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,  k, iterations=2)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)

        yellow = cv2.inRange(hsv, np.array([10, 50, 80]), np.array([40, 255, 255]))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN,  k)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), self.canny_t1, self.canny_t2)
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(yellow))

        fill_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, fill_w), 3))
        edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, fill_k, iterations=2)

        white = cv2.bitwise_or(white, edges)
        return white, yellow, roi

    def _contour_bottom_cx(self, c):
        pts      = c[:, 0, :]
        max_y    = pts[:, 1].max()
        thresh_y = max_y - max(1, int((max_y - pts[:, 1].min()) * 0.1))
        return int(pts[pts[:, 1] >= thresh_y][:, 0].mean())

    def _yellow_x_bounds(self, yellow):
        """노란선 마스크 하단 절반에서 왼쪽/오른쪽 경계 x를 계산.
        한쪽만 보이면 그쪽만 갱신하고 반대쪽은 이전값 유지.
        일정 프레임 이상 안 보이면 자동으로 경계 리셋."""
        h   = yellow.shape[0]
        mid = self.W // 2
        bottom_half = yellow[h // 2:, :]
        cols = np.where(bottom_half.any(axis=0))[0]

        left_cols  = cols[cols < mid]
        right_cols = cols[cols >= mid]

        # 왼쪽 노란선: 보이면 갱신 및 카운터 리셋, 안 보이면 카운터 증가
        if len(left_cols) > 0:
            self.prev_x_min = int(left_cols.max())
            self.yellow_lost_min = 0
        else:
            self.yellow_lost_min += 1
            if self.yellow_lost_min >= self.YELLOW_RESET_FRAMES:
                self.prev_x_min = 0
                self.yellow_lost_min = 0

        # 오른쪽 노란선: 보이면 갱신 및 카운터 리셋, 안 보이면 카운터 증가
        if len(right_cols) > 0:
            self.prev_x_max = int(right_cols.min())
            self.yellow_lost_max = 0
        else:
            self.yellow_lost_max += 1
            if self.yellow_lost_max >= self.YELLOW_RESET_FRAMES:
                self.prev_x_max = self.W
                self.yellow_lost_max = 0

        return self.prev_x_min, self.prev_x_max

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("프레임 읽기 실패", throttle_duration_sec=5.0)
            return
        try:
            self.canny_t1  = cv2.getTrackbarPos('Canny T1',  'Lane Detection')
            self.canny_t2  = max(self.canny_t1+1, cv2.getTrackbarPos('Canny T2', 'Lane Detection'))
            roi_pct        = cv2.getTrackbarPos('ROI Start', 'Lane Detection')
            white_v        = cv2.getTrackbarPos('White V',   'Lane Detection')
            white_s        = cv2.getTrackbarPos('White S',   'Lane Detection')
            min_area       = max(50, cv2.getTrackbarPos('Min Area', 'Lane Detection'))
            fill_w         = cv2.getTrackbarPos('Fill W',    'Lane Detection')
            self.roi_start = int(self.H * roi_pct / 100)
            roi_h          = self.H - self.roi_start
            cx             = int(self.W / 2)

            white, yellow, roi = self._build_mask(frame, white_v, white_s, fill_w)

            # 노란선 경계 계산 (한쪽만 보여도 이전값으로 보완)
            x_min, x_max = self._yellow_x_bounds(yellow)

            yel_dil    = cv2.dilate(yellow, cv2.getStructuringElement(cv2.MORPH_RECT, (25,25)))
            white_only = cv2.bitwise_and(white, cv2.bitwise_not(yel_dil))
            cnts, _    = cv2.findContours(white_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lanes = []
            for c in cnts:
                if cv2.contourArea(c) < min_area: continue
                bot_x = self._contour_bottom_cx(c)
                # 노란선 범위 바깥이면 무시
                if bot_x < x_min or bot_x > x_max: continue
                lanes.append((cv2.contourArea(c), bot_x, c))
            # 면적 큰 순으로 최대 3개만 선택 후 x좌표 오름차순 정렬
            lanes.sort(key=lambda t: t[0], reverse=True)
            lanes = lanes[:3]
            lanes.sort(key=lambda t: t[1])
            lanes = [(bot_x, c) for (_, bot_x, c) in lanes]

            # cx가 어느 contour 안에 있는지 판정
            test_y    = int(roi_h * 0.85)
            current_i = -1
            for i, (bot_x, c) in enumerate(lanes):
                if cv2.pointPolygonTest(c, (float(cx), float(test_y)), False) >= 0:
                    current_i = i
                    break

            if current_i >= 0:
                # 왼쪽 경계: 바로 왼쪽 흰선이 있으면 사용, 없으면(1차선) 노란선 x_min 사용
                if current_i > 0:
                    left_x = lanes[current_i - 1][0]
                else:
                    left_x = x_min
                # 오른쪽 경계: 바로 오른쪽 흰선이 있으면 사용, 없으면(3차선) 노란선 x_max 사용
                if current_i < len(lanes) - 1:
                    right_x = lanes[current_i + 1][0]
                else:
                    right_x = x_max
                center_x = (left_x + right_x) / 2
                self.current_lane_x = center_x
                offset = (center_x - cx) / cx
            else:
                offset = 999.0  # 차선 이탈

            # 노란선 경계 안쪽 yellow만 시각화에 사용
            yellow_vis = yellow.copy()
            yellow_vis[:, :x_min] = 0
            yellow_vis[:, x_max:] = 0

            # mask_vis 색칠
            mask_vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
            mask_vis[yellow_vis > 0] = [0, 0, 255]
            for i, (bot_x, c) in enumerate(lanes):
                color = (0, 255, 0) if i == current_i else (0, 255, 255)
                cv2.drawContours(mask_vis, [c], -1, color, cv2.FILLED)
                cv2.circle(mask_vis, (bot_x, roi_h - 15), 5, color, -1)

            # 노란선 경계 시각화
            cv2.line(mask_vis, (x_min, 0), (x_min, roi_h), (0, 180, 0), 1)
            cv2.line(mask_vis, (x_max, 0), (x_max, roi_h), (0, 180, 0), 1)

            # 카메라 중심선
            cv2.line(mask_vis, (cx, roi_h-18), (cx, roi_h-1), (255,200,0), 4)

            # HUD (lane 번호를 1-based로 표시)
            in_lane_lbl = f"lane {current_i + 1}" if current_i >= 0 else "---"
            cv2.putText(mask_vis, f"offset: {offset:.3f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(mask_vis, f"IN LANE: {in_lane_lbl}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0) if current_i>=0 else (100,100,100), 1)
            cv2.putText(mask_vis, f"white lanes: {len(lanes)}", (10, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            cv2.putText(mask_vis, f"[R] reset bounds", (10, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

            out_img = mask_vis
            cv2.imshow('Lane Detection', np.vstack([roi, mask_vis]))

            # ROS publish
            stamp = self.get_clock().now().to_msg()
            m  = self.bridge.cv2_to_imgmsg(frame,   encoding="bgr8"); m.header.stamp  = stamp
            m2 = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8"); m2.header.stamp = stamp
            self.color_pub.publish(m)
            self.lane_pub.publish(m2)
            off = Float32(); off.data = float(offset)
            self.offset_pub.publish(off)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): rclpy.shutdown()
            if k == ord('r'):
                self.prev_x_min = 0
                self.prev_x_max = self.W
                self.get_logger().info("노란선 경계 리셋!")

        except Exception as e:
            self.get_logger().error(f"에러: {e}", throttle_duration_sec=1.0)
            import traceback; traceback.print_exc()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = LaneDetectionFinal()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()