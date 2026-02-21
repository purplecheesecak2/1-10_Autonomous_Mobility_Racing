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
        self.roi_start = int(self.H * 0.65)  # 기본값을 65%로 올려서 상단 노이즈 원천 차단

        self.canny_t1 = 50
        self.canny_t2 = 150

        self.target_offset_dir = 0
        self.current_lane_x    = self.W / 2

        cv2.namedWindow('Lane Detection')
        cv2.createTrackbar('Canny T1',  'Lane Detection', 50,  255,  self._noop)
        cv2.createTrackbar('Canny T2',  'Lane Detection', 150, 255,  self._noop)
        cv2.createTrackbar('ROI Start', 'Lane Detection', 65,  90,   self._noop)  # 기본값 65로 상향
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

        # 경계선 두 줄 사이 틈 메꾸기
        fill_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, fill_w), 3))
        edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, fill_k, iterations=2)

        white = cv2.bitwise_or(white, edges)
        return white, yellow, roi

    def _contour_bottom_cx(self, c):
        """contour 최하단 픽셀들의 평균 x — bbox 중심보다 실제 차선 위치에 정확"""
        pts     = c[:, 0, :]
        max_y   = pts[:, 1].max()
        thresh_y = max_y - max(1, int((max_y - pts[:, 1].min()) * 0.1))
        bottom_pts = pts[pts[:, 1] >= thresh_y]
        return int(bottom_pts[:, 0].mean())

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
            cx             = self.W / 2

            white, yellow, roi = self._build_mask(frame, white_v, white_s, fill_w)

            # 흰선 contour 추출 (노란선 영역 제외)
            yel_dil    = cv2.dilate(yellow, cv2.getStructuringElement(cv2.MORPH_RECT, (25,25)))
            white_only = cv2.bitwise_and(white, cv2.bitwise_not(yel_dil))
            cnts, _    = cv2.findContours(white_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 면적 필터만 적용 (세로 비율 조건 제거 — 얇은 차선도 통과)
            lanes = []
            for c in cnts:
                if cv2.contourArea(c) < min_area: continue
                bot_x = self._contour_bottom_cx(c)
                lanes.append((bot_x, c))
            lanes.sort(key=lambda t: t[0])

            # 카메라 중앙 기준 가장 가까운 것 = target (+ offset_dir)
            target_i = -1
            if lanes:
                dists    = [abs(bx - cx) for bx, _ in lanes]
                closest  = int(np.argmin(dists))
                target_i = max(0, min(len(lanes)-1, closest + self.target_offset_dir))
                self.current_lane_x = lanes[target_i][0]

            # mask_vis 색칠
            mask_vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
            mask_vis[yellow > 0] = [0, 0, 255]
            for i, (bot_x, c) in enumerate(lanes):
                color = (0, 255, 0) if i == target_i else (0, 255, 255)
                cv2.drawContours(mask_vis, [c], -1, color, cv2.FILLED)
                cv2.circle(mask_vis, (bot_x, roi_h - 15), 5, color, -1)

            # 카메라 중심선
            cv2.line(mask_vis, (self.W//2, roi_h-18), (self.W//2, roi_h-1), (255,200,0), 4)

            # offset 계산 및 HUD
            offset = (self.current_lane_x - cx) / cx if lanes else 0.0
            cv2.putText(mask_vis, f"offset: {offset:.3f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            dir_lbl = "Center" if self.target_offset_dir==0 else (
                      "Left"   if self.target_offset_dir==-1 else "Right")
            cv2.putText(mask_vis, f"TARGET: {dir_lbl}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
            cv2.putText(mask_vis, f"white lanes: {len(lanes)}", (10, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

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