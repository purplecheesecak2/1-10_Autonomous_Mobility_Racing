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

        # ── 웹캠
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다!")
            return
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.W, self.H = 640, 480

        # ── ROI: 하단 45%만 사용 (바닥 차선만 보이는 구간)
        self.roi_start = int(self.H * 0.55)
        self.roi_h     = self.H - self.roi_start

        # ── Canny 파라미터
        self.canny_t1 = 50
        self.canny_t2 = 150

        # ── Sliding Window 파라미터
        self.nwindows = 9
        self.margin   = 40
        self.minpix   = 30

        # ── 차선 타겟 (0=중앙 흰선, 1=오른쪽 흰선)
        self.lane_target = 0

        # ── 이전 프레임 fit 저장
        self.fit_yellow  = None
        self.fit_yellow2 = None  # 오른쪽 노란선
        self.fit_white1  = None  # 중앙 흰선
        self.fit_white2  = None  # 오른쪽 흰선

        # ── OpenCV 창 & 트랙바
        cv2.namedWindow('1. ROI Mask')
        cv2.namedWindow('2. Lane Detection')
        cv2.createTrackbar('Canny T1', '1. ROI Mask', self.canny_t1, 255, self._noop)
        cv2.createTrackbar('Canny T2', '1. ROI Mask', self.canny_t2, 255, self._noop)
        cv2.createTrackbar('ROI Start', '1. ROI Mask', 55, 90, self._noop)  # H의 몇 % 아래부터
        cv2.createTrackbar('Windows',  '2. Lane Detection', self.nwindows, 15, self._noop)
        cv2.createTrackbar('Margin',   '2. Lane Detection', self.margin,  150, self._noop)

        # ── Publisher / Subscriber
        self.color_pub  = self.create_publisher(Image,   '/camera/color/image_raw', 1)
        self.lane_pub   = self.create_publisher(Image,   '/camera/lane/image_raw',  1)
        self.offset_pub = self.create_publisher(Float32, '/lane/offset',             1)
        self.create_subscription(Int32, '/lane/target', self._target_cb, 1)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("Lane Detection 시작 (No BEV)")

    # ────────────────────────────────────────────
    def _noop(self, _): pass

    def _target_cb(self, msg):
        prev = self.lane_target
        self.lane_target = int(msg.data)
        if prev != self.lane_target:
            label = "Center white" if self.lane_target == 0 else "Right white"
            self.get_logger().info(f"Lane target -> {label}")

    # ────────────────────────────────────────────
    def _build_mask(self, frame):
        """HSV + Canny로 노란/흰색 마스크 생성 (ROI에 직접 적용)"""
        roi = frame[self.roi_start:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        k   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 흰색 마스크 (채도 낮고 명도 높음)
        white = cv2.inRange(hsv, np.array([0,  0,  200]), np.array([180, 30, 255]))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,  k, iterations=2)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)

        # 노란색 마스크 (실내 형광등 고려해서 H/S 범위 확대)
        yellow = cv2.inRange(hsv, np.array([10, 50, 80]), np.array([40, 255, 255]))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN,  k)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

        # Canny 엣지 (흰선 얇을 때 보완)
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_t1, self.canny_t2)
        white = cv2.bitwise_or(white, edges)

        return white, yellow, roi

    # ────────────────────────────────────────────
    def _find_peaks(self, white_mask, yellow_mask):
        """히스토그램으로 차선 시작점 탐색"""
        h = white_mask.shape[0]
        hist_w = np.sum(white_mask[h//2:, :],  axis=0).astype(float)
        hist_y = np.sum(yellow_mask[h//2:, :], axis=0).astype(float)

        # 스무딩
        hist_w = np.convolve(hist_w, np.ones(15)/15, mode='same')
        hist_y = np.convolve(hist_y, np.ones(15)/15, mode='same')

        # 노란선: 왼쪽 절반에서 피크 (왼쪽 경계)
        y_peak = int(np.argmax(hist_y[:self.W//2]))
        if hist_y[:self.W//2].max() < 150:
            y_peak = None

        # 오른쪽 노란선: 오른쪽 절반에서 피크
        seg_y2 = hist_y[self.W//2:]
        y2_peak = int(np.argmax(seg_y2)) + self.W//2 if seg_y2.max() > 150 else None

        # 중앙 흰선: 전체의 20~60% 구간
        seg1 = hist_w[self.W//5 : self.W*6//10].copy()
        w1_peak = int(np.argmax(seg1)) + self.W//5 if seg1.max() > 150 else None

        # 오른쪽 흰선: 전체의 55~95% 구간 (오른쪽 노란선과 안 겹치면)
        seg2 = hist_w[self.W*55//100 : self.W*95//100].copy()
        w2_peak = int(np.argmax(seg2)) + self.W*55//100 if seg2.max() > 150 else None
        if w1_peak is not None and w2_peak is not None and abs(w1_peak - w2_peak) < 60:
            w2_peak = None

        return y_peak, w1_peak, w2_peak, y2_peak

    # ────────────────────────────────────────────
    def _sliding_window(self, binary, start_x, color, out_img):
        """단일 차선 Sliding Window → 2차 fit 반환"""
        h, w  = binary.shape
        win_h = h // self.nwindows
        nz    = np.nonzero(binary)
        nzy, nzx = nz[0], nz[1]

        cur_x = start_x
        inds  = []

        for win in range(self.nwindows):
            y_lo = h - (win + 1) * win_h
            y_hi = h - win * win_h
            x_lo = max(0,   cur_x - self.margin)
            x_hi = min(w-1, cur_x + self.margin)
            cv2.rectangle(out_img, (x_lo, y_lo), (x_hi, y_hi), color, 1)

            good = ((nzy >= y_lo) & (nzy < y_hi) &
                    (nzx >= x_lo) & (nzx < x_hi)).nonzero()[0]
            inds.append(good)
            if len(good) > self.minpix:
                cur_x = int(np.mean(nzx[good]))

        inds = np.concatenate(inds)
        px, py = nzx[inds], nzy[inds]

        if len(py) > 30:
            fit = np.polyfit(py, px, 2)
            out_img[py, px] = color
            return fit
        return None

    # ────────────────────────────────────────────
    def _draw_result(self, out_img, offset):
        """fit으로 차선 + 주행 영역 + offset 표시"""
        h     = self.roi_h
        ploty = np.linspace(0, h - 1, h)

        def curve(fit):
            return (fit[0]*ploty**2 + fit[1]*ploty + fit[2]).astype(np.int32) if fit is not None else None

        y_x  = curve(self.fit_yellow)
        w1_x = curve(self.fit_white1)
        w2_x = curve(self.fit_white2)

        # 노란선 (빨간색)
        if y_x is not None:
            pts = np.stack([y_x, ploty.astype(np.int32)], axis=1)
            cv2.polylines(out_img, [pts], False, (0, 0, 255), 5)

        # 중앙 흰선 (초록색)
        if w1_x is not None:
            pts = np.stack([w1_x, ploty.astype(np.int32)], axis=1)
            cv2.polylines(out_img, [pts], False, (0, 255, 0), 5)

        # 오른쪽 노란선 (빨간색)
        y2_x = curve(self.fit_yellow2)
        if y2_x is not None:
            pts = np.stack([y2_x, ploty.astype(np.int32)], axis=1)
            cv2.polylines(out_img, [pts], False, (0, 0, 255), 5)
            pts = np.stack([w2_x, ploty.astype(np.int32)], axis=1)
            cv2.polylines(out_img, [pts], False, (0, 255, 255), 5)

        # 주행 영역 반투명
        target_x = w1_x if self.lane_target == 0 else w2_x
        left_x   = y_x  if self.lane_target == 0 else w1_x
        if target_x is not None and left_x is not None:
            overlay  = out_img.copy()
            iy       = ploty.astype(np.int32)
            fill_pts = np.vstack([
                np.stack([left_x,          iy], axis=1),
                np.stack([target_x[::-1],  iy[::-1]], axis=1)
            ])
            cv2.fillPoly(overlay, [fill_pts], (0, 80, 0))
            cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)

        # offset 기준선
        cx = self.W // 2
        cv2.line(out_img, (cx, h-1), (cx, h-25), (255, 255, 255), 1)

        label = "TARGET: Center" if self.lane_target == 0 else "TARGET: Right"
        cv2.putText(out_img, label,                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0),  1)
        cv2.putText(out_img, f"offset: {offset:.3f}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        return out_img

    # ────────────────────────────────────────────
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("프레임 읽기 실패", throttle_duration_sec=5.0)
            return

        try:
            # 트랙바 읽기
            self.canny_t1  = cv2.getTrackbarPos('Canny T1',  '1. ROI Mask')
            self.canny_t2  = max(self.canny_t1 + 1,
                             cv2.getTrackbarPos('Canny T2',  '1. ROI Mask'))
            roi_pct        = cv2.getTrackbarPos('ROI Start', '1. ROI Mask')
            self.roi_start = int(self.H * roi_pct / 100)
            self.roi_h     = self.H - self.roi_start
            self.nwindows  = max(1,  cv2.getTrackbarPos('Windows', '2. Lane Detection'))
            self.margin    = max(10, cv2.getTrackbarPos('Margin',  '2. Lane Detection'))

            # 마스크 생성
            white, yellow, roi = self._build_mask(frame)

            # 피크 탐색
            y_peak, w1_peak, w2_peak, y2_peak = self._find_peaks(white, yellow)

            # 시각화 캔버스
            out_img = np.zeros((self.roi_h, self.W, 3), dtype=np.uint8)

            # 왼쪽 노란선 sliding window (빨간)
            yp  = y_peak if y_peak is not None else (
                  int(self.fit_yellow[2]) if self.fit_yellow is not None else 80)
            fit = self._sliding_window(yellow, yp, (0, 0, 255), out_img)
            if fit is not None: self.fit_yellow = fit

            # 중앙 흰선 sliding window (초록)
            if w1_peak is not None:
                fit = self._sliding_window(white, w1_peak, (0, 255, 0), out_img)
                if fit is not None: self.fit_white1 = fit

            # 오른쪽 흰선 sliding window (노란)
            if w2_peak is not None:
                fit = self._sliding_window(white, w2_peak, (0, 255, 255), out_img)
                if fit is not None: self.fit_white2 = fit

            # 오른쪽 노란선 sliding window (빨간) → fit_yellow2에 저장
            if y2_peak is not None:
                fit = self._sliding_window(yellow, y2_peak, (0, 0, 255), out_img)
                if fit is not None: self.fit_yellow2 = fit

            # offset 계산 (하단 기준)
            offset = 0.0
            cx     = self.W / 2
            ploty  = np.linspace(0, self.roi_h - 1, self.roi_h)

            target_fit = self.fit_white1 if self.lane_target == 0 else self.fit_white2
            left_fit   = self.fit_yellow if self.lane_target == 0 else self.fit_white1

            if target_fit is not None and left_fit is not None:
                t_x = target_fit[0]*ploty[-1]**2 + target_fit[1]*ploty[-1] + target_fit[2]
                l_x = left_fit[0] *ploty[-1]**2 + left_fit[1] *ploty[-1] + left_fit[2]
                offset = ((t_x + l_x) / 2 - cx) / cx
            elif target_fit is not None:
                t_x    = target_fit[0]*ploty[-1]**2 + target_fit[1]*ploty[-1] + target_fit[2]
                offset = (t_x - cx) / cx

            # 결과 그리기
            result = self._draw_result(out_img, offset)

            # ── 1번 창: ROI 마스크
            mask_vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
            mask_vis[yellow > 0] = [0, 0, 255]
            cv2.imshow('1. ROI Mask', np.vstack([roi, mask_vis]))

            # ── 2번 창: Lane Detection
            cv2.imshow('2. Lane Detection', result)

            # ── ROS2 퍼블리시
            stamp = self.get_clock().now().to_msg()
            m = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            m.header.stamp = stamp; self.color_pub.publish(m)
            m2 = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            m2.header.stamp = stamp; self.lane_pub.publish(m2)
            off = Float32(); off.data = float(offset)
            self.offset_pub.publish(off)

            # ── 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()
            elif key == ord('0'):
                self.lane_target = 0
                self.get_logger().info("수동 전환 -> Center white")
            elif key == ord('1'):
                self.lane_target = 1
                self.get_logger().info("수동 전환 -> Right white")
            elif key == ord('s'):
                self.get_logger().info(
                    f"[설정] Canny=({self.canny_t1},{self.canny_t2}) "
                    f"ROI={roi_pct}% Win={self.nwindows} "
                    f"Margin={self.margin} Offset={offset:.3f} "
                    f"Target={self.lane_target}")

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