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
        self.roi_start  = int(self.H * 0.45)  # 소실점 바로 아래부터
        self.roi_h      = self.H - self.roi_start

        # ── 파라미터
        self.canny_t1 = 50
        self.canny_t2 = 150
        self.bev_left  = 100   # 노란선 왼쪽 여유
        self.bev_right = 580   # 오른쪽 흰선 포함
        self.nwindows  = 9
        self.margin    = 60
        self.minpix    = 40

        # ── 차선 타겟 (0=중앙 흰선, 1=오른쪽 흰선)
        # 라이다 노드에서 /lane/target 으로 0 또는 1을 퍼블리시하면 됨
        self.lane_target = 0

        # ── 이전 프레임 fit 저장 (3개: 노란, 흰1, 흰2)
        self.fit_yellow = None
        self.fit_white1 = None  # 중앙 흰선 (노란선과 가까운 쪽)
        self.fit_white2 = None  # 오른쪽 흰선

        # ── BEV
        self.dst_points = np.float32([
            [0, 0], [self.W, 0], [self.W, self.roi_h], [0, self.roi_h]
        ])
        # BEV 후 각 차선 예상 x위치 (히스토그램 피크 탐색 범위 힌트)
        # 노란선: x ~ 80~180, 중앙흰선: x ~ 280~400, 오른쪽흰선: x ~ 450~600
        self._update_perspective()

        # ── OpenCV 창 & 트랙바
        for name in ['1. ROI + Canny', '2. BEV', '3. Lane Detection']:
            cv2.namedWindow(name)
        cv2.createTrackbar('Canny T1',  '1. ROI + Canny',    self.canny_t1,  255, self._noop)
        cv2.createTrackbar('Canny T2',  '1. ROI + Canny',    self.canny_t2,  255, self._noop)
        cv2.createTrackbar('BEV Left',  '2. BEV',            self.bev_left,  640, self._noop)
        cv2.createTrackbar('BEV Right', '2. BEV',            self.bev_right, 640, self._noop)
        cv2.createTrackbar('Windows',   '3. Lane Detection',  self.nwindows,   15, self._noop)
        cv2.createTrackbar('Margin',    '3. Lane Detection',  self.margin,    150, self._noop)

        # ── Publisher / Subscriber
        self.color_pub  = self.create_publisher(Image,   '/camera/color/image_raw', 1)
        self.lane_pub   = self.create_publisher(Image,   '/camera/lane/image_raw',  1)
        self.offset_pub = self.create_publisher(Float32, '/lane/offset',             1)
        # 라이다 노드가 장애물 감지 시 0→1 퍼블리시
        self.create_subscription(Int32, '/lane/target', self._target_cb, 1)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("Lane Detection 노드 시작 (target=0: 중앙 흰선)")

    # ────────────────────────────────────────────
    def _noop(self, _): pass

    def _target_cb(self, msg):
        prev = self.lane_target
        self.lane_target = int(msg.data)
        if prev != self.lane_target:
            label = "중앙 흰선" if self.lane_target == 0 else "오른쪽 흰선"
            self.get_logger().info(f"차선 타겟 변경 → {label}")

    def _update_perspective(self):
        # 상단은 소실점 근처(좁게), 하단은 카메라 화각 전체(넓게)
        # BEV Left/Right = 상단 소실점 x 범위
        self.src_points = np.float32([
            [self.bev_left,  0],
            [self.bev_right, 0],
            [self.W - 20,    self.roi_h],
            [20,             self.roi_h]
        ])
        self.M    = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    # ────────────────────────────────────────────
    def _build_binary(self, frame):
        """HSV + Canny로 노란/흰색 이진 마스크 생성 후 BEV 변환"""
        roi = frame[self.roi_start:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        k   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 흰색: 채도 낮고 명도 매우 높음 (바닥 반사광 제거)
        white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 25, 255]))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,  k, iterations=2)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)

        # 노란색 마스크
        yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN,  k)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

        # Canny (흰선 보완)
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_t1, self.canny_t2)
        white = cv2.bitwise_or(white, edges)

        # BEV 변환
        white_bev  = cv2.warpPerspective(white,  self.M, (self.W, self.roi_h))
        yellow_bev = cv2.warpPerspective(yellow, self.M, (self.W, self.roi_h))
        return white_bev, yellow_bev, roi

    # ────────────────────────────────────────────
    def _find_white_peaks(self, white_bev):
        """
        흰선 2개를 구간 분리로 찾기.
        - 중앙 흰선: 화면 1/4 ~ 2/4 구간
        - 오른쪽 흰선: 화면 2/4 ~ 끝 구간
        """
        hist = np.sum(white_bev[self.roi_h // 2:, :], axis=0).astype(float)
        hist_smooth = np.convolve(hist, np.ones(15)/15, mode='same')
        w = len(hist_smooth)

        peaks = []
        # 구간 1: 중앙 흰선 (전체의 20%~55%)
        seg1 = hist_smooth[w//5 : w*55//100].copy()
        idx1 = int(np.argmax(seg1)) + w//5
        if seg1.max() > 200:
            peaks.append(idx1)

        # 구간 2: 오른쪽 흰선 (전체의 50%~95%)
        seg2 = hist_smooth[w//2 : w*95//100].copy()
        idx2 = int(np.argmax(seg2)) + w//2
        if seg2.max() > 200 and abs(idx2 - idx1) > 60:  # 같은 차선 중복 방지
            peaks.append(idx2)

        peaks.sort()
        return peaks

    def _find_yellow_peak(self, yellow_bev):
        hist = np.sum(yellow_bev[self.roi_h // 2:, :], axis=0).astype(float)
        idx  = int(np.argmax(hist))
        return idx if hist[idx] > 200 else None

    # ────────────────────────────────────────────
    def _sliding_window(self, binary, start_x, color, out_img):
        """단일 차선 sliding window. fit 계수 반환"""
        h, w   = binary.shape
        win_h  = h // self.nwindows
        nz     = np.nonzero(binary)
        nzy, nzx = nz[0], nz[1]

        cur_x  = start_x
        inds   = []

        for win in range(self.nwindows):
            y_lo = h - (win + 1) * win_h
            y_hi = h - win * win_h
            x_lo = cur_x - self.margin
            x_hi = cur_x + self.margin
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
            # 픽셀 색칠
            out_img[py, px] = color
            return fit
        return None

    # ────────────────────────────────────────────
    def _draw_lanes(self, out_img, offset):
        """fit으로 차선 곡선 + 주행 영역 그리기"""
        h = self.roi_h
        ploty = np.linspace(0, h - 1, h)

        def eval_fit(fit):
            return fit[0]*ploty**2 + fit[1]*ploty + fit[2] if fit is not None else None

        y_x  = eval_fit(self.fit_yellow)
        w1_x = eval_fit(self.fit_white1)
        w2_x = eval_fit(self.fit_white2)

        # 노란선 (청록)
        if y_x is not None:
            pts = np.array([np.stack([y_x, ploty], axis=1)], dtype=np.int32)
            cv2.polylines(out_img, pts, False, (255, 200, 0), 5)

        # 중앙 흰선 (초록)
        if w1_x is not None:
            pts = np.array([np.stack([w1_x, ploty], axis=1)], dtype=np.int32)
            cv2.polylines(out_img, pts, False, (0, 255, 0), 5)

        # 오른쪽 흰선 (파랑)
        if w2_x is not None:
            pts = np.array([np.stack([w2_x, ploty], axis=1)], dtype=np.int32)
            cv2.polylines(out_img, pts, False, (255, 100, 0), 5)

        # 주행 영역 반투명 표시
        target_x = w1_x if self.lane_target == 0 else w2_x
        left_x   = y_x  if self.lane_target == 0 else w1_x
        if target_x is not None and left_x is not None:
            overlay = out_img.copy()
            left_pts  = np.stack([left_x,   ploty], axis=1).astype(np.int32)
            right_pts = np.stack([target_x, ploty], axis=1).astype(np.int32)
            fill_pts  = np.vstack([left_pts, right_pts[::-1]])  # 폐곡선
            cv2.fillPoly(overlay, [fill_pts], (0, 80, 0))
            cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)

        # offset & 타겟 표시
        cx = self.W // 2
        cv2.line(out_img, (cx, h-1), (cx, h-20), (255,255,255), 1)
        label = "TARGET: 중앙 흰선" if self.lane_target == 0 else "TARGET: 오른쪽 흰선"
        label = "TARGET: Center" if self.lane_target == 0 else "TARGET: Right"
        cv2.putText(out_img, label,               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
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
            self.canny_t1  = cv2.getTrackbarPos('Canny T1',  '1. ROI + Canny')
            self.canny_t2  = max(self.canny_t1 + 1,
                             cv2.getTrackbarPos('Canny T2',  '1. ROI + Canny'))
            bl = cv2.getTrackbarPos('BEV Left',  '2. BEV')
            br = cv2.getTrackbarPos('BEV Right', '2. BEV')
            self.bev_left  = bl
            self.bev_right = max(bl + 10, br)
            self.nwindows  = max(1,  cv2.getTrackbarPos('Windows', '3. Lane Detection'))
            self.margin    = max(10, cv2.getTrackbarPos('Margin',  '3. Lane Detection'))
            self._update_perspective()

            # 이진화 + BEV
            white_bev, yellow_bev, roi = self._build_binary(frame)

            # 피크 탐색
            white_peaks  = self._find_white_peaks(white_bev)
            yellow_peak  = self._find_yellow_peak(yellow_bev)

            # 시각화 캔버스
            out_img = np.zeros((self.roi_h, self.W, 3), dtype=np.uint8)

            # ── 노란선 sliding window
            yp = yellow_peak if yellow_peak is not None else (
                 int(self.fit_yellow[2]) if self.fit_yellow is not None else 80)
            fit = self._sliding_window(yellow_bev, yp, (0, 0, 255), out_img)
            if fit is not None:
                self.fit_yellow = fit

            # ── 흰선 2개 sliding window
            # white_peaks가 2개면 그대로, 1개면 이전 fit 사용
            if len(white_peaks) >= 2:
                f1 = self._sliding_window(white_bev, white_peaks[0], (0, 255, 0),     out_img)
                f2 = self._sliding_window(white_bev, white_peaks[1], (0, 255, 255),   out_img)
                if f1 is not None: self.fit_white1 = f1
                if f2 is not None: self.fit_white2 = f2
            elif len(white_peaks) == 1:
                # 하나만 보일 때: 어느 차선인지 판단 (노란선 기준 오른쪽)
                yref = yp
                if white_peaks[0] > yref + 50:
                    # 노란선보다 오른쪽 → 중앙 흰선으로 간주
                    f1 = self._sliding_window(white_bev, white_peaks[0], (0, 255, 0), out_img)
                    if f1 is not None: self.fit_white1 = f1
            # peaks 0개면 이전 fit 그대로 유지

            # ── offset 계산
            ploty   = np.linspace(0, self.roi_h - 1, self.roi_h)
            offset  = 0.0
            cx      = self.W / 2

            target_fit = self.fit_white1 if self.lane_target == 0 else self.fit_white2
            left_fit   = self.fit_yellow if self.lane_target == 0 else self.fit_white1

            if target_fit is not None and left_fit is not None:
                t_x = target_fit[0]*ploty[-1]**2 + target_fit[1]*ploty[-1] + target_fit[2]
                l_x = left_fit[0] *ploty[-1]**2 + left_fit[1] *ploty[-1] + left_fit[2]
                lane_center = (t_x + l_x) / 2
                offset = (lane_center - cx) / cx  # -1.0 ~ +1.0
            elif target_fit is not None:
                # 한쪽만 보일 때: 차선까지 거리로 offset
                t_x  = target_fit[0]*ploty[-1]**2 + target_fit[1]*ploty[-1] + target_fit[2]
                offset = (t_x - cx) / cx

            # ── 그리기
            result = self._draw_lanes(out_img, offset)

            # ── 1번 창
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges    = cv2.Canny(cv2.GaussianBlur(gray_roi,(5,5),0), self.canny_t1, self.canny_t2)
            cv2.imshow('1. ROI + Canny', cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

            # ── 2번 창: ROI + BEV 마스크
            roi_vis = roi.copy()
            pts = self.src_points.astype(np.int32)
            cv2.polylines(roi_vis, [pts], True, (0,255,0), 2)
            for p in pts: cv2.circle(roi_vis, tuple(p), 5, (0,0,255), -1)
            bev_vis = cv2.cvtColor(white_bev, cv2.COLOR_GRAY2BGR)
            bev_vis[yellow_bev > 0] = [255, 200, 0]
            cv2.imshow('2. BEV', np.vstack([roi_vis, bev_vis]))

            # ── 3번 창
            cv2.imshow('3. Lane Detection', result)

            # ── ROS2 퍼블리시
            stamp = self.get_clock().now().to_msg()
            m = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            m.header.stamp = stamp; self.color_pub.publish(m)
            m2 = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            m2.header.stamp = stamp; self.lane_pub.publish(m2)
            off_msg = Float32(); off_msg.data = float(offset)
            self.offset_pub.publish(off_msg)

            # ── 키
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()
            elif key == ord('0'):
                self.lane_target = 0
                self.get_logger().info("수동 전환 → 중앙 흰선")
            elif key == ord('1'):
                self.lane_target = 1
                self.get_logger().info("수동 전환 → 오른쪽 흰선")
            elif key == ord('s'):
                self.get_logger().info(
                    f"[설정] Canny=({self.canny_t1},{self.canny_t2}) "
                    f"BEV=({self.bev_left},{self.bev_right}) "
                    f"Win={self.nwindows} Margin={self.margin} "
                    f"Offset={offset:.3f} Target={self.lane_target}")

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