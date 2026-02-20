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
        self.roi_start = int(self.H * 0.55)
        self.roi_h     = self.H - self.roi_start

        # 파라미터
        self.canny_t1 = 50
        self.canny_t2 = 150
        self.nwindows = 9
        self.margin   = 60
        self.minpix   = 50

        # w2 추가 (흰선 최대 3개)
        self.fit_prev   = {'y0': None, 'y1': None, 'w0': None, 'w1': None, 'w2': None}
        self.fail_count = {'y0': 0,    'y1': 0,    'w0': 0,    'w1': 0,    'w2': 0}
        self.FAIL_RESET = 3

        # 노란선/흰선 안정화용 이전 피크 저장
        self.y_peak_prev = [None, None]   # [왼쪽 노란선, 오른쪽 노란선]
        self.w_peak_prev = [None, None, None]  # 흰선 최대 3개
        self.Y_SMOOTH    = 0.6            # 이전 피크 가중치 (0~1, 클수록 안정적)
        self.W_SMOOTH    = 0.5
        self.W_JUMP      = 60             # 흰선 튐 판정 거리 (px)

        self.target_offset_dir = 0
        self.current_lane_x = self.W / 2

        cv2.namedWindow('1. ROI Mask')
        cv2.namedWindow('2. Lane Detection')
        cv2.createTrackbar('Canny T1',  '1. ROI Mask', self.canny_t1, 255, self._noop)
        cv2.createTrackbar('Canny T2',  '1. ROI Mask', self.canny_t2, 255, self._noop)
        cv2.createTrackbar('ROI Start', '1. ROI Mask', 55,  90,  self._noop)
        cv2.createTrackbar('White V',   '1. ROI Mask', 200, 255, self._noop)
        cv2.createTrackbar('White S',   '1. ROI Mask', 30,  100, self._noop)
        cv2.createTrackbar('Windows',   '2. Lane Detection', self.nwindows, 15,  self._noop)
        cv2.createTrackbar('Margin',    '2. Lane Detection', self.margin,   150, self._noop)

        self.color_pub  = self.create_publisher(Image,   '/camera/color/image_raw', 1)
        self.lane_pub   = self.create_publisher(Image,   '/camera/lane/image_raw',  1)
        self.offset_pub = self.create_publisher(Float32, '/lane/offset',             1)
        self.create_subscription(Int32, '/lane/target', self._target_cb, 1)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("Lane Detection 시작")

    # ────────────────────────────────────────────
    def _noop(self, _): pass

    def _target_cb(self, msg):
        self.target_offset_dir = int(msg.data)

    # ────────────────────────────────────────────
    def _build_mask(self, frame, white_v=200, white_s=30):
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
        white = cv2.bitwise_or(white, edges)

        return white, yellow, roi

    # ────────────────────────────────────────────
    def _extract_peaks(self, hist, min_val=150, min_dist=50, max_peaks=2):
        peaks = []
        tmp = hist.copy()
        # 가장자리 제외 없음 → 화면 끝에 걸친 선도 탐지
        for _ in range(max_peaks):
            idx = int(np.argmax(tmp))
            if tmp[idx] < min_val:
                break
            peaks.append(idx)
            tmp[max(0, idx-min_dist):min(len(tmp), idx+min_dist)] = 0
        peaks.sort()
        return peaks

    def _find_peaks(self, white, yellow):
        h  = white.shape[0]
        hw = np.convolve(np.sum(white[h*2//3:,  :], axis=0).astype(float), np.ones(15)/15, mode='same')
        hy = np.convolve(np.sum(yellow[h*2//3:, :], axis=0).astype(float), np.ones(15)/15, mode='same')

        # 노란선 피크 탐색
        raw_y_peaks = self._extract_peaks(hy, min_val=150, min_dist=50, max_peaks=2)

        # ── 노란선 안정화: 이전 피크와 가중 평균
        y_peaks = []
        # 왼쪽(인덱스0), 오른쪽(인덱스1) 각각 처리
        left_raws  = [p for p in raw_y_peaks if p < self.W // 2]
        right_raws = [p for p in raw_y_peaks if p >= self.W // 2]

        for i, raws in enumerate([left_raws, right_raws]):
            if raws:
                new_p = raws[0]
                if self.y_peak_prev[i] is not None:
                    # 이전 위치에서 너무 멀리 튀면(>80px) 이전값 유지
                    if abs(new_p - self.y_peak_prev[i]) > 80:
                        new_p = self.y_peak_prev[i]
                    else:
                        new_p = int(self.Y_SMOOTH * self.y_peak_prev[i] + (1 - self.Y_SMOOTH) * new_p)
                self.y_peak_prev[i] = new_p
                y_peaks.append(new_p)
            else:
                # 이번 프레임에서 못 찾았으면 이전값 유지 (None이면 그냥 스킵)
                if self.y_peak_prev[i] is not None:
                    y_peaks.append(self.y_peak_prev[i])

        y_peaks.sort()

        # 노란선 기준으로 흰선 탐색 범위 결정
        left_bound  = 0
        right_bound = self.W
        if len(y_peaks) >= 1:
            if y_peaks[0] < self.W // 2:
                left_bound = y_peaks[0]
            else:
                right_bound = y_peaks[0]
        if len(y_peaks) >= 2:
            left_bound  = y_peaks[0]
            right_bound = y_peaks[1]

        hw_filtered = hw.copy()
        hw_filtered[:left_bound]  = 0
        hw_filtered[right_bound:] = 0

        for yp in y_peaks:
            hw_filtered[max(0, yp-60):min(len(hw_filtered), yp+60)] = 0

        # 흰선 최대 3개
        # 하단 1/2로 범위 넓혀서 가장자리 흰선 픽셀 더 잡기
        hw_full = np.convolve(np.sum(white[h//2:, :], axis=0).astype(float), np.ones(15)/15, mode='same')
        hw_full[:left_bound]  = 0
        hw_full[right_bound:] = 0
        for yp in y_peaks:
            hw_full[max(0, yp-60):min(len(hw_full), yp+60)] = 0

        raw_w_peaks = self._extract_peaks(hw_full, min_val=150, min_dist=80, max_peaks=3)

        # ── 흰선 안정화: x 오름차순 정렬 후 순서대로 이전 피크와 1:1 매칭
        raw_w_peaks.sort()
        prev_valid = [p for p in self.w_peak_prev if p is not None]
        prev_valid.sort()

        w_peaks = []
        for i, rp in enumerate(raw_w_peaks):
            if i < len(prev_valid):
                pp = prev_valid[i]
                if abs(rp - pp) > self.W_JUMP:
                    w_peaks.append(pp)   # 튀면 이전값 유지
                else:
                    w_peaks.append(int(self.W_SMOOTH * pp + (1 - self.W_SMOOTH) * rp))
            else:
                w_peaks.append(rp)

        # 이전 피크 업데이트
        for i in range(len(self.w_peak_prev)):
            self.w_peak_prev[i] = w_peaks[i] if i < len(w_peaks) else None

        return y_peaks, w_peaks

    # ────────────────────────────────────────────
    def _sliding_window(self, binary, start_x, out_img):
        h, w  = binary.shape
        win_h = h // self.nwindows
        nz    = np.nonzero(binary)
        nzy, nzx = nz[0], nz[1]
        cur_x = start_x
        inds  = []
        for win in range(self.nwindows):
            y_lo = h - (win+1)*win_h;  y_hi = h - win*win_h
            x_lo = max(0, cur_x-self.margin); x_hi = min(w-1, cur_x+self.margin)
            cv2.rectangle(out_img, (x_lo,y_lo), (x_hi,y_hi), (60,60,60), 1)
            good = ((nzy>=y_lo)&(nzy<y_hi)&(nzx>=x_lo)&(nzx<x_hi)).nonzero()[0]
            inds.append(good)
            if len(good) > self.minpix:
                cur_x = int(np.mean(nzx[good]))
        inds   = np.concatenate(inds)
        px, py = nzx[inds], nzy[inds]
        if len(py) > 100:
            return np.polyfit(py, px, 2), px, py
        return None, px, py

    # ────────────────────────────────────────────
    def _update_fit(self, key, mask, peak, out_img, pixel_color):
        if peak is not None:
            fit, px, py = self._sliding_window(mask, peak, out_img)
            if fit is not None:
                self.fit_prev[key]   = fit
                self.fail_count[key] = 0
                out_img[py, px]      = pixel_color
                return
        self.fail_count[key] += 1
        if self.fail_count[key] >= self.FAIL_RESET:
            self.fit_prev[key]   = None
            self.fail_count[key] = 0

    # ────────────────────────────────────────────
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("프레임 읽기 실패", throttle_duration_sec=5.0)
            return
        try:
            self.canny_t1  = cv2.getTrackbarPos('Canny T1',  '1. ROI Mask')
            self.canny_t2  = max(self.canny_t1+1, cv2.getTrackbarPos('Canny T2', '1. ROI Mask'))
            roi_pct        = cv2.getTrackbarPos('ROI Start', '1. ROI Mask')
            white_v        = cv2.getTrackbarPos('White V',   '1. ROI Mask')
            white_s        = cv2.getTrackbarPos('White S',   '1. ROI Mask')
            self.roi_start = int(self.H * roi_pct / 100)
            self.roi_h     = self.H - self.roi_start
            self.nwindows  = max(1,  cv2.getTrackbarPos('Windows', '2. Lane Detection'))
            self.margin    = max(10, cv2.getTrackbarPos('Margin',  '2. Lane Detection'))

            white, yellow, roi = self._build_mask(frame, white_v, white_s)
            y_peaks, w_peaks   = self._find_peaks(white, yellow)

            out_img = np.zeros((self.roi_h, self.W, 3), dtype=np.uint8)

            # 노란선 (빨간)
            self._update_fit('y0', yellow, y_peaks[0] if len(y_peaks)>0 else None, out_img, (0,0,255))
            self._update_fit('y1', yellow, y_peaks[1] if len(y_peaks)>1 else None, out_img, (0,0,255))

            # 흰선 최대 3개 (회색)
            self._update_fit('w0', white, w_peaks[0] if len(w_peaks)>0 else None, out_img, (150,150,150))
            self._update_fit('w1', white, w_peaks[1] if len(w_peaks)>1 else None, out_img, (150,150,150))
            self._update_fit('w2', white, w_peaks[2] if len(w_peaks)>2 else None, out_img, (150,150,150))

            # ── 주행 차선 결정
            ploty  = np.linspace(0, self.roi_h-1, self.roi_h)
            last_y = self.roi_h - 1

            def eval_x(fit):
                return fit[0]*last_y**2 + fit[1]*last_y + fit[2] if fit is not None else None

            w_fits = []
            for key in ['w0','w1','w2']:
                fx = eval_x(self.fit_prev[key])
                if fx is not None:
                    w_fits.append((fx, key))
            w_fits.sort(key=lambda t: t[0])

            cx = self.W / 2
            if len(w_fits) > 0:
                closest_i = int(np.argmin([abs(fx - cx) for fx, _ in w_fits]))
                target_i  = max(0, min(len(w_fits)-1, closest_i + self.target_offset_dir))
                current_key = w_fits[target_i][1]
                self.current_lane_x = w_fits[target_i][0]
            else:
                current_key = None

            # ── 곡선 그리기
            for key in ['y0','y1']:
                fit = self.fit_prev[key]
                if fit is None: continue
                x = (fit[0]*ploty**2 + fit[1]*ploty + fit[2]).astype(np.int32)
                cv2.polylines(out_img, [np.stack([x, ploty.astype(np.int32)], axis=1)], False, (0,0,255), 5)

            for fx, key in w_fits:
                fit   = self.fit_prev[key]
                color = (0,255,0) if key == current_key else (0,255,255)
                x = (fit[0]*ploty**2 + fit[1]*ploty + fit[2]).astype(np.int32)
                cv2.polylines(out_img, [np.stack([x, ploty.astype(np.int32)], axis=1)], False, color, 5)

            # ── offset 계산
            offset = (self.current_lane_x - cx) / cx if len(w_fits) > 0 else 0.0


            cv2.putText(out_img, f"offset: {offset:.3f}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            lbl = "TARGET: Center" if self.target_offset_dir==0 else (
                  "TARGET: Left"   if self.target_offset_dir==-1 else "TARGET: Right")
            cv2.putText(out_img, lbl, (10,42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
            # 인식된 흰선 수 표시
            cv2.putText(out_img, f"white lanes: {len(w_fits)}", (10,64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            mask_vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
            mask_vis[yellow > 0] = [0, 0, 255]
            # 크로스헤어 (마스크 이미지 하단 중앙)
            mh, mw = mask_vis.shape[:2]
            ch_x = mw // 2
            cv2.line(mask_vis, (ch_x, mh - 18), (ch_x, mh - 1), (255, 200, 0), 4)
            cv2.imshow('1. ROI Mask', np.vstack([roi, mask_vis]))
            cv2.imshow('2. Lane Detection', out_img)

            stamp = self.get_clock().now().to_msg()
            m = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            m.header.stamp = stamp; self.color_pub.publish(m)
            m2 = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8")
            m2.header.stamp = stamp; self.lane_pub.publish(m2)
            off = Float32(); off.data = float(offset)
            self.offset_pub.publish(off)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): rclpy.shutdown()
            elif key == ord('s'):
                self.get_logger().info(
                    f"[설정] Canny=({self.canny_t1},{self.canny_t2}) "
                    f"ROI={roi_pct}% Win={self.nwindows} "
                    f"Margin={self.margin} Offset={offset:.3f}")

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