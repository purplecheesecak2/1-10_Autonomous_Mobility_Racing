#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32
from std_msgs.msg import Int8MultiArray
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
import tf2_ros
from geometry_msgs.msg import TransformStamped
import math
import json
import os

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

        # ── 디버그 모드 (True: OpenCV 창 표시 / False: 창 없이 토픽만 퍼블리시) ──
        self.debug_mode = True
        # ── RViz 모드 (True: MarkerArray 퍼블리시 / False: 실차 주행 시 스킵) ──
        self.rviz_mode  = True
        # ──────────────────────────────────────────────────────────────────────

        # ── 튜닝 파라미터 파일 경로 (S키로 저장, 다음 실행 때 자동 로드) ──────
        self.params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lane_params.json')
        self._default_params = {'canny_t1': 50, 'canny_t2': 150, 'roi_pct': 65,
                                'white_v': 200, 'white_s': 30, 'min_area': 300, 'fill_w': 15}
        saved = self._load_params()
        # ──────────────────────────────────────────────────────────────────────

        # ── 카메라 부착 파라미터 (부착 후 실측해서 여기만 수정) ──────────────
        self.cam_x     =  0.0    # 카메라 전후 위치 m (차량 뒤쪽이면 음수, 앞쪽이면 양수)
        self.cam_z     =  0.30   # 카메라 높이 m
        self.cam_pitch = -15.0   # 카메라 pitch 각도 deg (아래를 향하면 음수)
        # 픽셀 → 미터 스케일 (바닥 캘리브레이션 후 조정)
        self.scale_x   =  3.0    # 카메라 전방 시야 거리 m (ROI 세로 = 이 거리에 해당)
        self.scale_y   =  2.0    # 카메라 좌우 시야 거리 m (화면 가로 = 이 거리에 해당)
        # ──────────────────────────────────────────────────────────────────────

        if self.debug_mode:
            cv2.namedWindow('Lane Detection')
            cv2.createTrackbar('Canny T1',  'Lane Detection', saved['canny_t1'],  255,  self._noop)
            cv2.createTrackbar('Canny T2',  'Lane Detection', saved['canny_t2'], 255,  self._noop)
            cv2.createTrackbar('ROI Start', 'Lane Detection', saved['roi_pct'],  90,   self._noop)
            cv2.createTrackbar('White V',   'Lane Detection', saved['white_v'],  255,  self._noop)
            cv2.createTrackbar('White S',   'Lane Detection', saved['white_s'],  100,  self._noop)
            cv2.createTrackbar('Min Area',  'Lane Detection', saved['min_area'], 3000, self._noop)
            cv2.createTrackbar('Fill W',    'Lane Detection', saved['fill_w'],   60,   self._noop)

        self.color_pub  = self.create_publisher(Image,        '/camera/color/image_raw', 1)
        self.lane_pub   = self.create_publisher(Image,        '/camera/lane/image_raw',  1)
        self.offset_pub = self.create_publisher(Float32,      '/lane/offset',             1)
        self.marker_pub     = self.create_publisher(MarkerArray, "/lane/markers",       1)
        self.path_left_pub       = self.create_publisher(Path, "/perception/left_lane",      1)
        self.path_center_pub     = self.create_publisher(Path, "/perception/current_lane",   1)
        self.path_right_pub      = self.create_publisher(Path, "/perception/right_lane",     1)
        self.left_boundary_pub   = self.create_publisher(Path, "/perception/left_boundary",  1)
        self.right_boundary_pub  = self.create_publisher(Path, "/perception/right_boundary", 1)
        self.lane_status_pub     = self.create_publisher(Int8MultiArray, "/perception/lane_status", 1)
        # TF broadcaster (camera -> base_link)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._broadcast_camera_tf()
        self.create_subscription(Int32, '/lane/target', self._target_cb, 1)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("Lane Detection 시작")

    def _load_params(self):
        """저장된 lane_params.json 로드. 없으면 기본값 반환."""
        if os.path.exists(self.params_path):
            try:
                with open(self.params_path, 'r') as f:
                    params = json.load(f)
                self.get_logger().info(f"파라미터 로드: {self.params_path}")
                # 기본값에 없는 키 보완
                for k, v in self._default_params.items():
                    params.setdefault(k, v)
                return params
            except Exception as e:
                self.get_logger().warning(f"파라미터 로드 실패, 기본값 사용: {e}")
        return dict(self._default_params)

    def _save_params(self, canny_t1, canny_t2, roi_pct, white_v, white_s, min_area, fill_w):
        """현재 트랙바 값을 lane_params.json에 저장."""
        params = {'canny_t1': canny_t1, 'canny_t2': canny_t2, 'roi_pct': roi_pct,
                  'white_v': white_v, 'white_s': white_s, 'min_area': min_area, 'fill_w': fill_w}
        try:
            with open(self.params_path, 'w') as f:
                json.dump(params, f, indent=2)
            self.get_logger().info(f"파라미터 저장 완료: {self.params_path}  {params}")
        except Exception as e:
            self.get_logger().error(f"파라미터 저장 실패: {e}")

    def _noop(self, _): pass
    def _target_cb(self, msg): self.target_offset_dir = int(msg.data)

    def _broadcast_camera_tf(self):
        """카메라 → base_link TF 브로드캐스트
        카메라 위치: 차량 뒤쪽 중앙, 높이 30cm, 전방을 향함
        ROS 차량 좌표계: x=전방, y=좌측, z=상방
        카메라가 뒤쪽에 달려 전방을 보므로: x 오프셋 음수(뒤쪽)
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id  = 'camera'
        # 위치: 뒤쪽 0m(나중에 실측 후 조정), 높이 0.30m
        t.transform.translation.x = self.cam_x   # 차량 전후 위치 m
        t.transform.translation.y = 0.0           # 좌우 중앙
        t.transform.translation.z = self.cam_z    # 높이 m
        # 회전: 카메라 pitch (아래를 향하면 음수)
        pitch = math.radians(self.cam_pitch)
        t.transform.rotation.x = math.sin(pitch / 2)
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = math.cos(pitch / 2)
        self.tf_broadcaster.sendTransform(t)

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
        보이면 즉시 갱신, 안 보이면 0(왼쪽) 또는 W(오른쪽)으로 리셋.
        오른쪽 노란선이 없는 트랙에서 오른쪽 흰 차선이 잘리는 문제 수정."""
        h   = yellow.shape[0]
        mid = self.W // 2
        bottom_half = yellow[h // 2:, :]
        cols = np.where(bottom_half.any(axis=0))[0]

        left_cols  = cols[cols < mid]
        right_cols = cols[cols >= mid]

        if len(left_cols) > 0:
            self.prev_x_min = int(left_cols.max())
        else:
            self.prev_x_min = 0  # 왼쪽 노란선 없으면 왼쪽 끝으로 리셋

        if len(right_cols) > 0:
            self.prev_x_max = int(right_cols.min())
        else:
            self.prev_x_max = self.W  # 오른쪽 노란선 없으면 오른쪽 끝으로 리셋

        return self.prev_x_min, self.prev_x_max

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("프레임 읽기 실패", throttle_duration_sec=5.0)
            return
        try:
            if self.debug_mode:
                self.canny_t1  = cv2.getTrackbarPos('Canny T1',  'Lane Detection')
                self.canny_t2  = max(self.canny_t1+1, cv2.getTrackbarPos('Canny T2', 'Lane Detection'))
                roi_pct        = cv2.getTrackbarPos('ROI Start', 'Lane Detection')
                white_v        = cv2.getTrackbarPos('White V',   'Lane Detection')
                white_s        = cv2.getTrackbarPos('White S',   'Lane Detection')
                min_area       = max(50, cv2.getTrackbarPos('Min Area', 'Lane Detection'))
                fill_w         = cv2.getTrackbarPos('Fill W',    'Lane Detection')
            else:
                # debug_mode=False 시 저장된 lane_params.json 값 사용
                _p       = self._load_params()
                self.canny_t1 = _p['canny_t1']
                self.canny_t2 = _p['canny_t2']
                roi_pct  = _p['roi_pct']
                white_v  = _p['white_v']
                white_s  = _p['white_s']
                min_area = _p['min_area']
                fill_w   = _p['fill_w']
            self.roi_start = int(self.H * roi_pct / 100)
            roi_h          = self.H - self.roi_start
            cx             = int(self.W / 2)

            white, yellow, roi = self._build_mask(frame, white_v, white_s, fill_w)

            # 노란선 경계 계산 (한쪽만 보여도 이전값으로 보완)
            x_min, x_max = self._yellow_x_bounds(yellow)

            # 노란선의 y축 최대값(가장 아래 픽셀) 계산 → 흰 차선 y 필터 기준
            yel_cols = np.where(yellow.any(axis=1))[0]
            yel_y_max = int(yel_cols.max()) if len(yel_cols) > 0 else roi_h
            # 노란선 하단 기준으로 일정 비율 이상 내려와야 유효한 차선으로 인정
            y_threshold = max(0, yel_y_max - int(roi_h * 0.35))

            yel_dil    = cv2.dilate(yellow, cv2.getStructuringElement(cv2.MORPH_RECT, (25,25)))
            white_only = cv2.bitwise_and(white, cv2.bitwise_not(yel_dil))
            cnts, _    = cv2.findContours(white_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lanes = []
            for c in cnts:
                if cv2.contourArea(c) < min_area: continue
                bot_x = self._contour_bottom_cx(c)
                # x축: 노란선 범위 바깥이면 무시
                if bot_x < x_min or bot_x > x_max: continue
                # y축: 컨투어 하단이 노란선 y_threshold보다 위에 있으면 노이즈로 무시
                pts_y = c[:, 0, 1]
                if pts_y.max() < y_threshold: continue
                lanes.append((cv2.contourArea(c), bot_x, c))
            # 면적 큰 순으로 최대 3개만 선택 후 x좌표 오름차순 정렬
            lanes.sort(key=lambda t: t[0], reverse=True)
            lanes = lanes[:3]
            lanes.sort(key=lambda t: t[1])
            lanes = [(bot_x, c) for (_, bot_x, c) in lanes]

            # 크로스헤어 범위 정의 (세로 절반으로 축소)
            cross_cy   = int(roi_h * 0.85)          # 크로스헤어 중심 y
            cross_h    = int(roi_h * 0.07)           # 세로 반길이 (원래 0.15 → 절반인 0.07)
            cross_w    = int(self.W * 0.04)          # 가로 반길이 (그대로)
            cross_y_top = cross_cy - cross_h
            cross_y_bot = cross_cy + cross_h

            # 크로스헤어 샘플 포인트: 세로선 + 가로선 위의 점들
            cross_pts = []
            for y in range(cross_y_top, cross_y_bot + 1, max(1, cross_h // 5)):
                cross_pts.append((float(cx), float(y)))
            for x in range(cx - cross_w, cx + cross_w + 1, max(1, cross_w // 5)):
                cross_pts.append((float(x), float(cross_cy)))

            # 크로스헤어 어느 점이든 컨투어에 닿으면 해당 차선으로 판정
            current_i = -1
            for i, (bot_x, c) in enumerate(lanes):
                for pt in cross_pts:
                    if cv2.pointPolygonTest(c, pt, False) >= 0:
                        current_i = i
                        break
                if current_i >= 0:
                    break

            if current_i >= 0:
                # ── target에 따라 offset 계산 기준 차선 결정 ──────────────────
                target = self.target_offset_dir

                if target == 1:
                    # 왼쪽 차선 변경 명령
                    if current_i > 0:
                        # 왼쪽 흰선 기준으로 그 차선 중앙 계산
                        if current_i - 1 > 0:
                            ref_left_x  = lanes[current_i - 2][0]
                        else:
                            ref_left_x  = x_min
                        ref_right_x = lanes[current_i][0]
                        center_x = (ref_left_x + ref_right_x) / 2
                    else:
                        # 이미 1차선 — 왼쪽 없음, 명령 무시하고 현재 차선 유지
                        if current_i > 0:
                            left_x = lanes[current_i - 1][0]
                        else:
                            left_x = x_min
                        if current_i < len(lanes) - 1:
                            right_x = lanes[current_i + 1][0]
                        else:
                            right_x = x_max
                        center_x = (left_x + right_x) / 2

                elif target == -1:
                    # 오른쪽 차선 변경 명령
                    if current_i < len(lanes) - 1:
                        # 오른쪽 흰선 기준으로 그 차선 중앙 계산
                        ref_left_x  = lanes[current_i][0]
                        if current_i + 2 < len(lanes):
                            ref_right_x = lanes[current_i + 2][0]
                        else:
                            ref_right_x = x_max
                        center_x = (ref_left_x + ref_right_x) / 2
                    else:
                        # 이미 최우측 차선 — 오른쪽 없음, 명령 무시하고 현재 차선 유지
                        if current_i > 0:
                            left_x = lanes[current_i - 1][0]
                        else:
                            left_x = x_min
                        if current_i < len(lanes) - 1:
                            right_x = lanes[current_i + 1][0]
                        else:
                            right_x = x_max
                        center_x = (left_x + right_x) / 2

                else:
                    # target == 0: 현재 차선 중앙 추종 (평소)
                    if current_i > 0:
                        left_x = lanes[current_i - 1][0]
                    else:
                        left_x = x_min
                    if current_i < len(lanes) - 1:
                        right_x = lanes[current_i + 1][0]
                    else:
                        right_x = x_max
                    center_x = (left_x + right_x) / 2
                # ──────────────────────────────────────────────────────────────

                self.current_lane_x = center_x
                offset = (center_x - cx) / cx
            else:
                offset = 999.0  # 차선 미감지

            if self.debug_mode:
                # 노란선 경계 안쪽 yellow만 시각화에 사용
                yellow_vis = yellow.copy()
                yellow_vis[:, :x_min] = 0
                yellow_vis[:, x_max:] = 0

                # ── 차선 색깔 결정 ──────────────────────────────────────────
                # target = 0 : 현재 차선 초록, 나머지 노란
                # target = 1 : 왼쪽 차선 파란, 현재 차선 노란 (변경 중)
                #              왼쪽 차선 없으면 현재 차선 초록 (명령 무시)
                # target = -1: 오른쪽 차선 파란, 현재 차선 노란 (변경 중)
                #              오른쪽 차선 없으면 현재 차선 초록 (명령 무시)
                _target = self.target_offset_dir
                if current_i >= 0:
                    if _target == 1 and current_i > 0:
                        _green_i = -1
                        _blue_i  = current_i - 1
                    elif _target == -1 and current_i < len(lanes) - 1:
                        _green_i = -1
                        _blue_i  = current_i + 1
                    else:
                        _green_i = current_i
                        _blue_i  = -1
                else:
                    _green_i = -1
                    _blue_i  = -1

                # mask_vis 색칠
                mask_vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
                mask_vis[yellow_vis > 0] = [0, 0, 255]
                for i, (bot_x, c) in enumerate(lanes):
                    if i == _green_i:
                        color = (0, 255, 0)      # 초록: 현재 주행 차선
                    elif i == _blue_i:
                        color = (255, 100, 0)    # 파란: 변경 목표 차선
                    else:
                        color = (0, 255, 255)    # 노란: 나머지
                    cv2.drawContours(mask_vis, [c], -1, color, cv2.FILLED)
                    cv2.circle(mask_vis, (bot_x, roi_h - 15), 5, color, -1)

                # 노란선 경계 시각화
                cv2.line(mask_vis, (x_min, 0), (x_min, roi_h), (0, 180, 0), 1)
                cv2.line(mask_vis, (x_max, 0), (x_max, roi_h), (0, 180, 0), 1)

                # 크로스헤어: 차선 감지 여부만 반영
                cross_color = (0, 255, 0) if current_i >= 0 else (255, 200, 0)
                cv2.line(mask_vis, (cx, cross_y_top), (cx, cross_y_bot), cross_color, 2)
                cv2.line(mask_vis, (cx - cross_w, cross_cy), (cx + cross_w, cross_cy), cross_color, 2)

                # HUD
                cv2.putText(mask_vis, f"offset: {offset:.3f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(mask_vis, f"white lanes: {len(lanes)}", (10, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            stamp = self.get_clock().now().to_msg()

            def centerline_from_contour(contour, deg=2, n_out=15):
                """컨투어 → 행별 중심점 추출 → 다항식 피팅 → 매끄러운 곡선 포인트 반환
                deg: 피팅 차수 (2=포물선, 3=S자 곡선)
                n_out: 출력 포인트 수
                """
                mask = np.zeros((roi_h, self.W), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                raw_pts = []
                rows = np.where(mask.any(axis=1))[0]
                if len(rows) == 0:
                    return []
                for row in rows:
                    cols = np.where(mask[row] > 0)[0]
                    if len(cols) == 0:
                        continue
                    raw_pts.append((int((cols[0] + cols[-1]) / 2), int(row)))
                if len(raw_pts) < deg + 1:
                    return raw_pts
                # 다항식 피팅: y(픽셀 행) → x(픽셀 열)
                ys = np.array([p[1] for p in raw_pts])
                xs = np.array([p[0] for p in raw_pts])
                try:
                    coeffs = np.polyfit(ys, xs, deg)
                except Exception:
                    return raw_pts
                poly  = np.poly1d(coeffs)
                y_out = np.linspace(ys.min(), ys.max(), n_out)
                x_out = np.clip(poly(y_out), 0, self.W - 1)
                return [(int(x), int(y)) for x, y in zip(x_out, y_out)]

            def make_curve_marker(marker_id, pts, r, g, b, thickness=0.007):
                """포인트 리스트 → LINE_STRIP Marker"""
                mk = Marker()
                mk.header.frame_id = "camera"
                mk.header.stamp = stamp
                mk.ns = "lanes"
                mk.id = marker_id
                mk.type = Marker.LINE_STRIP
                mk.action = Marker.ADD
                mk.scale.x = thickness
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 1.0
                for (px, py) in pts:
                    p = Point()
                    p.x = (px - self.W / 2) / self.W
                    p.y = -(py - roi_h / 2) / roi_h
                    p.z = 0.0
                    mk.points.append(p)
                return mk

            # ── 노란선 컨투어 분류 (좌/우) ── (MarkerArray/Path 공통)
            yel_cnts, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mid_x = self.W // 2
            left_yel  = [c for c in yel_cnts if self._contour_bottom_cx(c) <= mid_x and cv2.contourArea(c) > 200]
            right_yel = [c for c in yel_cnts if self._contour_bottom_cx(c) >  mid_x and cv2.contourArea(c) > 200]
            best_left_yel  = max(left_yel,  key=cv2.contourArea) if left_yel  else None
            best_right_yel = max(right_yel, key=cv2.contourArea) if right_yel else None

            # ── centerline 캐싱: 컨투어별로 딱 한 번만 피팅 ──────────────────
            # {contour_id: smooth_pts} 형태로 저장해서 MarkerArray/Path 양쪽에서 재활용
            _cache = {}
            def get_centerline(contour, n_out=15):
                key = id(contour)
                if key not in _cache:
                    _cache[key] = centerline_from_contour(contour, n_out=n_out)
                return _cache[key]

            def pts_to_path(smooth_pts, frame_id='base_link'):
                """캐싱된 centerline 포인트 → nav_msgs/Path 변환"""
                path = Path()
                path.header.stamp    = stamp
                path.header.frame_id = frame_id
                if not smooth_pts:
                    return path
                scale_x = self.scale_x / roi_h
                scale_y = self.scale_y / self.W
                for (cx_row, row) in smooth_pts:
                    pose = PoseStamped()
                    pose.header = path.header
                    pose.pose.position.x = (roi_h - row) * scale_x
                    pose.pose.position.y = (self.W / 2 - cx_row) * scale_y
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    path.poses.append(pose)
                return path
            # ──────────────────────────────────────────────────────────────────

            # ── MarkerArray 퍼블리시 (RViz 시각화, rviz_mode=True일 때만) ──────
            if self.rviz_mode:
                marker_array = MarkerArray()
                for del_ns in ["lanes", "yellow_left", "yellow_right", "white_lane",
                               "yellow_left_outline", "yellow_right_outline", "white_lane_outline",
                               "lane_center", "my_lane", "all"]:
                    dm = Marker()
                    dm.header.frame_id = "camera"
                    dm.header.stamp = stamp
                    dm.ns = del_ns
                    dm.action = Marker.DELETEALL
                    marker_array.markers.append(dm)

                if best_left_yel is not None:
                    pts = get_centerline(best_left_yel)
                    if pts:
                        marker_array.markers.append(make_curve_marker(0, pts, 1.0, 0.0, 0.0))

                for i, (bot_x, c) in enumerate(lanes):
                    pts = get_centerline(c)
                    if not pts:
                        continue
                    if i == current_i:
                        r2, g2, b2 = 0.0, 1.0, 0.0
                    else:
                        r2, g2, b2 = 1.0, 1.0, 0.0
                    marker_array.markers.append(make_curve_marker(i + 1, pts, r2, g2, b2))

                if best_right_yel is not None:
                    pts = get_centerline(best_right_yel)
                    if pts:
                        marker_array.markers.append(make_curve_marker(4, pts, 1.0, 0.0, 0.0))

                self.marker_pub.publish(marker_array)
            # ──────────────────────────────────────────────────────────────────

            # ── /perception/* 토픽 퍼블리시 (캐싱된 centerline 재활용) ────────
            empty_path = Path(); empty_path.header.stamp = stamp; empty_path.header.frame_id = 'base_link'

            if current_i >= 0:
                self.path_center_pub.publish(pts_to_path(get_centerline(lanes[current_i][1])))
                if current_i > 0:
                    self.path_left_pub.publish(pts_to_path(get_centerline(lanes[current_i - 1][1])))
                else:
                    self.path_left_pub.publish(empty_path)
                if current_i < len(lanes) - 1:
                    self.path_right_pub.publish(pts_to_path(get_centerline(lanes[current_i + 1][1])))
                else:
                    self.path_right_pub.publish(empty_path)
            else:
                self.path_left_pub.publish(empty_path)
                self.path_center_pub.publish(empty_path)
                self.path_right_pub.publish(empty_path)

            if best_left_yel is not None:
                self.left_boundary_pub.publish(pts_to_path(get_centerline(best_left_yel)))
            else:
                self.left_boundary_pub.publish(empty_path)

            if best_right_yel is not None:
                self.right_boundary_pub.publish(pts_to_path(get_centerline(best_right_yel)))
            else:
                self.right_boundary_pub.publish(empty_path)
            # ──────────────────────────────────────────────────────────────────

            # ── /perception/lane_status 퍼블리시 ──────────────────────────────
            lane_status = Int8MultiArray()
            lane_status.data = [
                1 if current_i >= 0                else 0,  # data[0] current_lane_detected
                1 if current_i > 0                 else 0,  # data[1] left_lane_detected
                1 if current_i >= 0 and current_i < len(lanes) - 1 else 0,  # data[2] right_lane_detected
                1 if best_left_yel  is not None    else 0,  # data[3] left_boundary_detected
                1 if best_right_yel is not None    else 0,  # data[4] right_boundary_detected
                current_i + 1 if current_i >= 0   else 0,  # data[5] current_lane_id (1-based, 0=미감지)
            ]
            self.lane_status_pub.publish(lane_status)
            # ──────────────────────────────────────────────────────────────────

            # ROS publish
            off = Float32(); off.data = float(offset)
            self.offset_pub.publish(off)

            if self.debug_mode:
                out_img = mask_vis
                cv2.imshow('Lane Detection', np.vstack([roi, mask_vis]))
                m  = self.bridge.cv2_to_imgmsg(frame,   encoding="bgr8"); m.header.stamp  = stamp
                m2 = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8"); m2.header.stamp = stamp
                self.color_pub.publish(m)
                self.lane_pub.publish(m2)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): rclpy.shutdown()
                if k == ord('s'):
                    self._save_params(self.canny_t1, self.canny_t2,
                                      cv2.getTrackbarPos('ROI Start', 'Lane Detection'),
                                      cv2.getTrackbarPos('White V',   'Lane Detection'),
                                      cv2.getTrackbarPos('White S',   'Lane Detection'),
                                      cv2.getTrackbarPos('Min Area',  'Lane Detection'),
                                      cv2.getTrackbarPos('Fill W',    'Lane Detection'))

        except Exception as e:
            self.get_logger().error(f"에러: {e}", throttle_duration_sec=1.0)
            import traceback; traceback.print_exc()

    def destroy_node(self):
        self.cap.release()
        if self.debug_mode:
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