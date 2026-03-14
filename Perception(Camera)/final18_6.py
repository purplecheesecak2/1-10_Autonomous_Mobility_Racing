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


# ── 상수 ────────────────────────────────────────────────────────────────────
OFFSET_INVALID  = float('nan')   # 차선 미검출 시 offset 값
_X_COAST_MAX    = 30             # 황색선 소실 후 유지할 최대 프레임 수


class LaneDetectionFinal(Node):
    def __init__(self):
        super().__init__('lane_detection_final_node')
        self.bridge = CvBridge()

        self.calib = self._load_or_build_calib()

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다!")
            return
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.W, self.H = 640, 480
        self._init_undistort_map()

        self.roi_start = int(self.H * 0.65)

        self.canny_t1 = 50
        self.canny_t2 = 150

        self.target_offset_dir = 0
        self.current_lane_x    = self.W / 2

        self.prev_x_min   = 0
        self.prev_x_max   = self.W
        self._x_min_coast = 0
        self._x_max_coast = 0

        self.debug_mode = True
        self.rviz_mode  = True

        # ── 파라미터 관련 ──────────────────────────────────────────────────
        self.params_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'lane_params.json')
        self._default_params = {
            'canny_t1': 50, 'canny_t2': 150, 'roi_pct': 65,
            'white_v': 200, 'white_s': 30, 'min_area': 300, 'fill_w': 15,
        }
        self._params      = self._load_params()
        self._param_mtime = self._get_param_mtime()   # 파일 변경 감지용

        # ── 카메라 외부 파라미터 ───────────────────────────────────────────
        self.cam_x     =  0.06
        self.cam_z     =  0.235
        self.cam_pitch = -0.4
        self.scale_x   =  240
        self.scale_y   =  3.2

        # ── 디버그 윈도우 & 트랙바 ─────────────────────────────────────────
        if self.debug_mode:
            cv2.namedWindow('Lane Detection')
            p = self._params
            cv2.createTrackbar('Canny T1',  'Lane Detection', p['canny_t1'],  255,  self._noop)
            cv2.createTrackbar('Canny T2',  'Lane Detection', p['canny_t2'],  255,  self._noop)
            cv2.createTrackbar('ROI Start', 'Lane Detection', p['roi_pct'],   90,   self._noop)
            cv2.createTrackbar('White V',   'Lane Detection', p['white_v'],   255,  self._noop)
            cv2.createTrackbar('White S',   'Lane Detection', p['white_s'],   100,  self._noop)
            cv2.createTrackbar('Min Area',  'Lane Detection', p['min_area'],  3000, self._noop)
            cv2.createTrackbar('Fill W',    'Lane Detection', p['fill_w'],    60,   self._noop)

        # ── ROS 퍼블리셔 ───────────────────────────────────────────────────
        self.color_pub           = self.create_publisher(Image,        '/camera/color/image_raw',      1)
        self.lane_pub            = self.create_publisher(Image,        '/camera/lane/image_raw',       1)
        self.offset_pub          = self.create_publisher(Float32,      '/lane/offset',                 1)
        self.marker_pub          = self.create_publisher(MarkerArray,  '/lane/markers',                1)
        self.path_left_pub       = self.create_publisher(Path,         '/perception/left_lane',        1)
        self.path_center_pub     = self.create_publisher(Path,         '/perception/current_lane',     1)
        self.path_right_pub      = self.create_publisher(Path,         '/perception/right_lane',       1)
        self.left_boundary_pub   = self.create_publisher(Path,         '/perception/left_boundary',    1)
        self.right_boundary_pub  = self.create_publisher(Path,         '/perception/right_boundary',   1)
        self.lane_status_pub     = self.create_publisher(Int8MultiArray, '/perception/lane_status',    1)

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._broadcast_camera_tf()
        self.create_subscription(Int32, '/lane/target', self._target_cb, 1)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("Lane Detection 시작")

    # ════════════════════════════════════════════════════════════════════════
    # 초기화 헬퍼
    # ════════════════════════════════════════════════════════════════════════

    def _load_or_build_calib(self) -> dict:
        fx, fy = 1009.5, 1075.4
        cx, cy = 620.6543, 346.9380
        k1, k2, k3 =  0.0903, -0.1301, 0.0
        p1, p2     =  0.0, 0.0
        camera_matrix = np.array([[fx,  0, cx],
                                  [ 0, fy, cy],
                                  [ 0,  0,  1]], dtype=np.float64)
        dist_coeffs   = np.array([[k1, k2, p1, p2, k3]], dtype=np.float64)
        return {'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs}

    def _init_undistort_map(self):
        cam_mtx = self.calib['camera_matrix']
        dist    = self.calib['dist_coeffs']
        new_mtx, self._undist_roi = cv2.getOptimalNewCameraMatrix(
            cam_mtx, dist, (self.W, self.H), alpha=0)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            cam_mtx, dist, None, new_mtx, (self.W, self.H), cv2.CV_16SC2)
        self.get_logger().info("왜곡 보정 맵 초기화 완료")

    def _undistort(self, img: np.ndarray) -> np.ndarray:
        dst = cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)
        x, y, rw, rh = self._undist_roi
        if rw > 0 and rh > 0:
            dst = dst[y:y+rh, x:x+rw]
            if dst.shape[1] != self.W or dst.shape[0] != self.H:
                self.W, self.H  = dst.shape[1], dst.shape[0]
                self.prev_x_max = self.W
                self.current_lane_x = self.W / 2
        return dst

    def _broadcast_camera_tf(self):
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id  = 'camera'
        t.transform.translation.x = self.cam_x
        t.transform.translation.y = 0.0
        t.transform.translation.z = self.cam_z
        pitch = math.radians(self.cam_pitch)
        t.transform.rotation.x = math.sin(pitch / 2)
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = math.cos(pitch / 2)
        self.tf_broadcaster.sendTransform(t)

    # ════════════════════════════════════════════════════════════════════════
    # 파라미터 IO  (파일 변경 시에만 재로드)
    # ════════════════════════════════════════════════════════════════════════

    def _get_param_mtime(self) -> float:
        try:
            return os.path.getmtime(self.params_path)
        except OSError:
            return 0.0

    def _load_params(self) -> dict:
        if os.path.exists(self.params_path):
            try:
                with open(self.params_path, 'r') as f:
                    params = json.load(f)
                self.get_logger().info(f"파라미터 로드: {self.params_path}")
                for k, v in self._default_params.items():
                    params.setdefault(k, v)
                return params
            except Exception as e:
                self.get_logger().warning(f"파라미터 로드 실패, 기본값 사용: {e}")
        return dict(self._default_params)

    def _maybe_reload_params(self):
        """파일이 변경됐을 때만 재로드 (disk I/O 최소화)."""
        mtime = self._get_param_mtime()
        if mtime != self._param_mtime:
            self._params      = self._load_params()
            self._param_mtime = mtime

    def _save_params(self, canny_t1, canny_t2, roi_pct,
                     white_v, white_s, min_area, fill_w):
        params = {
            'canny_t1': canny_t1, 'canny_t2': canny_t2, 'roi_pct': roi_pct,
            'white_v': white_v,   'white_s':  white_s,
            'min_area': min_area, 'fill_w':   fill_w,
        }
        try:
            with open(self.params_path, 'w') as f:
                json.dump(params, f, indent=2)
            self._params      = params
            self._param_mtime = self._get_param_mtime()
            self.get_logger().info(f"파라미터 저장 완료: {params}")
        except Exception as e:
            self.get_logger().error(f"파라미터 저장 실패: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # 콜백
    # ════════════════════════════════════════════════════════════════════════

    def _noop(self, _): pass

    def _target_cb(self, msg):
        self.target_offset_dir = int(msg.data)

    # ════════════════════════════════════════════════════════════════════════
    # 이미지 처리 메서드
    # ════════════════════════════════════════════════════════════════════════

    def _build_mask(self, frame, white_v, white_s, fill_w):
        roi = frame[self.roi_start:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        k   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        white = cv2.inRange(hsv,
                            np.array([0,   0,   white_v]),
                            np.array([180, white_s, 255]))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,  k, iterations=2)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)

        yellow = cv2.inRange(hsv,
                             np.array([10,  50,  80]),
                             np.array([40, 255, 255]))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN,  k)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0),
                          self.canny_t1, self.canny_t2)
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(yellow))

        fill_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, fill_w), 3))
        edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, fill_k, iterations=2)

        white = cv2.bitwise_or(white, edges)
        return white, yellow, roi

    def _contour_bottom_cx(self, c) -> int:
        """컨투어 하단부 x 중심을 반환. 빈 결과에 대한 fallback 처리 포함."""
        pts      = c[:, 0, :]
        max_y    = pts[:, 1].max()
        thresh_y = max_y - max(1, int((max_y - pts[:, 1].min()) * 0.1))
        bottom   = pts[pts[:, 1] >= thresh_y][:, 0]
        if len(bottom) == 0:
            return int(pts[:, 0].mean())   # fallback: 전체 x 평균
        return int(bottom.mean())

    def _yellow_x_bounds(self, yellow, best_left_yel, best_right_yel):
        """황색선 컨투어 우선, 없으면 픽셀 열 기반으로 x 경계 추정."""
        h   = yellow.shape[0]
        mid = self.W // 2

        if best_right_yel is not None:
            self.prev_x_max   = self._contour_bottom_cx(best_right_yel)
            self._x_max_coast = 0
        else:
            bottom_zone = yellow[int(h * 0.75):, :]
            right_cols  = np.where(bottom_zone.any(axis=0))[0]
            right_cols  = right_cols[right_cols >= mid]
            if len(right_cols) > 0:
                self.prev_x_max   = int(right_cols.min())
                self._x_max_coast = 0
            else:
                self._x_max_coast += 1
                if self._x_max_coast > _X_COAST_MAX:
                    self.prev_x_max = self.W

        if best_left_yel is not None:
            self.prev_x_min   = self._contour_bottom_cx(best_left_yel)
            self._x_min_coast = 0
        else:
            bottom_zone = yellow[int(h * 0.75):, :]
            left_cols   = np.where(bottom_zone.any(axis=0))[0]
            left_cols   = left_cols[left_cols < mid]
            if len(left_cols) > 0:
                self.prev_x_min   = int(left_cols.max())
                self._x_min_coast = 0
            else:
                self._x_min_coast += 1
                if self._x_min_coast > _X_COAST_MAX:
                    self.prev_x_min = 0

        return self.prev_x_min, self.prev_x_max

    # ════════════════════════════════════════════════════════════════════════
    # 기하학 / RViz 헬퍼 메서드
    # ════════════════════════════════════════════════════════════════════════

    def _centerline_from_contour(self, contour, roi_h: int,
                                  deg: int = 2, n_out: int = 40) -> list:
        """컨투어에서 중심선 픽셀 좌표 목록을 반환."""
        mask = np.zeros((roi_h, self.W), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        rows = np.where(mask.any(axis=1))[0]
        if len(rows) == 0:
            return []

        raw_pts = []
        for row in rows:
            cols = np.where(mask[row] > 0)[0]
            if len(cols) == 0:
                continue
            raw_pts.append((int((cols[0] + cols[-1]) / 2), int(row)))

        if len(raw_pts) < deg + 1:
            return raw_pts

        ys = np.array([p[1] for p in raw_pts])
        xs = np.array([p[0] for p in raw_pts])
        try:
            coeffs = np.polyfit(ys, xs, deg)
        except np.linalg.LinAlgError:
            return raw_pts

        poly  = np.poly1d(coeffs)
        y_out = np.linspace(ys.min(), ys.max(), n_out)
        x_out = np.clip(poly(y_out), 0, self.W - 1)
        return [(int(x), int(y)) for x, y in zip(x_out, y_out)]

    def _make_curve_marker(self, marker_id, pts, r, g, b,
                           stamp, thickness: float = 0.007) -> Marker:
        mk = Marker()
        mk.header.frame_id = 'camera'
        mk.header.stamp    = stamp
        mk.ns              = 'lanes'
        mk.id              = marker_id
        mk.type            = Marker.LINE_STRIP
        mk.action          = Marker.ADD
        mk.scale.x         = thickness
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 1.0
        roi_h = self.H - self.roi_start
        for (px, py) in pts:
            p   = Point()
            p.x = (px - self.W / 2) / self.W
            p.y = -(py - roi_h / 2) / roi_h
            p.z = 0.0
            mk.points.append(p)
        return mk

    def _pts_to_path(self, smooth_pts: list, stamp,
                     frame_id: str = 'base_link') -> Path:
        path = Path()
        path.header.stamp    = stamp
        path.header.frame_id = frame_id
        if not smooth_pts:
            return path
        roi_h   = self.H - self.roi_start
        scale_x = self.scale_x / roi_h
        scale_y = self.scale_y / self.W
        for (cx_row, row) in smooth_pts:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = (roi_h - row)          * scale_x
            pose.pose.position.y = (self.W / 2 - cx_row)  * scale_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    # ════════════════════════════════════════════════════════════════════════
    # 메인 루프
    # ════════════════════════════════════════════════════════════════════════

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("프레임 읽기 실패",
                                      throttle_duration_sec=5.0)
            return
        try:
            frame = self._undistort(frame)

            # ── 파라미터 읽기 ─────────────────────────────────────────────
            if self.debug_mode:
                self.canny_t1 = cv2.getTrackbarPos('Canny T1',  'Lane Detection')
                self.canny_t2 = max(self.canny_t1 + 1,
                                    cv2.getTrackbarPos('Canny T2', 'Lane Detection'))
                roi_pct   = cv2.getTrackbarPos('ROI Start', 'Lane Detection')
                white_v   = cv2.getTrackbarPos('White V',   'Lane Detection')
                white_s   = cv2.getTrackbarPos('White S',   'Lane Detection')
                min_area  = max(50, cv2.getTrackbarPos('Min Area', 'Lane Detection'))
                fill_w    = cv2.getTrackbarPos('Fill W',    'Lane Detection')
            else:
                self._maybe_reload_params()          # 변경됐을 때만 재로드
                p         = self._params
                self.canny_t1 = p['canny_t1']
                self.canny_t2 = p['canny_t2']
                roi_pct   = p['roi_pct']
                white_v   = p['white_v']
                white_s   = p['white_s']
                min_area  = p['min_area']
                fill_w    = p['fill_w']

            self.roi_start = int(self.H * roi_pct / 100)
            roi_h          = self.H - self.roi_start
            cx             = int(self.W / 2)

            # ── 마스크 생성 ───────────────────────────────────────────────
            white, yellow, _roi = self._build_mask(frame, white_v, white_s, fill_w)

            # ── 황색선 컨투어 ─────────────────────────────────────────────
            yel_cnts, _ = cv2.findContours(
                yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mid_x = self.W // 2
            left_yel  = [c for c in yel_cnts
                         if self._contour_bottom_cx(c) <= mid_x
                         and cv2.contourArea(c) > 200]
            right_yel = [c for c in yel_cnts
                         if self._contour_bottom_cx(c) >  mid_x
                         and cv2.contourArea(c) > 200]
            best_left_yel  = max(left_yel,  key=cv2.contourArea) if left_yel  else None
            best_right_yel = max(right_yel, key=cv2.contourArea) if right_yel else None

            # ── x 경계 ────────────────────────────────────────────────────
            x_min, x_max = self._yellow_x_bounds(yellow,
                                                  best_left_yel,
                                                  best_right_yel)

            # ── 흰색 차선 컨투어 ──────────────────────────────────────────
            yel_dil    = cv2.dilate(yellow,
                                    cv2.getStructuringElement(
                                        cv2.MORPH_RECT, (11, 11)))
            white_only = cv2.bitwise_and(white, cv2.bitwise_not(yel_dil))
            cnts, _    = cv2.findContours(
                white_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            y_threshold = int(roi_h * 0.40)
            yellow_visible = (best_left_yel is not None or best_right_yel is not None)

            lanes = []
            for c in cnts:
                if cv2.contourArea(c) < min_area:
                    continue
                bot_x = self._contour_bottom_cx(c)
                pts_x = c[:, 0, 0]
                pts_y = c[:, 0, 1]
                if pts_y.max() < y_threshold:
                    continue
                if yellow_visible:
                    if bot_x < x_min or bot_x > x_max:
                        continue
                    x_median = int(np.median(pts_x))
                    if x_median < x_min or x_median > x_max:
                        continue
                    in_range = np.sum((pts_x >= x_min) & (pts_x <= x_max))
                    if in_range / len(pts_x) < 0.8:
                        continue
                lanes.append((cv2.contourArea(c), bot_x, c))

            lanes.sort(key=lambda t: t[0], reverse=True)
            lanes = lanes[:3]
            lanes.sort(key=lambda t: t[1])
            lanes = [(bot_x, c) for (_, bot_x, c) in lanes]

            # ── 현재 차선 인덱스 ──────────────────────────────────────────
            current_i = -1
            if lanes:
                probe_pt = (float(cx), float(int(roi_h * 0.85)))
                for i, (bot_x, c) in enumerate(lanes):
                    if cv2.pointPolygonTest(c, probe_pt, False) >= 0:
                        current_i = i
                        break
                if current_i < 0:
                    dists     = [abs(bot_x - self.current_lane_x)
                                 for (bot_x, _) in lanes]
                    current_i = int(np.argmin(dists))

            # ── 목표 차선 중심 계산 ───────────────────────────────────────
            if current_i >= 0:
                center_x = self._calc_center_x(
                    lanes, current_i, x_min, x_max,
                    self.target_offset_dir)
                self.current_lane_x = center_x
                offset = (center_x - cx) / cx
            else:
                offset = OFFSET_INVALID

            # ── 중심선 캐시 (tobytes 기반, 안전한 해시) ───────────────────
            _cl_cache: dict = {}

            def get_centerline(contour):
                key = contour.tobytes()
                if key not in _cl_cache:
                    _cl_cache[key] = self._centerline_from_contour(
                        contour, roi_h)
                return _cl_cache[key]

            stamp = self.get_clock().now().to_msg()
            empty_path = Path()
            empty_path.header.stamp    = stamp
            empty_path.header.frame_id = 'base_link'

            # ── RViz 마커 퍼블리시 ────────────────────────────────────────
            if self.rviz_mode:
                self._publish_markers(
                    stamp, lanes, current_i,
                    best_left_yel, best_right_yel, get_centerline)

            # ── Path 퍼블리시 ─────────────────────────────────────────────
            if current_i >= 0:
                self.path_center_pub.publish(
                    self._pts_to_path(get_centerline(lanes[current_i][1]), stamp))
                if current_i > 0:
                    self.path_left_pub.publish(
                        self._pts_to_path(get_centerline(lanes[current_i - 1][1]), stamp))
                else:
                    self.path_left_pub.publish(empty_path)
                if current_i < len(lanes) - 1:
                    self.path_right_pub.publish(
                        self._pts_to_path(get_centerline(lanes[current_i + 1][1]), stamp))
                else:
                    self.path_right_pub.publish(empty_path)
            else:
                self.path_left_pub.publish(empty_path)
                self.path_center_pub.publish(empty_path)
                self.path_right_pub.publish(empty_path)

            if best_left_yel is not None:
                self.left_boundary_pub.publish(
                    self._pts_to_path(get_centerline(best_left_yel), stamp))
            else:
                self.left_boundary_pub.publish(empty_path)

            if best_right_yel is not None:
                self.right_boundary_pub.publish(
                    self._pts_to_path(get_centerline(best_right_yel), stamp))
            else:
                self.right_boundary_pub.publish(empty_path)

            # ── lane_status 퍼블리시 ──────────────────────────────────────
            lane_status      = Int8MultiArray()
            lane_status.data = [
                1 if current_i >= 0 else 0,
                1 if current_i > 0  else 0,
                1 if current_i >= 0 and current_i < len(lanes) - 1 else 0,
                1 if best_left_yel  is not None else 0,
                1 if best_right_yel is not None else 0,
                current_i + 1 if current_i >= 0 else 0,
            ]
            self.lane_status_pub.publish(lane_status)

            # ── offset 퍼블리시 ───────────────────────────────────────────
            off      = Float32()
            off.data = 0.0 if math.isnan(offset) else float(offset)
            # 구독자가 nan을 처리하지 못하는 경우를 위해
            # /lane/offset 은 0.0 fallback, 상세 상태는 lane_status로 판단
            self.offset_pub.publish(off)

            # ── 원본 프레임 오버레이 ──────────────────────────────────────
            overlay = self._draw_overlay(
                frame, lanes, current_i,
                best_left_yel, best_right_yel,
                offset, get_centerline)

            # ── 디버그 윈도우 & 이미지 퍼블리시 ──────────────────────────
            if self.debug_mode:
                mask_vis = self._build_debug_mask_vis(
                    white, yellow, lanes, current_i,
                    x_min, x_max, cx, roi_h, offset)

                cv2.imshow('Lane Detection', np.vstack([overlay, mask_vis]))
                m  = self.bridge.cv2_to_imgmsg(frame,    encoding='bgr8')
                m2 = self.bridge.cv2_to_imgmsg(mask_vis, encoding='bgr8')
                m.header.stamp = m2.header.stamp = stamp
                self.color_pub.publish(m)
                self.lane_pub.publish(m2)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    rclpy.shutdown()
                elif k == ord('s'):
                    self._save_params(
                        self.canny_t1,
                        self.canny_t2,
                        cv2.getTrackbarPos('ROI Start', 'Lane Detection'),
                        cv2.getTrackbarPos('White V',   'Lane Detection'),
                        cv2.getTrackbarPos('White S',   'Lane Detection'),
                        cv2.getTrackbarPos('Min Area',  'Lane Detection'),
                        cv2.getTrackbarPos('Fill W',    'Lane Detection'))

        except Exception as e:
            self.get_logger().error(f"에러: {e}",
                                    throttle_duration_sec=1.0)
            import traceback; traceback.print_exc()

    # ════════════════════════════════════════════════════════════════════════
    # 차선 중심 계산 (분리된 메서드)
    # ════════════════════════════════════════════════════════════════════════

    def _calc_center_x(self, lanes, current_i, x_min, x_max, target) -> float:
        """
        target:  0 = 현재 차선 유지
                 1 = 왼쪽 차선으로 이동
                -1 = 오른쪽 차선으로 이동
        """
        n = len(lanes)

        if target == 1:
            if current_i > 0:
                # 왼쪽 차선으로 실제 이동
                ref_left  = lanes[current_i - 2][0] if current_i - 2 >= 0 else x_min
                ref_right = lanes[current_i][0]
                return (ref_left + ref_right) / 2.0
            else:
                # 이미 가장 왼쪽 — 이동 불가, 명시적으로 현재 유지
                self.get_logger().warning(
                    "왼쪽 차선 없음 — 현재 차선 유지", throttle_duration_sec=2.0)
                return self.current_lane_x

        elif target == -1:
            if current_i < n - 1:
                # 오른쪽 차선으로 실제 이동
                ref_left  = lanes[current_i][0]
                ref_right = lanes[current_i + 2][0] if current_i + 2 < n else x_max
                return (ref_left + ref_right) / 2.0
            else:
                # 이미 가장 오른쪽 — 이동 불가
                self.get_logger().warning(
                    "오른쪽 차선 없음 — 현재 차선 유지", throttle_duration_sec=2.0)
                return self.current_lane_x

        else:  # target == 0 : 현재 차선 유지
            left_x  = lanes[current_i - 1][0] if current_i > 0     else x_min
            right_x = lanes[current_i + 1][0] if current_i < n - 1 else x_max
            return (left_x + right_x) / 2.0

    # ════════════════════════════════════════════════════════════════════════
    # 시각화 헬퍼 메서드
    # ════════════════════════════════════════════════════════════════════════

    def _publish_markers(self, stamp, lanes, current_i,
                         best_left_yel, best_right_yel, get_centerline):
        marker_array = MarkerArray()
        # 이전 마커 전체 삭제
        for ns in ["lanes", "yellow_left", "yellow_right", "white_lane",
                   "yellow_left_outline", "yellow_right_outline",
                   "white_lane_outline", "lane_center", "my_lane", "all"]:
            dm = Marker()
            dm.header.frame_id = 'camera'
            dm.header.stamp    = stamp
            dm.ns              = ns
            dm.action          = Marker.DELETEALL
            marker_array.markers.append(dm)

        if best_left_yel is not None:
            pts = get_centerline(best_left_yel)
            if pts:
                marker_array.markers.append(
                    self._make_curve_marker(0, pts, 1.0, 0.0, 0.0, stamp))

        for i, (bot_x, c) in enumerate(lanes):
            pts = get_centerline(c)
            if not pts:
                continue
            r2, g2, b2 = (0.0, 1.0, 0.0) if i == current_i else (1.0, 1.0, 0.0)
            marker_array.markers.append(
                self._make_curve_marker(i + 1, pts, r2, g2, b2, stamp))

        if best_right_yel is not None:
            pts = get_centerline(best_right_yel)
            if pts:
                marker_array.markers.append(
                    self._make_curve_marker(4, pts, 1.0, 0.0, 0.0, stamp))

        self.marker_pub.publish(marker_array)

    def _draw_overlay(self, frame, lanes, current_i,
                      best_left_yel, best_right_yel,
                      offset, get_centerline) -> np.ndarray:
        """원본 프레임 위에 중심선 오버레이를 그려 반환."""
        overlay = frame.copy()

        def draw_line(pts_roi, color, thickness=2):
            if len(pts_roi) < 2:
                return
            pts_full = [(x, y + self.roi_start) for (x, y) in pts_roi]
            cv2.polylines(overlay,
                          [np.array(pts_full, dtype=np.int32)],
                          isClosed=False,
                          color=color,
                          thickness=thickness)

        if best_left_yel is not None:
            draw_line(get_centerline(best_left_yel), (0, 0, 255))
        if best_right_yel is not None:
            draw_line(get_centerline(best_right_yel), (0, 0, 255))

        for i, (bot_x, c) in enumerate(lanes):
            color = (0, 255, 0) if i == current_i else (0, 255, 255)
            draw_line(get_centerline(c), color)

        offset_str = f"offset: {offset:.3f}" if not math.isnan(offset) else "offset: N/A"
        cv2.putText(overlay, offset_str, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return overlay

    def _build_debug_mask_vis(self, white, yellow, lanes, current_i,
                               x_min, x_max, cx, roi_h, offset) -> np.ndarray:
        """디버그용 마스크 시각화 이미지를 생성."""
        yellow_vis = yellow.copy()
        yellow_vis[:, :x_min] = 0
        yellow_vis[:, x_max:] = 0

        _target = self.target_offset_dir
        if current_i >= 0:
            if _target == 1 and current_i > 0:
                _green_i, _blue_i = -1, current_i - 1
            elif _target == -1 and current_i < len(lanes) - 1:
                _green_i, _blue_i = -1, current_i + 1
            else:
                _green_i, _blue_i = current_i, -1
        else:
            _green_i, _blue_i = -1, -1

        mask_vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
        mask_vis[yellow_vis > 0] = [0, 0, 255]

        for i, (bot_x, c) in enumerate(lanes):
            color = (0, 255, 0) if i == _green_i else (0, 255, 255)
            cv2.drawContours(mask_vis, [c], -1, color, cv2.FILLED)
            cv2.circle(mask_vis, (bot_x, roi_h - 15), 5, color, -1)

        cv2.line(mask_vis, (x_min, 0), (x_min, roi_h), (0, 180, 0), 1)
        cv2.line(mask_vis, (x_max, 0), (x_max, roi_h), (0, 180, 0), 1)
        cv2.circle(mask_vis, (cx, int(roi_h * 0.85)), 6,
                   (0, 255, 0) if current_i >= 0 else (255, 200, 0), -1)

        offset_str = f"offset: {offset:.3f}" if not math.isnan(offset) else "offset: N/A"
        cv2.putText(mask_vis, offset_str, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(mask_vis, f"white lanes: {len(lanes)}", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return mask_vis

    # ════════════════════════════════════════════════════════════════════════
    # 종료
    # ════════════════════════════════════════════════════════════════════════

    def destroy_node(self):
        if self.debug_mode:
            # 현재 트랙바 값으로 마지막 파라미터 자동 저장
            try:
                self._save_params(
                    self.canny_t1,
                    self.canny_t2,
                    cv2.getTrackbarPos('ROI Start', 'Lane Detection'),
                    cv2.getTrackbarPos('White V',   'Lane Detection'),
                    cv2.getTrackbarPos('White S',   'Lane Detection'),
                    cv2.getTrackbarPos('Min Area',  'Lane Detection'),
                    cv2.getTrackbarPos('Fill W',    'Lane Detection'))
            except Exception:
                pass
            cv2.destroyAllWindows()
        self.cap.release()
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