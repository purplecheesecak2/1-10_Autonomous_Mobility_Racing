#!/usr/bin/env python3
"""
Lane Follow Node — Pure Pursuit (Camera Path 기반)
2026 자율주행 모빌리티 레이스 1/10 대회용

[입력]
  /perception/current_lane   Path           - 현재 차선 중심선 (base_link 기준, x=전방, y=좌측)
  /perception/left_lane      Path           - 왼쪽 차선
  /perception/right_lane     Path           - 오른쪽 차선
  /perception/lane_status    Int8MultiArray - [인식, 좌유무, 우유무, 좌경계, 우경계, 인덱스]
  /lane/offset               Float32        - 횡방향 오프셋 (차선 미인식 시 0.0 fallback)
  /front_vehicle/detected    Bool           - 전방 장애물 유무
  /front_vehicle/distances   Float32MultiArray - 전방 장애물 거리 목록 (m)

[출력]
  /drive       AckermannDriveStamped - 조향각(rad) + 속도(m/s) → esp32_control_node
  /lane/target Int32                 - 차선 변경 명령 (0=유지, 1=좌, -1=우) → 카메라 노드
"""

import math
import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import Path
from std_msgs.msg import Float32, Int32
from std_msgs.msg import Int8MultiArray, Float32MultiArray
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDriveStamped


class LaneFollowNode(Node):

    def __init__(self):
        super().__init__('lane_follow_node')

        # ── 튜닝 파라미터 ──────────────────────────────────────────────────
        # 차량
        self.declare_parameter('wheelbase',     0.25)  # 축거 (m) — 실측 후 수정
        # Pure Pursuit lookahead
        # Path 포인트: index 0 = 가장 먼 곳, index N-1 = 차량 바로 앞
        # 작을수록 멀리 봄(부드럽고 느린 반응), 클수록 가까이 봄(민감하고 빠른 반응)
        # 권장 범위: 4(안정) ~ 10(민감), 초기값 7로 시작
        self.declare_parameter('lookahead_idx', 7)
        # 속도
        self.declare_parameter('max_speed',     0.4)   # 최대 속도 (m/s) — 벤치 테스트는 낮게
        self.declare_parameter('min_speed',     0.2)   # 최소 속도 (m/s)
        self.declare_parameter('speed_curve_k', 0.4)   # 조향각 크기에 따른 감속 게인
        # 조향
        # esp32_control 최대 ±45° = ±0.785 rad. 처음엔 여유 있게 설정
        self.declare_parameter('max_steer',     0.52)  # 최대 조향각 (rad) ≈ 30°
        # 장애물 (LiDAR)
        self.declare_parameter('stop_dist',     0.30)  # 긴급 정지 거리 (m)
        self.declare_parameter('avoid_dist',    0.70)  # 차선 변경 트리거 거리 (m)
        self.declare_parameter('lc_cooldown',   2.0)   # 차선 변경 쿨다운 (s)

        self.L          = self.get_parameter('wheelbase').value
        self.lh_idx     = self.get_parameter('lookahead_idx').value
        self.max_spd    = self.get_parameter('max_speed').value
        self.min_spd    = self.get_parameter('min_speed').value
        self.curve_k    = self.get_parameter('speed_curve_k').value
        self.max_steer  = self.get_parameter('max_steer').value
        self.stop_dist  = self.get_parameter('stop_dist').value
        self.avoid_dist = self.get_parameter('avoid_dist').value
        self.lc_cool    = self.get_parameter('lc_cooldown').value

        # ── 상태 변수 ──────────────────────────────────────────────────────
        self.cur_path   : list = []
        self.left_path  : list = []
        self.right_path : list = []

        self.lane_detected : bool  = False   # lane_status[0]
        self.left_exists   : bool  = False   # lane_status[1]
        self.right_exists  : bool  = False   # lane_status[2]
        self.lane_offset   : float = 0.0

        self.obs_detected  : bool  = False
        self.obs_min_dist  : float = 99.0    # 가장 가까운 전방 장애물 거리

        self.target_lane   : int   = 0       # 0=유지, 1=좌, -1=우
        self.last_lc_time  = None

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(Path, '/perception/current_lane', self._cb_cur,    1)
        self.create_subscription(Path, '/perception/left_lane',    self._cb_left,   1)
        self.create_subscription(Path, '/perception/right_lane',   self._cb_right,  1)
        self.create_subscription(Int8MultiArray, '/perception/lane_status',
                                 self._cb_status, 1)
        self.create_subscription(Float32, '/lane/offset',          self._cb_offset, 1)
        self.create_subscription(Bool,    '/front_vehicle/detected',
                                 self._cb_obs_flag, 1)
        self.create_subscription(Float32MultiArray, '/front_vehicle/distances',
                                 self._cb_obs_dist, 1)

        # ── Publishers ─────────────────────────────────────────────────────
        # esp32_control_node.cpp 구독 토픽: /drive (AckermannDriveStamped)
        #   steering_angle 단위: rad  (노드 내부에서 deg로 변환)
        self.drive_pub  = self.create_publisher(AckermannDriveStamped, '/drive',        1)
        self.target_pub = self.create_publisher(Int32,                  '/lane/target', 1)

        # ── 제어 루프 30 Hz ────────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self._control_loop)

        self.get_logger().info(
            f'LaneFollow(PurePursuit) 시작\n'
            f'  wheelbase={self.L}m  lookahead_idx={self.lh_idx}\n'
            f'  speed={self.min_spd}~{self.max_spd} m/s\n'
            f'  max_steer=±{math.degrees(self.max_steer):.0f}°\n'
            f'  → /drive (AckermannDriveStamped, rad)')

    # ── Callbacks ──────────────────────────────────────────────────────────

    @staticmethod
    def _path_to_list(msg: Path) -> list:
        return [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _cb_cur(self,   msg): self.cur_path   = self._path_to_list(msg)
    def _cb_left(self,  msg): self.left_path  = self._path_to_list(msg)
    def _cb_right(self, msg): self.right_path = self._path_to_list(msg)

    def _cb_status(self, msg):
        if len(msg.data) >= 3:
            self.lane_detected = bool(msg.data[0])
            self.left_exists   = bool(msg.data[1])
            self.right_exists  = bool(msg.data[2])

    def _cb_offset(self, msg):
        self.lane_offset = msg.data

    def _cb_obs_flag(self, msg):
        self.obs_detected = msg.data

    def _cb_obs_dist(self, msg):
        if msg.data:
            self.obs_min_dist = float(min(msg.data))
        else:
            self.obs_min_dist = 99.0
            self.obs_detected = False

    # ── Pure Pursuit ───────────────────────────────────────────────────────

    def _get_lookahead(self, path: list):
        """Path 배열에서 lookahead_idx번째 포인트를 반환."""
        if not path:
            return None
        return path[min(self.lh_idx, len(path) - 1)]

    def _pure_pursuit(self, tx: float, ty: float) -> float:
        """
        base_link 기준 목표점 (tx=전방, ty=좌측 양수) → 조향각 (rad)
        delta = atan2(2 * L * sin(alpha), ld)
        """
        ld = math.hypot(tx, ty)
        if ld < 1e-4:
            return 0.0
        alpha = math.atan2(ty, tx)
        delta = math.atan2(2.0 * self.L * math.sin(alpha), ld)
        return float(np.clip(delta, -self.max_steer, self.max_steer))

    def _offset_fallback(self) -> float:
        """Path 없을 때 offset 기반 단순 비례 조향.
        offset = (center_x - img_cx) / img_cx
          > 0 : 차선 중심이 이미지 우측 → 차가 좌측으로 치우침 → 우회전 (음수 조향)
        """
        return float(np.clip(
            -self.lane_offset * self.max_steer * 1.5,
            -self.max_steer, self.max_steer))

    # ── 차선 변경 결정 ─────────────────────────────────────────────────────

    def _decide_lane_change(self):
        """전방 장애물에 따라 target_lane 결정 후 카메라 노드에 전달."""
        now = self.get_clock().now()

        # 쿨다운 중: 변경 금지
        if self.last_lc_time is not None:
            if (now - self.last_lc_time).nanoseconds * 1e-9 < self.lc_cool:
                return

        # 전방 여유 충분 → 현재 차선 유지
        if not self.obs_detected or self.obs_min_dist >= self.avoid_dist:
            if self.target_lane != 0:
                self.target_lane = 0
                self._pub_target()
            return

        # 회피 필요
        left_ok  = self.left_exists
        right_ok = self.right_exists
        new_target = self.target_lane

        if left_ok and right_ok:
            new_target = 1    # 일단 좌측 우선 (대회 규정상 추월 방향 무관)
        elif left_ok:
            new_target = 1
        elif right_ok:
            new_target = -1
        # 양쪽 모두 없으면 감속만

        if new_target != self.target_lane:
            self.target_lane  = new_target
            self.last_lc_time = now
            self._pub_target()
            self.get_logger().info(
                f'차선 변경 → {new_target:+d}  '
                f'전방={self.obs_min_dist:.2f}m')

    def _pub_target(self):
        msg = Int32()
        msg.data = self.target_lane
        self.target_pub.publish(msg)

    # ── 메인 제어 루프 ─────────────────────────────────────────────────────

    def _send_drive(self, steer_rad: float, speed: float):
        """AckermannDriveStamped 발행. steering_angle 단위: rad"""
        msg = AckermannDriveStamped()
        msg.header.stamp        = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer_rad)
        msg.drive.speed          = float(speed)
        self.drive_pub.publish(msg)

    def _control_loop(self):

        # ① 차선 미인식 → 정지
        if not self.lane_detected:
            self._send_drive(0.0, 0.0)
            return

        # ② 긴급 정지 (전방 장애물 너무 가까움)
        if self.obs_detected and self.obs_min_dist <= self.stop_dist:
            self._send_drive(0.0, 0.0)
            self.get_logger().warn(
                f'긴급 정지! 전방={self.obs_min_dist:.2f}m',
                throttle_duration_sec=1.0)
            return

        # ③ 차선 변경 결정
        self._decide_lane_change()

        # ④ 목표 Path 선택
        if self.target_lane == 1 and self.left_path:
            path = self.left_path
        elif self.target_lane == -1 and self.right_path:
            path = self.right_path
        else:
            path = self.cur_path

        # ⑤ Pure Pursuit 조향각 (rad)
        target_pt = self._get_lookahead(path)
        if target_pt is not None:
            steer = self._pure_pursuit(target_pt[0], target_pt[1])
        else:
            steer = self._offset_fallback()

        # ⑥ 속도 결정
        #   조향각 클수록 감속 / 장애물 가까울수록 감속
        curvature = abs(math.tan(steer) / self.L) if self.L > 0 else 0.0
        if self.obs_detected:
            obs_ratio = float(np.clip(
                (self.obs_min_dist - self.stop_dist) /
                max(self.avoid_dist - self.stop_dist, 1e-6),
                0.0, 1.0))
        else:
            obs_ratio = 1.0

        speed = float(max(
            self.min_spd,
            self.max_spd * obs_ratio - self.curve_k * curvature))

        self._send_drive(steer, speed)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
