#!/usr/bin/env python3
"""
Front Vehicle Detector (LiDAR only) - ROS2 rclpy

[목표]
- 벽/콘/기둥 같은 환경물은 거의 없고, "앞에는 상대 차량만 있을 가능성"이 높다는 가정에서
  LiDAR(/scan)만으로 전방 차량 존재 여부를 감지한다.
- 여기서는 "차량 분류"가 아니라, 전방 ROI에서 차량 크기처럼 보이는 물체 클러스터를 찾아
  '앞차'로 판단하는 방식이다.

[입력]
- /scan (sensor_msgs/LaserScan)

[출력]
- /front_vehicle/detected   (std_msgs/Bool)    : 앞차 감지 여부
- /front_vehicle/distance   (std_msgs/Float32) : 앞차까지 거리(m), 없으면 inf
- /front_vehicle/angle      (std_msgs/Float32) : 앞차 방향(rad), 없으면 0
- /front_vehicle/width      (std_msgs/Float32) : 앞차로 잡힌 클러스터 폭(m), 없으면 0
- /front_vehicle/debug      (std_msgs/String)  : 튜닝/디버그 문자열

[왜 차선(2차선/3차선) 상관없나?]
- 이 노드는 "앞차가 있냐/어디 있냐"만 알려준다.
- 차선 변경/회피는 다른 노드(Planning)가 distance/angle을 받아서 결정하면 된다.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String


@dataclass
class Cluster:
    """클러스터(물체) 대표값"""
    cx: float       # 중심 x (전방 +)
    cy: float       # 중심 y (좌측 +)
    dist_min: float # 클러스터 내 최소거리(가장 가까운 점)
    width: float    # 클러스터 폭(끝점간 거리 근사)
    n: int          # 점 개수


class FrontVehicleDetector(Node):
    def __init__(self):
        super().__init__("front_vehicle_detector")

        # -------------------- 파라미터(대회 현장 튜닝 포인트) --------------------
        self.scan_topic = self.declare_parameter("scan_topic", "/scan").value

        # 전방 ROI(정면 기준): +/- 몇 도까지만 "앞차 후보"로 볼지
        # - 너무 넓으면 옆 라인/잡음이 섞일 수 있음
        self.front_roi_deg = float(self.declare_parameter("front_roi_deg", 90.0).value)

        # 유효 거리 범위(노이즈 제거)
        self.range_min = float(self.declare_parameter("range_min", 0.25).value)
        self.range_max = float(self.declare_parameter("range_max", 8.0).value)

        # 클러스터링(연속 점 묶기) 기준
        # - 인접 점 간 실제 거리(gap)가 이 값보다 작으면 같은 물체로 묶음
        # - 차량이 작으면 값을 너무 크게 두면 서로 다른 물체가 합쳐질 수 있음
        self.gap_threshold = float(self.declare_parameter("gap_threshold", 0.18).value)

        # 클러스터로 인정할 최소 점 개수
        self.min_cluster_points = int(self.declare_parameter("min_cluster_points", 6).value)

        # 차량 "크기처럼 보이는" 클러스터 폭 필터
        # - F1TENTH 1/10 차량 기준으로 대략 0.15~0.35m 근처가 자주 나옴
        # - 트랙/라이다 위치에 따라 달라서 현장 튜닝 필요
        self.min_width = float(self.declare_parameter("min_width", 0.10).value)
        self.max_width = float(self.declare_parameter("max_width", 0.60).value)

        # 최종 감지 조건: "앞차"로 확정하려면 이 거리 이내여야 한다
        # - 너무 멀면 오탐 가능성이 올라가서 제한을 거는 게 안정적
        self.detect_max_distance = float(self.declare_parameter("detect_max_distance", 5.0).value)

        # 감지 흔들림 방지(히스테리시스)
        # - detected가 True가 된 후, 이 거리보다 멀어질 때만 False로 내려가게 해서 깜빡임 감소
        self.release_distance = float(self.declare_parameter("release_distance", 5.5).value)
        # ----------------------------------------------------------------------

        # -------------------- 퍼블리셔 --------------------
        self.pub_detected = self.create_publisher(Bool, "/front_vehicle/detected", 10)
        self.pub_distance = self.create_publisher(Float32, "/front_vehicle/distance", 10)
        self.pub_angle = self.create_publisher(Float32, "/front_vehicle/angle", 10)
        self.pub_width = self.create_publisher(Float32, "/front_vehicle/width", 10)
        self.pub_debug = self.create_publisher(String, "/front_vehicle/debug", 10)

        # -------------------- 구독 --------------------
        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)

        # 내부 상태(깜빡임 줄이기용)
        self.is_detected = False
        self.last_dist = float("inf")

        self.get_logger().info(
            f"[front_vehicle_detector] start. scan={self.scan_topic} publish=/front_vehicle/*"
        )

    # -------------------- 유틸 함수 --------------------
    @staticmethod
    def deg2rad(deg: float) -> float:
        return deg * math.pi / 180.0

    def extract_front_points(self, scan: LaserScan) -> List[Tuple[float, float, float]]:
        """
        전방 ROI 내의 (x,y,r) 점들을 추출한다.
        - x: 전방(+), y: 좌측(+)
        """
        roi = self.deg2rad(self.front_roi_deg)
        a0 = max(-roi, scan.angle_min)
        a1 = min(+roi, scan.angle_max)
        if a1 <= a0:
            return []

        i0 = int((a0 - scan.angle_min) / scan.angle_increment)
        i1 = int((a1 - scan.angle_min) / scan.angle_increment)

        pts = []
        for i in range(max(0, i0), min(len(scan.ranges) - 1, i1) + 1):
            r = scan.ranges[i]
            if not math.isfinite(r):
                continue
            if r < self.range_min or r > self.range_max:
                continue

            ang = scan.angle_min + i * scan.angle_increment
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            pts.append((x, y, r))
        return pts

    def cluster_points(self, pts: List[Tuple[float, float, float]]) -> List[Cluster]:
        """
        연속 점 기반의 단순 클러스터링.
        - 전방 ROI에서 스캔 인덱스가 연속이면 같은 물체일 가능성이 높다.
        - 인접 점 간 gap(유클리드 거리)이 threshold 이하이면 같은 클러스터로 묶는다.
        """
        if not pts:
            return []

        raw_clusters: List[List[Tuple[float, float, float]]] = []
        cur = [pts[0]]

        for p in pts[1:]:
            x0, y0, _ = cur[-1]
            x1, y1, _ = p
            gap = math.hypot(x1 - x0, y1 - y0)

            if gap <= self.gap_threshold:
                cur.append(p)
            else:
                raw_clusters.append(cur)
                cur = [p]
        raw_clusters.append(cur)

        clusters: List[Cluster] = []
        for c in raw_clusters:
            # 너무 작은 클러스터는 노이즈로 버림
            if len(c) < self.min_cluster_points:
                continue

            xs = [p[0] for p in c]
            ys = [p[1] for p in c]
            rs = [p[2] for p in c]

            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            dist_min = min(rs)

            # 폭(width) 추정: 클러스터의 양 끝점 거리(간단 근사)
            xL, yL, _ = c[0]
            xR, yR, _ = c[-1]
            width = math.hypot(xR - xL, yR - yL)

            # 차량 크기처럼 보이는 폭만 통과
            if width < self.min_width or width > self.max_width:
                continue

            clusters.append(Cluster(cx=cx, cy=cy, dist_min=dist_min, width=width, n=len(c)))

        return clusters

    def select_front_vehicle(self, clusters: List[Cluster]) -> Optional[Cluster]:
        """
        후보 클러스터 중 '앞차'로 쓸 대상을 선택.
        - 기본 전략: 가장 가까운(dist_min이 최소) 클러스터
        """
        if not clusters:
            return None
        clusters.sort(key=lambda c: c.dist_min)
        return clusters[0]

    # -------------------- 메인 콜백 --------------------
    def on_scan(self, scan: LaserScan):
        """
        1) 전방 ROI 점 추출
        2) 클러스터링
        3) 차량 후보 선택
        4) 감지 여부(detected)/거리/각도 publish
        """
        pts = self.extract_front_points(scan)
        clusters = self.cluster_points(pts)
        front = self.select_front_vehicle(clusters)

        # 기본값(미감지 상태)
        detected = False
        dist = float("inf")
        ang = 0.0
        width = 0.0

        if front is not None:
            # 후보가 있으면 거리/각도 계산
            dist = float(front.dist_min)
            ang = float(math.atan2(front.cy, front.cx))   # 라이다 좌표계 기준 방향(rad)
            width = float(front.width)

            # 거리 조건으로 감지 결정
            if dist <= self.detect_max_distance:
                detected = True

        # -------------------- 히스테리시스(깜빡임 방지) --------------------
        # - 한번 감지됐으면, 조금 멀어질 때까지는 계속 감지 상태 유지
        if self.is_detected:
            # 감지 중인데 dist가 release_distance보다 멀어지면 해제
            if (not math.isfinite(dist)) or (dist > self.release_distance):
                self.is_detected = False
        else:
            # 미감지 중인데 detected 조건을 만족하면 감지로 전환
            if detected:
                self.is_detected = True

        # 감지 상태에 따라 출력값 정리
        if not self.is_detected:
            dist_out = float("inf")
            ang_out = 0.0
            width_out = 0.0
        else:
            dist_out = dist
            ang_out = ang
            width_out = width

        # -------------------- publish --------------------
        self.pub_detected.publish(Bool(data=self.is_detected))
        self.pub_distance.publish(Float32(data=float(dist_out)))
        self.pub_angle.publish(Float32(data=float(ang_out)))
        self.pub_width.publish(Float32(data=float(width_out)))

        dbg = (
            f"det={self.is_detected} "
            f"cand={front is not None} "
            f"dist={dist_out:.2f} ang(rad)={ang_out:.2f} width={width_out:.2f} "
            f"clusters={len(clusters)} roi=+/-{self.front_roi_deg}deg"
        )
        self.pub_debug.publish(String(data=dbg))


def main():
    rclpy.init()
    node = FrontVehicleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
