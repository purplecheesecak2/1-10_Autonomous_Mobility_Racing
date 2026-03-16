#!/usr/bin/env python3
"""
Front Vehicle Detector (LiDAR only) - ROS2 rclpy

[목표]
- LiDAR(/scan)만으로 전방 ROI 내의 모든 물체 클러스터를 감지한다.
- 크기 필터 없이 보이는 클러스터를 전부 추출하여 publish한다.
- 여러 물체를 동시에 인식하고, 각각의 거리/각도/폭/좌표(x,y)를 배열로 전송한다.

[입력]
- /scan (sensor_msgs/LaserScan)

[출력]
- /front_vehicle/detected        (std_msgs/Bool)              : 1개 이상 물체 감지 여부
- /front_vehicle/count           (std_msgs/Int32)             : 감지된 물체 수
- /front_vehicle/distances       (std_msgs/Float32MultiArray) : 각 물체까지 거리(m) 배열 (가까운 순)
- /front_vehicle/angles          (std_msgs/Float32MultiArray) : 각 물체 방향(rad) 배열
- /front_vehicle/widths          (std_msgs/Float32MultiArray) : 각 물체 폭(m) 배열
- /front_vehicle/xs              (std_msgs/Float32MultiArray) : 각 물체 중심 x좌표(m) 배열 (base_link 기준, 전방 +)
- /front_vehicle/ys              (std_msgs/Float32MultiArray) : 각 물체 중심 y좌표(m) 배열 (좌측 +)
- /front_vehicle/debug           (std_msgs/String)            : 튜닝/디버그 문자열
- /front_vehicle/markers         (visualization_msgs/MarkerArray) : RViz 시각화용

  ※ distances[i], angles[i], widths[i], xs[i], ys[i] 가 i번째 물체의 정보
    배열은 거리 오름차순(가까운 것 먼저)으로 정렬된다.
  ※ RViz ID는 트래킹 ID (물체가 유지되는 동안 고정)
  ※ 터미널 [0],[1],[2]는 거리순 인덱스 (Planning에 넘겨지는 값)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray, Int32, String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Cluster:
    """클러스터(물체) 대표값"""
    cx: float        # 중심 x (전방 +)
    cy: float        # 중심 y (좌측 +)
    dist_min: float  # 클러스터 내 최소거리(가장 가까운 점)
    angle: float     # 중심 방향(rad)
    width: float     # 클러스터 폭(끝점 간 거리 근사)
    n: int           # 점 개수
    track_id: int = -1  # 트래킹 ID (RViz 시각화용)



class FrontVehicleDetector(Node):

    # 라이다 → base_link x축 오프셋 (라이다가 base_link보다 앞에 있으므로 +)
    LIDAR_TO_BASELINK = 0.27  # 27cm

    def __init__(self):
        super().__init__("front_vehicle_detector")

        # -------------------- 파라미터(대회 현장 튜닝 포인트) --------------------
        self.scan_topic = self.declare_parameter("scan_topic", "/scan").value
        self.front_roi_deg = float(self.declare_parameter("front_roi_deg", 90.0).value)
        self.range_min = float(self.declare_parameter("range_min", 0.25).value)
        self.range_max = float(self.declare_parameter("range_max", 1.5).value)
        self.gap_threshold = float(self.declare_parameter("gap_threshold", 0.18).value)
        self.min_cluster_points = int(self.declare_parameter("min_cluster_points", 4).value)
        # ----------------------------------------------------------------------

        # -------------------- 퍼블리셔 --------------------
        self.pub_detected  = self.create_publisher(Bool,              "/front_vehicle/detected",  10)
        self.pub_count     = self.create_publisher(Int32,             "/front_vehicle/count",     10)
        self.pub_distances = self.create_publisher(Float32MultiArray, "/front_vehicle/distances", 10)
        self.pub_angles    = self.create_publisher(Float32MultiArray, "/front_vehicle/angles",    10)
        self.pub_widths    = self.create_publisher(Float32MultiArray, "/front_vehicle/widths",    10)
        self.pub_xs        = self.create_publisher(Float32MultiArray, "/front_vehicle/xs",        10)
        self.pub_ys        = self.create_publisher(Float32MultiArray, "/front_vehicle/ys",        10)
        self.pub_debug     = self.create_publisher(String,            "/front_vehicle/debug",     10)
        self.pub_markers   = self.create_publisher(MarkerArray,       "/front_vehicle/markers",   10)

        # -------------------- 트래킹 변수 --------------------
        # {track_id: (cx, cy)} 이전 프레임 위치
        self.tracked: Dict[int, Tuple[float, float]] = {}
        # 재사용 가능한 ID 풀 (사라진 물체의 ID를 재사용)
        self.free_ids: List[int] = []
        self.next_new_id: int = 0
        # 같은 물체로 판단할 최대 이동 거리(m)
        self.track_max_dist: float = 0.8

        # -------------------- 구독 --------------------
        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self.on_scan, 10
        )

        self.get_logger().info(
            f"[front_vehicle_detector] start. scan={self.scan_topic} "
            f"publish=/front_vehicle/* (multi-object mode, no size filter)"
        )

    # -------------------- 유틸 --------------------
    @staticmethod
    def deg2rad(deg: float) -> float:
        return deg * math.pi / 180.0

    def extract_front_points(self, scan: LaserScan) -> List[Tuple[float, float, float]]:
        roi = self.deg2rad(self.front_roi_deg)
        a0 = max(-roi, scan.angle_min)
        a1 = min(+roi, scan.angle_max)
        if a1 <= a0:
            return []
        i0 = int((a0 - scan.angle_min) / scan.angle_increment)
        i1 = int((a1 - scan.angle_min) / scan.angle_increment)
        pts: List[Tuple[float, float, float]] = []
        for i in range(max(0, i0), min(len(scan.ranges) - 1, i1) + 1):
            r = scan.ranges[i]
            if not math.isfinite(r):
                continue
            if r < self.range_min or r > self.range_max:
                continue
            ang = scan.angle_min + i * scan.angle_increment
            pts.append((r * math.cos(ang), r * math.sin(ang), r))
        return pts

    def cluster_points(self, pts: List[Tuple[float, float, float]]) -> List[Cluster]:
        if not pts:
            return []
        raw_clusters: List[List[Tuple[float, float, float]]] = []
        cur = [pts[0]]
        for p in pts[1:]:
            x0, y0, _ = cur[-1]
            x1, y1, _ = p
            if math.hypot(x1 - x0, y1 - y0) <= self.gap_threshold:
                cur.append(p)
            else:
                raw_clusters.append(cur)
                cur = [p]
        raw_clusters.append(cur)
        clusters: List[Cluster] = []
        for c in raw_clusters:
            if len(c) < self.min_cluster_points:
                continue
            xs = [p[0] for p in c]
            ys = [p[1] for p in c]
            rs = [p[2] for p in c]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            dist_min = min(rs)
            angle = math.atan2(cy, cx)
            xL, yL, _ = c[0]
            xR, yR, _ = c[-1]
            width = math.hypot(xR - xL, yR - yL)
            clusters.append(Cluster(cx=cx, cy=cy, dist_min=dist_min,
                                    angle=angle, width=width, n=len(c)))
        clusters.sort(key=lambda cl: cl.dist_min)
        return clusters

    def _alloc_id(self) -> int:
        """사용 가능한 ID 반환. 재사용 ID가 있으면 먼저 씀."""
        if self.free_ids:
            return self.free_ids.pop(0)
        tid = self.next_new_id
        self.next_new_id += 1
        return tid

    def assign_track_ids(self, clusters: List[Cluster]) -> None:
        """
        이전 프레임 위치와 비교해 같은 물체면 동일 ID 유지.
        사라진 물체의 ID는 free_ids에 넣어 재사용 → ID가 무한히 커지지 않음.
        """
        new_tracked: Dict[int, Tuple[float, float]] = {}
        used_ids: Set[int] = set()

        for cl in clusters:
            best_id: Optional[int] = None
            best_dist = self.track_max_dist

            for tid, (tx, ty) in self.tracked.items():
                if tid in used_ids:
                    continue
                d = math.hypot(cl.cx - tx, cl.cy - ty)
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is None:
                best_id = self._alloc_id()

            cl.track_id = best_id
            new_tracked[best_id] = (cl.cx, cl.cy)
            used_ids.add(best_id)

        # 사라진 물체 ID → free_ids에 반환
        for tid in self.tracked:
            if tid not in used_ids:
                self.free_ids.append(tid)
        self.free_ids.sort()  # 작은 번호부터 재사용

        self.tracked = new_tracked

    def build_markers(self, clusters: List[Cluster]) -> MarkerArray:
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        lifetime_ns = 300_000_000  # 0.3초

        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for cl in clusters:
            tid = cl.track_id

            # ---------- ① 초록 박스 (0.5m 고정) ----------
            box = Marker()
            box.header.frame_id = "laser"
            box.header.stamp = stamp
            box.ns = "obstacle_box"
            box.id = tid
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = cl.cx
            box.pose.position.y = cl.cy
            box.pose.position.z = 0.0
            box.pose.orientation.w = 1.0
            box.scale.x = 0.5
            box.scale.y = 0.5
            box.scale.z = 0.5
            box.color.r = 0.0
            box.color.g = 1.0
            box.color.b = 0.0
            box.color.a = 0.7
            box.lifetime.nanosec = lifetime_ns
            marker_array.markers.append(box)

            # ---------- ② 흰색 ID 텍스트 ----------
            text = Marker()
            text.header.frame_id = "laser"
            text.header.stamp = stamp
            text.ns = "obstacle_text"
            text.id = tid
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cl.cx
            text.pose.position.y = cl.cy
            text.pose.position.z = 0.7
            text.pose.orientation.w = 1.0
            text.scale.z = 0.35
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"ID: {tid}"
            text.lifetime.nanosec = lifetime_ns
            marker_array.markers.append(text)

        return marker_array

    # -------------------- 메인 콜백 --------------------
    def on_scan(self, scan: LaserScan):
        pts = self.extract_front_points(scan)
        clusters = self.cluster_points(pts)

        # 트래킹 ID 할당 (RViz 시각화용)
        self.assign_track_ids(clusters)
        markers = self.build_markers(clusters)
        self.pub_markers.publish(markers)

        detected  = len(clusters) > 0
        distances = [float(cl.dist_min) for cl in clusters]
        angles    = [float(cl.angle)    for cl in clusters]
        widths    = [float(cl.width)    for cl in clusters]
        xs        = [float(cl.cx) + self.LIDAR_TO_BASELINK for cl in clusters]  # base_link 기준
        ys        = [float(cl.cy)       for cl in clusters]

        # -------------------- publish --------------------
        self.pub_detected.publish(Bool(data=detected))

        count_msg = Int32()
        count_msg.data = len(clusters)
        self.pub_count.publish(count_msg)

        dist_msg = Float32MultiArray()
        dist_msg.data = distances
        self.pub_distances.publish(dist_msg)

        ang_msg = Float32MultiArray()
        ang_msg.data = angles
        self.pub_angles.publish(ang_msg)

        width_msg = Float32MultiArray()
        width_msg.data = widths
        self.pub_widths.publish(width_msg)

        xs_msg = Float32MultiArray()
        xs_msg.data = xs
        self.pub_xs.publish(xs_msg)

        ys_msg = Float32MultiArray()
        ys_msg.data = ys
        self.pub_ys.publish(ys_msg)

        # 디버그: 상위 3개까지만 출력 (거리순 인덱스 기준)
        top_str = " | ".join(
            f"[{i}] d={cl.dist_min:.2f}m a={math.degrees(cl.angle):.1f}deg "
            f"w={cl.width:.2f}m x={cl.cx + self.LIDAR_TO_BASELINK:.2f}m y={cl.cy:.2f}m n={cl.n}"
            for i, cl in enumerate(clusters[:3])
        )
        dbg = (
            f"detected={detected} count={len(clusters)} "
            f"roi=+/-{self.front_roi_deg}deg "
            f"gap={self.gap_threshold}m min_pts={self.min_cluster_points} :: "
            f"{top_str if top_str else 'none'}"
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
