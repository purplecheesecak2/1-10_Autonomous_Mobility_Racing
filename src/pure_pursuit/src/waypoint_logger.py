import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import math
import os

# 파일 저장 경로 (본인 계정명 louisdarong 확인)
file_path = '/home/louisdarong/ros2_ws/waypoints.csv'

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger')
        # 토픽 이름 확인 (/ego_racecar/odom)
        self.subscription = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10)
        self.file = open(file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.last_x = 0.0
        self.last_y = 0.0
        print(f"[{file_path}]에 경로 기록을 시작합니다... (종료: Ctrl+C)")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # 차량이 0.2m 이상 움직였을 때만 점을 찍음 (데이터 중복 방지)
        dist = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
        if dist > 0.2:
            self.writer.writerow([x, y])
            self.last_x = x
            self.last_y = y
            # print(f"기록됨: {x:.2f}, {y:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.file.close()
        node.destroy_node()
        rclpy.shutdown()
        print("파일 저장 완료!")

if __name__ == '__main__':
    main()
