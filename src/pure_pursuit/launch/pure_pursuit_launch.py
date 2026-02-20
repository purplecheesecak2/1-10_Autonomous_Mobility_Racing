import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pure_pursuit',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen',
            parameters=[{
                'lookahead_distance': 1.0,  # 전방 주시 거리 (Ld)
                'max_speed': 2.0,           # 직선 구간 속도
                'min_speed': 0.83           # 곡선 구간 속도 (규정 제11조 반영) 
            }]
        )
    ])
