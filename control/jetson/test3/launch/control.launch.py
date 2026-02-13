from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('test3')
    config_file = os.path.join(pkg_share, 'config', 'params.yaml')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/ttyUSB0',
            description='Serial port for ESP32 communication'
        ),

        DeclareLaunchArgument(
            'baud_rate',
            default_value='115200',
            description='Serial baud rate'
        ),

        # Control node
        Node(
            package='test3',
            executable='jetson_esp32_control',
            name='jetson_esp32_control',
            output='screen',
            parameters=[
                config_file,
                {
                    'serial_port': LaunchConfiguration('serial_port'),
                    'baud_rate': LaunchConfiguration('baud_rate'),
                }
            ]
        ),
    ])
