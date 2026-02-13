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

        DeclareLaunchArgument(
            'simple_mode',
            default_value='false',
            description='Use simple test mode (constant commands)'
        ),

        DeclareLaunchArgument(
            'test_steering',
            default_value='0.0',
            description='Test steering angle (degrees) for simple mode'
        ),

        DeclareLaunchArgument(
            'test_speed',
            default_value='0.5',
            description='Test speed (m/s) for simple mode'
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

        # Test publisher node
        Node(
            package='test3',
            executable='test_publisher',
            name='test_publisher',
            output='screen',
            parameters=[
                {
                    'test_steering': LaunchConfiguration('test_steering'),
                    'test_speed': LaunchConfiguration('test_speed'),
                }
            ],
            arguments=['--simple'] if LaunchConfiguration('simple_mode') == 'true' else []
        ),
    ])
