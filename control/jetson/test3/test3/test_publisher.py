#!/usr/bin/env python3
"""
Test Publisher Node
Publishes test steering and speed commands for testing the control system
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import math


class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')

        # Publishers
        self.steering_pub = self.create_publisher(Float32, 'desired_steering_angle', 10)
        self.speed_pub = self.create_publisher(Float32, 'target_speed', 10)

        # Timer for publishing commands (10Hz)
        self.timer = self.create_timer(0.1, self.publish_commands)

        # Test state
        self.test_phase = 0
        self.phase_counter = 0
        self.phase_duration = 30  # 3 seconds per phase (30 * 0.1s)

        self.get_logger().info('=== Test Publisher Started ===')
        self.get_logger().info('Publishing test commands...')
        self.print_phase_info()

    def print_phase_info(self):
        """Print current test phase information"""
        phases = [
            "Phase 0: 직진 (속도 0.5 m/s)",
            "Phase 1: 좌회전 (조향 -15도, 속도 0.3 m/s)",
            "Phase 2: 우회전 (조향 +15도, 속도 0.3 m/s)",
            "Phase 3: 급회전 (조향 -25도, 속도 자동 감속)",
            "Phase 4: 정지 (속도 0.0 m/s)"
        ]
        if self.test_phase < len(phases):
            self.get_logger().info(f'\n>>> {phases[self.test_phase]}')

    def publish_commands(self):
        """Publish test commands based on current phase"""
        steering_msg = Float32()
        speed_msg = Float32()

        # Define test patterns
        if self.test_phase == 0:
            # Phase 0: Straight (3 seconds)
            steering_msg.data = 0.0
            speed_msg.data = 0.5

        elif self.test_phase == 1:
            # Phase 1: Left turn (3 seconds)
            steering_msg.data = -15.0
            speed_msg.data = 0.3

        elif self.test_phase == 2:
            # Phase 2: Right turn (3 seconds)
            steering_msg.data = 15.0
            speed_msg.data = 0.3

        elif self.test_phase == 3:
            # Phase 3: Sharp turn (3 seconds) - should trigger speed reduction
            steering_msg.data = -25.0
            speed_msg.data = 0.8  # Will be reduced by control node

        elif self.test_phase == 4:
            # Phase 4: Stop (3 seconds)
            steering_msg.data = 0.0
            speed_msg.data = 0.0

        else:
            # Cycle back to phase 0
            self.test_phase = 0
            self.phase_counter = 0
            self.print_phase_info()
            return

        # Publish
        self.steering_pub.publish(steering_msg)
        self.speed_pub.publish(speed_msg)

        # Progress to next phase
        self.phase_counter += 1
        if self.phase_counter >= self.phase_duration:
            self.phase_counter = 0
            self.test_phase += 1
            self.print_phase_info()


class SimpleTestPublisher(Node):
    """Simple continuous test publisher - just forward motion"""
    def __init__(self):
        super().__init__('simple_test_publisher')

        # Publishers
        self.steering_pub = self.create_publisher(Float32, 'desired_steering_angle', 10)
        self.speed_pub = self.create_publisher(Float32, 'target_speed', 10)

        # Timer (10Hz)
        self.timer = self.create_timer(0.1, self.publish_commands)

        # Parameters
        self.declare_parameter('test_steering', 0.0)
        self.declare_parameter('test_speed', 0.5)

        self.test_steering = self.get_parameter('test_steering').value
        self.test_speed = self.get_parameter('test_speed').value

        self.get_logger().info('=== Simple Test Publisher Started ===')
        self.get_logger().info(f'Steering: {self.test_steering} deg, Speed: {self.test_speed} m/s')

    def publish_commands(self):
        """Publish constant commands"""
        steering_msg = Float32()
        speed_msg = Float32()

        steering_msg.data = self.test_steering
        speed_msg.data = self.test_speed

        self.steering_pub.publish(steering_msg)
        self.speed_pub.publish(speed_msg)


def main(args=None):
    rclpy.init(args=args)

    # Choose which test publisher to use
    import sys
    if '--simple' in sys.argv:
        node = SimpleTestPublisher()
    else:
        node = TestPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
