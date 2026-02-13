#!/usr/bin/env python3
"""
Jetson ESP32 Control Node (Python)
Receives planning commands and sends to ESP32 via Serial
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import serial
import math
import time


class JetsonESP32Control(Node):
    def __init__(self):
        super().__init__('jetson_esp32_control')

        # ========== DECLARE PARAMETERS ==========
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('max_steering', 30.0)
        self.declare_parameter('min_steering', -30.0)
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('min_speed', 0.0)
        self.declare_parameter('steering_filter_alpha', 0.3)
        self.declare_parameter('high_steering_threshold', 20.0)
        self.declare_parameter('speed_reduction_factor', 0.5)
        self.declare_parameter('timeout_threshold', 0.5)
        self.declare_parameter('planning_steering_topic', 'desired_steering_angle')
        self.declare_parameter('planning_speed_topic', 'target_speed')
        self.declare_parameter('odom_topic', 'odom')

        # ========== GET PARAMETERS ==========
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.max_steering = self.get_parameter('max_steering').value
        self.min_steering = self.get_parameter('min_steering').value
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = self.get_parameter('min_speed').value
        self.steering_filter_alpha = self.get_parameter('steering_filter_alpha').value
        self.high_steering_threshold = self.get_parameter('high_steering_threshold').value
        self.speed_reduction_factor = self.get_parameter('speed_reduction_factor').value
        self.timeout_threshold = self.get_parameter('timeout_threshold').value

        planning_steering_topic = self.get_parameter('planning_steering_topic').value
        planning_speed_topic = self.get_parameter('planning_speed_topic').value
        odom_topic = self.get_parameter('odom_topic').value

        # ========== INITIALIZE SERIAL ==========
        self.serial_conn = None
        if not self.init_serial():
            self.get_logger().error(f'Failed to open serial port: {self.serial_port}')
            raise RuntimeError('Serial initialization failed')

        # ========== STATE VARIABLES ==========
        self.desired_steering = 0.0
        self.target_speed = 0.0
        self.current_speed = 0.0
        self.filtered_steering = 0.0

        self.last_steering_time = None
        self.last_speed_time = None
        self.last_odom_time = None

        self.received_steering = False
        self.received_speed = False
        self.received_odom = False

        # ========== SUBSCRIBERS ==========
        self.steering_sub = self.create_subscription(
            Float32,
            planning_steering_topic,
            self.steering_callback,
            10
        )

        self.speed_sub = self.create_subscription(
            Float32,
            planning_speed_topic,
            self.speed_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        # ========== TIMER ==========
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50Hz

        self.get_logger().info('=== Jetson ESP32 Control Node (Python) Initialized ===')
        self.get_logger().info(f'Serial Port: {self.serial_port} @ {self.baud_rate} baud')
        self.get_logger().info('Waiting for planning commands...')

    def init_serial(self):
        """Initialize serial connection to ESP32"""
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=0.1,
                write_timeout=0.1
            )
            self.get_logger().info(f'Serial port opened: {self.serial_port}')
            return True
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')
            return False

    def steering_callback(self, msg):
        """Callback for steering command"""
        if math.isnan(msg.data) or math.isinf(msg.data):
            self.get_logger().warn('Invalid steering value (NaN/Inf)')
            return

        self.desired_steering = msg.data
        self.last_steering_time = self.get_clock().now()
        self.received_steering = True

    def speed_callback(self, msg):
        """Callback for speed command"""
        if math.isnan(msg.data) or math.isinf(msg.data):
            self.get_logger().warn('Invalid speed value (NaN/Inf)')
            return

        self.target_speed = msg.data
        self.last_speed_time = self.get_clock().now()
        self.received_speed = True

    def odom_callback(self, msg):
        """Callback for odometry"""
        speed = msg.twist.twist.linear.x
        if math.isnan(speed) or math.isinf(speed):
            self.get_logger().warn('Invalid odometry value (NaN/Inf)')
            return

        self.current_speed = speed
        self.last_odom_time = self.get_clock().now()
        self.received_odom = True

    def limit_steering(self, steering):
        """Limit steering to valid range"""
        return max(self.min_steering, min(self.max_steering, steering))

    def limit_speed(self, speed):
        """Limit speed to valid range"""
        return max(self.min_speed, min(self.max_speed, speed))

    def filter_steering(self, raw_steering):
        """Apply low-pass filter to steering"""
        self.filtered_steering = (
            self.steering_filter_alpha * raw_steering +
            (1.0 - self.steering_filter_alpha) * self.filtered_steering
        )
        return self.filtered_steering

    def adjust_speed_for_steering(self, speed, steering):
        """Reduce speed for high steering angles"""
        abs_steering = abs(steering)
        if abs_steering > self.high_steering_threshold:
            return speed * self.speed_reduction_factor
        return speed

    def check_timeout(self):
        """Check if commands have timed out"""
        if not self.received_steering or not self.received_speed:
            return False

        now = self.get_clock().now()
        steering_dt = (now - self.last_steering_time).nanoseconds / 1e9
        speed_dt = (now - self.last_speed_time).nanoseconds / 1e9

        if steering_dt > self.timeout_threshold or speed_dt > self.timeout_threshold:
            self.get_logger().warn(
                'Planning timeout! No commands sent to ESP32.',
                throttle_duration_sec=1.0
            )
            return True

        return False

    def send_to_esp32(self, steering_deg, speed_mps):
        """Send commands to ESP32 via Serial"""
        if self.serial_conn is None or not self.serial_conn.is_open:
            self.get_logger().warn(
                'Serial port not open',
                throttle_duration_sec=1.0
            )
            return

        try:
            # CSV format: "steering,speed\n"
            command = f"{steering_deg:.2f},{speed_mps:.2f}\n"
            self.serial_conn.write(command.encode())
        except serial.SerialException as e:
            self.get_logger().error(
                f'Serial write error: {e}',
                throttle_duration_sec=1.0
            )

    def control_loop(self):
        """Main control loop (50Hz)"""
        # CRITICAL: Only send commands when we have planning data
        # This allows manual controller to take over
        if not self.received_steering or not self.received_speed:
            return  # Don't send anything

        # Check for timeout
        if self.check_timeout():
            return  # Don't send anything on timeout

        # Process steering
        limited_steering = self.limit_steering(self.desired_steering)
        filtered = self.filter_steering(limited_steering)

        # Process speed
        limited_speed = self.limit_speed(self.target_speed)
        adjusted_speed = self.adjust_speed_for_steering(limited_speed, filtered)

        # Send to ESP32
        self.send_to_esp32(filtered, adjusted_speed)

    def destroy_node(self):
        """Cleanup on shutdown"""
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('Serial port closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = JetsonESP32Control()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
