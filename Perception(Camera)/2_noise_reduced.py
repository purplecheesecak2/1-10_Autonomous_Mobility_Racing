#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class WebcamGrayscale(Node):
    def __init__(self):
        # ROS2 노드 초기화
        super().__init__('webcam_grayscale_node')
        
        # CvBridge 초기화
        self.bridge = CvBridge()
        
        # ⭐ GStreamer 대신 V4L2 백엔드 사용
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # 웹캠이 제대로 열렸는지 확인
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다!")
            self.get_logger().error("카메라가 연결되어 있는지 확인하세요.")
            self.get_logger().error("ls /dev/video* 명령어로 카메라를 확인하세요.")
            return
        
        # ⭐ 카메라 설정 (V4L2 백엔드에서는 순서가 중요)
        # 먼저 포맷 설정
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        # 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # FPS 설정
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 실제 설정된 값 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # ⭐ Noise Reduction 파라미터
        self.gaussian_kernel_size = 5  # 가우시안 블러 커널 크기 (홀수)
        
        # Publisher 설정 (Queue 크기 줄이기)
        self.image_pub = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.gray_pub = self.create_publisher(Image, '/camera/gray/image_raw', 1)
        self.denoised_pub = self.create_publisher(Image, '/camera/denoised/image_raw', 1)  # ⭐ 추가
        
        # ⭐ 타이머 설정 (30Hz = 0.033초마다 실행)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("웹캠 Grayscale + Noise Reduction 노드가 시작되었습니다.")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📹 컬러 이미지 토픽: /camera/color/image_raw")
        self.get_logger().info("⚫ Grayscale 토픽: /camera/gray/image_raw")
        self.get_logger().info("✨ Denoised 토픽: /camera/denoised/image_raw")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📐 해상도: {actual_width}x{actual_height}")
        self.get_logger().info(f"🎬 FPS: {actual_fps}")
        self.get_logger().info(f"🔧 Gaussian Kernel: {self.gaussian_kernel_size}x{self.gaussian_kernel_size}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("🖼️  이미지 보기:")
        self.get_logger().info("   ros2 run rqt_image_view rqt_image_view /camera/denoised/image_raw")
        self.get_logger().info("=" * 60)
        self.get_logger().info("종료: Ctrl+C")
        self.get_logger().info("=" * 60)
    
    def timer_callback(self):
        """타이머 콜백 - 프레임을 읽고 grayscale로 변환하여 publish"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning("프레임을 읽을 수 없습니다.", throttle_duration_sec=5.0)
            return
        
        try:
            # 1. 원본 컬러 이미지를 ROS2 메시지로 변환 후 publish
            color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            color_msg.header.stamp = self.get_clock().now().to_msg()
            color_msg.header.frame_id = "camera_frame"
            self.image_pub.publish(color_msg)
            
            # 2. Grayscale로 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Grayscale 이미지를 ROS2 메시지로 변환 후 publish
            gray_msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
            gray_msg.header.stamp = self.get_clock().now().to_msg()
            gray_msg.header.frame_id = "camera_frame"
            self.gray_pub.publish(gray_msg)
            
            # 3. ⭐ Noise Reduction (Gaussian Blur 적용)
            denoised_frame = cv2.GaussianBlur(
                gray_frame,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                0  # sigmaX (0이면 커널 크기로부터 자동 계산)
            )
            
            # Denoised 이미지를 ROS2 메시지로 변환 후 publish
            denoised_msg = self.bridge.cv2_to_imgmsg(denoised_frame, encoding="mono8")
            denoised_msg.header.stamp = self.get_clock().now().to_msg()
            denoised_msg.header.frame_id = "camera_frame"
            self.denoised_pub.publish(denoised_msg)
            
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 에러: {e}")
    
    def destroy_node(self):
        """노드 종료 시 리소스 정리"""
        self.get_logger().info("노드를 종료합니다...")
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WebcamGrayscale()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()