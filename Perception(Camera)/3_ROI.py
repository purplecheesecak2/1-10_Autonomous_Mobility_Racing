#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class WebcamWithROI(Node):
    def __init__(self):
        # ROS2 노드 초기화
        super().__init__('webcam_with_roi_node')
        
        # CvBridge 초기화
        self.bridge = CvBridge()
        
        # V4L2 백엔드로 웹캠 열기
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다!")
            return
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 이미지 크기
        self.img_width = 640
        self.img_height = 480
        
        # ⭐ ROI 설정 - 하단 절반만 사용
        self.roi_start_height = self.img_height // 2  # 240 (화면 절반 지점)
        
        # Noise Reduction 파라미터
        self.gaussian_kernel_size = 5
        
        # Publisher 설정
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.gray_pub = self.create_publisher(Image, '/camera/gray/image_raw', 1)
        self.denoised_pub = self.create_publisher(Image, '/camera/denoised/image_raw', 1)
        self.roi_pub = self.create_publisher(Image, '/camera/roi/image_raw', 1)  # ⭐ ROI 추가
        
        # 타이머 설정 (30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Webcam with ROI 노드가 시작되었습니다.")
        self.get_logger().info("=" * 70)
        self.get_logger().info("처리 순서:")
        self.get_logger().info("  1️⃣  원본 컬러")
        self.get_logger().info("  2️⃣  Grayscale 변환")
        self.get_logger().info("  3️⃣  Noise Reduction (Gaussian Blur)")
        self.get_logger().info("  4️⃣  ROI 설정 (하단 절반만 추출)")
        self.get_logger().info("=" * 70)
        self.get_logger().info("📹 토픽:")
        self.get_logger().info("  /camera/color/image_raw    - 원본 컬러")
        self.get_logger().info("  /camera/gray/image_raw     - Grayscale")
        self.get_logger().info("  /camera/denoised/image_raw - Noise Reduction")
        self.get_logger().info("  /camera/roi/image_raw      - ROI (하단 절반)")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"🔧 설정:")
        self.get_logger().info(f"  해상도: {self.img_width}x{self.img_height}")
        self.get_logger().info(f"  Gaussian Kernel: {self.gaussian_kernel_size}x{self.gaussian_kernel_size}")
        self.get_logger().info(f"  ROI 시작 높이: {self.roi_start_height} (화면 절반)")
        self.get_logger().info(f"  ROI 크기: {self.img_width}x{self.img_height - self.roi_start_height}")
        self.get_logger().info("=" * 70)
    
    def apply_roi(self, image):
        """
        하단 절반만 추출하는 ROI 적용
        
        Args:
            image: 입력 이미지
            
        Returns:
            roi_image: ROI가 적용된 이미지 (하단 절반만)
        """
        # 하단 절반만 크롭
        roi_image = image[self.roi_start_height:self.img_height, 0:self.img_width]
        return roi_image
    
    def timer_callback(self):
        """타이머 콜백"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning("프레임을 읽을 수 없습니다.", throttle_duration_sec=5.0)
            return
        
        try:
            # 1️⃣ 원본 컬러 이미지 publish
            color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            color_msg.header.stamp = self.get_clock().now().to_msg()
            color_msg.header.frame_id = "camera_frame"
            self.color_pub.publish(color_msg)
            
            # 2️⃣ Grayscale 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
            gray_msg.header.stamp = self.get_clock().now().to_msg()
            gray_msg.header.frame_id = "camera_frame"
            self.gray_pub.publish(gray_msg)
            
            # 3️⃣ Noise Reduction (Gaussian Blur)
            denoised_frame = cv2.GaussianBlur(
                gray_frame,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                0
            )
            denoised_msg = self.bridge.cv2_to_imgmsg(denoised_frame, encoding="mono8")
            denoised_msg.header.stamp = self.get_clock().now().to_msg()
            denoised_msg.header.frame_id = "camera_frame"
            self.denoised_pub.publish(denoised_msg)
            
            # 4️⃣ ROI 적용 (하단 절반만 추출)
            roi_frame = self.apply_roi(denoised_frame)
            roi_msg = self.bridge.cv2_to_imgmsg(roi_frame, encoding="mono8")
            roi_msg.header.stamp = self.get_clock().now().to_msg()
            roi_msg.header.frame_id = "camera_frame"
            self.roi_pub.publish(roi_msg)
            
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
        node = WebcamWithROI()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()