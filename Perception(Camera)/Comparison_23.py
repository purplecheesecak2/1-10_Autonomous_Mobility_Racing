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
        
        # ROI 설정
        self.roi_start_height = self.img_height // 2  # 240
        
        # Noise Reduction 파라미터
        self.gaussian_kernel_size = 5
        
        # ⭐ OpenCV 윈도우 생성 (원본 크기 그대로 표시)
        cv2.namedWindow('Denoised (Full)', cv2.WINDOW_NORMAL)
        cv2.namedWindow('ROI (Bottom Half)', cv2.WINDOW_NORMAL)
        
        # 창 크기를 이미지 크기에 맞게 고정
        cv2.resizeWindow('Denoised (Full)', 640, 480)
        cv2.resizeWindow('ROI (Bottom Half)', 640, 240)  # ⭐ ROI는 절반 높이
        
        # Publisher 설정
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.gray_pub = self.create_publisher(Image, '/camera/gray/image_raw', 1)
        self.denoised_pub = self.create_publisher(Image, '/camera/denoised/image_raw', 1)
        self.roi_pub = self.create_publisher(Image, '/camera/roi/image_raw', 1)
        
        # 타이머 설정 (30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Webcam with ROI 노드가 시작되었습니다.")
        self.get_logger().info("=" * 70)
        self.get_logger().info("📺 OpenCV 창:")
        self.get_logger().info("  'Denoised (Full)' - 전체 화면 (640x480)")
        self.get_logger().info("  'ROI (Bottom Half)' - 하단 절반만 (640x240)")
        self.get_logger().info("=" * 70)
        self.get_logger().info("💡 두 창의 크기가 다른 게 정상입니다!")
        self.get_logger().info("   ROI 창이 절반 높이로 작게 보입니다.")
        self.get_logger().info("=" * 70)
        self.get_logger().info("⌨️  'q' 키를 누르면 종료")
        self.get_logger().info("=" * 70)
    
    def timer_callback(self):
        """타이머 콜백"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning("프레임을 읽을 수 없습니다.", throttle_duration_sec=5.0)
            return
        
        try:
            # 1. 원본 컬러 이미지 publish
            color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            color_msg.header.stamp = self.get_clock().now().to_msg()
            color_msg.header.frame_id = "camera_frame"
            self.color_pub.publish(color_msg)
            
            # 2. Grayscale 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
            gray_msg.header.stamp = self.get_clock().now().to_msg()
            gray_msg.header.frame_id = "camera_frame"
            self.gray_pub.publish(gray_msg)
            
            # 3. Noise Reduction (Gaussian Blur)
            denoised_frame = cv2.GaussianBlur(
                gray_frame,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                0
            )
            denoised_msg = self.bridge.cv2_to_imgmsg(denoised_frame, encoding="mono8")
            denoised_msg.header.stamp = self.get_clock().now().to_msg()
            denoised_msg.header.frame_id = "camera_frame"
            self.denoised_pub.publish(denoised_msg)
            
            # 4. ROI 적용 (하단 절반만 추출)
            roi_frame = denoised_frame[self.roi_start_height:self.img_height, 0:self.img_width]
            roi_msg = self.bridge.cv2_to_imgmsg(roi_frame, encoding="mono8")
            roi_msg.header.stamp = self.get_clock().now().to_msg()
            roi_msg.header.frame_id = "camera_frame"
            self.roi_pub.publish(roi_msg)
            
            # ⭐ OpenCV 창에 원본 크기 그대로 표시
            cv2.imshow('Denoised (Full)', denoised_frame)  # 640x480
            cv2.imshow('ROI (Bottom Half)', roi_frame)     # 640x240 (절반!)
            
            # 키 입력 확인
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' 키가 눌렸습니다. 종료합니다...")
                rclpy.shutdown()
            
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 에러: {e}")
    
    def destroy_node(self):
        """노드 종료 시 리소스 정리"""
        self.get_logger().info("노드를 종료합니다...")
        self.cap.release()
        cv2.destroyAllWindows()
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
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()