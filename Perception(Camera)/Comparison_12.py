#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class WebcamNoiseReductionComparison(Node):
    def __init__(self):
        # ROS2 노드 초기화
        super().__init__('webcam_noise_reduction_comparison_node')
        
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
        
        # Noise Reduction 파라미터
        self.gaussian_kernel_size = 5
        
        # OpenCV 윈도우 생성
        cv2.namedWindow('Comparison')
        
        # 트랙바로 커널 크기 조정 가능하게
        cv2.createTrackbar('Kernel Size', 'Comparison', 
                          self.gaussian_kernel_size, 15, self.on_trackbar)
        
        # Publisher 설정
        self.image_pub = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.gray_pub = self.create_publisher(Image, '/camera/gray/image_raw', 1)
        self.denoised_pub = self.create_publisher(Image, '/camera/denoised/image_raw', 1)
        
        # 타이머 설정 (30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Noise Reduction Comparison 노드가 시작되었습니다.")
        self.get_logger().info("=" * 70)
        self.get_logger().info("📺 OpenCV 창에서 Grayscale vs Denoised 비교 가능")
        self.get_logger().info("🎛️  트랙바로 Kernel Size 조정")
        self.get_logger().info("⌨️  'q' 키를 누르면 종료")
        self.get_logger().info("=" * 70)
        self.get_logger().info("💡 차이점:")
        self.get_logger().info("  왼쪽(Grayscale): 노이즈 있음 (거칠고 픽셀이 울퉁불퉁)")
        self.get_logger().info("  오른쪽(Denoised): 노이즈 제거 (부드럽고 매끄러움)")
        self.get_logger().info("=" * 70)
    
    def on_trackbar(self, val):
        """트랙바 콜백"""
        pass
    
    def timer_callback(self):
        """타이머 콜백"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning("프레임을 읽을 수 없습니다.", throttle_duration_sec=5.0)
            return
        
        try:
            # 트랙바에서 커널 크기 읽기
            kernel_size = cv2.getTrackbarPos('Kernel Size', 'Comparison')
            
            # 커널 크기를 홀수로 보정
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 1:
                kernel_size = 1
            
            self.gaussian_kernel_size = kernel_size
            
            # Grayscale 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Noise Reduction (Gaussian Blur)
            denoised_frame = cv2.GaussianBlur(
                gray_frame,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                0
            )
            
            # ⭐ 비교를 위해 두 이미지를 옆으로 나란히 붙이기
            # 텍스트 추가를 위해 컬러로 변환
            gray_display = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            denoised_display = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
            
            # 텍스트 추가
            cv2.putText(gray_display, 'Grayscale (Original)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(denoised_display, f'Denoised (Kernel: {self.gaussian_kernel_size}x{self.gaussian_kernel_size})', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 두 이미지를 가로로 연결
            comparison = np.hstack([gray_display, denoised_display])
            
            # 가운데 구분선 추가
            h = comparison.shape[0]
            w = comparison.shape[1]
            cv2.line(comparison, (w//2, 0), (w//2, h), (0, 0, 255), 2)
            
            # 비교 이미지 표시
            cv2.imshow('Comparison', comparison)
            
            # 키 입력 확인
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' 키가 눌렸습니다. 종료합니다...")
                rclpy.shutdown()
            
            # ROS2 토픽으로 publish
            color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            color_msg.header.stamp = self.get_clock().now().to_msg()
            color_msg.header.frame_id = "camera_frame"
            self.image_pub.publish(color_msg)
            
            gray_msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
            gray_msg.header.stamp = self.get_clock().now().to_msg()
            gray_msg.header.frame_id = "camera_frame"
            self.gray_pub.publish(gray_msg)
            
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
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WebcamNoiseReductionComparison()
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