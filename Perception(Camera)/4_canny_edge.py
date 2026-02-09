#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CannyEdgeDetection(Node):
    def __init__(self):
        # ROS2 노드 초기화
        super().__init__('canny_edge_detection_node')
        
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
        
        # ⭐ Canny Edge 파라미터 (초기값)
        self.canny_threshold1 = 50   # Low threshold
        self.canny_threshold2 = 150  # High threshold
        
        # ⭐ OpenCV 창 생성 (Canny Edge만)
        cv2.namedWindow('Canny Edge Control')
        
        # ⭐ 트랙바 생성 (Canny Edge Control 창에 붙임)
        cv2.createTrackbar('Threshold1', 'Canny Edge Control', self.canny_threshold1, 255, self.on_trackbar)
        cv2.createTrackbar('Threshold2', 'Canny Edge Control', self.canny_threshold2, 255, self.on_trackbar)
        
        # Publisher 설정
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.gray_pub = self.create_publisher(Image, '/camera/gray/image_raw', 1)
        self.denoised_pub = self.create_publisher(Image, '/camera/denoised/image_raw', 1)
        self.roi_pub = self.create_publisher(Image, '/camera/roi/image_raw', 1)
        self.edge_pub = self.create_publisher(Image, '/camera/edge/image_raw', 1)  # ⭐ Edge 추가
        
        # 타이머 설정 (30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Canny Edge Detection 노드가 시작되었습니다.")
        self.get_logger().info("=" * 70)
        self.get_logger().info("처리 순서:")
        self.get_logger().info("  1️⃣  원본 컬러")
        self.get_logger().info("  2️⃣  Grayscale 변환")
        self.get_logger().info("  3️⃣  Noise Reduction (Gaussian Blur)")
        self.get_logger().info("  4️⃣  ROI 설정 (하단 절반)")
        self.get_logger().info("  5️⃣  Canny Edge Detection")
        self.get_logger().info("=" * 70)
        self.get_logger().info("📹 토픽:")
        self.get_logger().info("  /camera/color/image_raw    - 원본")
        self.get_logger().info("  /camera/gray/image_raw     - Grayscale")
        self.get_logger().info("  /camera/denoised/image_raw - Denoised")
        self.get_logger().info("  /camera/roi/image_raw      - ROI")
        self.get_logger().info("  /camera/edge/image_raw     - Canny Edge")
        self.get_logger().info("=" * 70)
        self.get_logger().info("🎛️  트랙바 사용법:")
        self.get_logger().info("  Threshold1 (Low): 낮은 임계값 (약한 엣지)")
        self.get_logger().info("  Threshold2 (High): 높은 임계값 (강한 엣지)")
        self.get_logger().info("  💡 Threshold1 < Threshold2 유지!")
        self.get_logger().info("=" * 70)
        self.get_logger().info("⌨️  키 조작:")
        self.get_logger().info("  's' - 현재 설정값 출력")
        self.get_logger().info("  'q' - 종료")
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
            # 트랙바에서 현재 값 읽기
            self.canny_threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny Edge Control')
            self.canny_threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny Edge Control')
            
            # Threshold1이 Threshold2보다 크면 자동 조정
            if self.canny_threshold1 > self.canny_threshold2:
                self.canny_threshold2 = self.canny_threshold1 + 1
                cv2.setTrackbarPos('Threshold2', 'Canny Edge Control', self.canny_threshold2)
            
            # 1️⃣ 원본 컬러 이미지
            color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            color_msg.header.stamp = self.get_clock().now().to_msg()
            color_msg.header.frame_id = "camera_frame"
            self.color_pub.publish(color_msg)
            
            # 2️⃣ Grayscale 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
            gray_msg.header.stamp = self.get_clock().now().to_msg()
            self.gray_pub.publish(gray_msg)
            
            # 3️⃣ Noise Reduction (Gaussian Blur)
            denoised_frame = cv2.GaussianBlur(
                gray_frame,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                0
            )
            denoised_msg = self.bridge.cv2_to_imgmsg(denoised_frame, encoding="mono8")
            denoised_msg.header.stamp = self.get_clock().now().to_msg()
            self.denoised_pub.publish(denoised_msg)
            
            # 4️⃣ ROI 적용 (하단 절반)
            roi_frame = denoised_frame[self.roi_start_height:self.img_height, 0:self.img_width]
            roi_msg = self.bridge.cv2_to_imgmsg(roi_frame, encoding="mono8")
            roi_msg.header.stamp = self.get_clock().now().to_msg()
            self.roi_pub.publish(roi_msg)
            
            # 5️⃣ Canny Edge Detection
            edge_frame = cv2.Canny(
                roi_frame,
                self.canny_threshold1,  # Low threshold
                self.canny_threshold2   # High threshold
            )
            edge_msg = self.bridge.cv2_to_imgmsg(edge_frame, encoding="mono8")
            edge_msg.header.stamp = self.get_clock().now().to_msg()
            self.edge_pub.publish(edge_msg)
            
            # ⭐ OpenCV 창에 Canny Edge만 표시 (트랙바 포함)
            # 텍스트 없이 깔끔하게 표시
            cv2.imshow('Canny Edge Control', edge_frame)
            
            # 키 입력 확인
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' 키가 눌렸습니다. 종료합니다...")
                rclpy.shutdown()
            elif key == ord('s'):
                self.get_logger().info("=" * 70)
                self.get_logger().info("현재 Canny Edge 설정:")
                self.get_logger().info(f"  Threshold1 (Low): {self.canny_threshold1}")
                self.get_logger().info(f"  Threshold2 (High): {self.canny_threshold2}")
                self.get_logger().info("=" * 70)
                self.get_logger().info("코드에 적용할 값:")
                self.get_logger().info(f"self.canny_threshold1 = {self.canny_threshold1}")
                self.get_logger().info(f"self.canny_threshold2 = {self.canny_threshold2}")
                self.get_logger().info("=" * 70)
            
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
        node = CannyEdgeDetection()
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