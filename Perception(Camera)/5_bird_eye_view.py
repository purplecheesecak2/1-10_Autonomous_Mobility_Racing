#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LaneDetectionBEVInteractive(Node):
    def __init__(self):
        # ROS2 노드 초기화
        super().__init__('lane_detection_bev_interactive_node')
        
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
        self.roi_height = self.img_height - self.roi_start_height  # 240
        
        # Noise Reduction 파라미터
        self.gaussian_kernel_size = 5
        
        # Canny Edge 파라미터
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # ⭐ BEV 포인트 초기값 (ROI 기준)
        self.src_lt_x = 100   # 좌상 X
        self.src_rt_x = 540   # 우상 X
        
        # OpenCV 창 생성
        cv2.namedWindow('ROI with Points')
        cv2.namedWindow('Canny Edge')
        cv2.namedWindow('Birds-Eye View')
        
        # ⭐ 트랙바 생성
        # Canny 트랙바
        cv2.createTrackbar('Canny T1', 'Canny Edge', self.canny_threshold1, 255, self.on_trackbar)
        cv2.createTrackbar('Canny T2', 'Canny Edge', self.canny_threshold2, 255, self.on_trackbar)
        
        # BEV 포인트 트랙바 (ROI with Points 창에)
        cv2.createTrackbar('Left Top X', 'ROI with Points', self.src_lt_x, 640, self.on_trackbar)
        cv2.createTrackbar('Right Top X', 'ROI with Points', self.src_rt_x, 640, self.on_trackbar)
        
        # Publisher 설정
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.gray_pub = self.create_publisher(Image, '/camera/gray/image_raw', 1)
        self.denoised_pub = self.create_publisher(Image, '/camera/denoised/image_raw', 1)
        self.roi_pub = self.create_publisher(Image, '/camera/roi/image_raw', 1)
        self.edge_pub = self.create_publisher(Image, '/camera/edge/image_raw', 1)
        self.bev_pub = self.create_publisher(Image, '/camera/bev/image_raw', 1)
        
        # 타이머 설정 (30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Lane Detection BEV Interactive 노드가 시작되었습니다.")
        self.get_logger().info("=" * 70)
        self.get_logger().info("🎛️  트랙바:")
        self.get_logger().info("  [Canny Edge 창]")
        self.get_logger().info("    - Canny T1, T2: Canny Edge 임계값")
        self.get_logger().info("  [ROI with Points 창]")
        self.get_logger().info("    - Left Top X: 좌상단 포인트 X좌표")
        self.get_logger().info("    - Right Top X: 우상단 포인트 X좌표")
        self.get_logger().info("=" * 70)
        self.get_logger().info("⌨️  키 조작:")
        self.get_logger().info("  's' - 현재 설정값 출력 (코드에 복사)")
        self.get_logger().info("  'q' - 종료")
        self.get_logger().info("=" * 70)
    
    def on_trackbar(self, val):
        """트랙바 콜백"""
        pass
    
    def get_perspective_matrix(self):
        """현재 트랙바 값으로 원근 변환 행렬 계산"""
        # 소스 포인트 (사다리꼴)
        src_points = np.float32([
            [self.src_lt_x, 0],              # 좌상
            [self.src_rt_x, 0],              # 우상
            [self.img_width, self.roi_height], # 우하 (고정)
            [0, self.roi_height]              # 좌하 (고정)
        ])
        
        # 목적지 포인트 (직사각형)
        # 좌우 마진 150px
        dst_points = np.float32([
            [150, 0],
            [self.img_width - 150, 0],
            [self.img_width - 150, self.roi_height],
            [150, self.roi_height]
        ])
        
        # 원근 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return matrix, src_points
    
    def timer_callback(self):
        """타이머 콜백"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning("프레임을 읽을 수 없습니다.", throttle_duration_sec=5.0)
            return
        
        try:
            # 트랙바에서 현재 값 읽기
            self.canny_threshold1 = cv2.getTrackbarPos('Canny T1', 'Canny Edge')
            self.canny_threshold2 = cv2.getTrackbarPos('Canny T2', 'Canny Edge')
            self.src_lt_x = cv2.getTrackbarPos('Left Top X', 'ROI with Points')
            self.src_rt_x = cv2.getTrackbarPos('Right Top X', 'ROI with Points')
            
            # Canny Threshold 자동 조정
            if self.canny_threshold1 > self.canny_threshold2:
                self.canny_threshold2 = self.canny_threshold1 + 1
                cv2.setTrackbarPos('Canny T2', 'Canny Edge', self.canny_threshold2)
            
            # BEV 포인트 유효성 검사 (Left < Right)
            if self.src_lt_x >= self.src_rt_x:
                self.src_rt_x = self.src_lt_x + 10
                cv2.setTrackbarPos('Right Top X', 'ROI with Points', self.src_rt_x)
            
            # 1️⃣ 원본 컬러 이미지
            color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            color_msg.header.stamp = self.get_clock().now().to_msg()
            self.color_pub.publish(color_msg)
            
            # 2️⃣ Grayscale 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding="mono8")
            gray_msg.header.stamp = self.get_clock().now().to_msg()
            self.gray_pub.publish(gray_msg)
            
            # 3️⃣ Noise Reduction
            denoised_frame = cv2.GaussianBlur(
                gray_frame,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                0
            )
            denoised_msg = self.bridge.cv2_to_imgmsg(denoised_frame, encoding="mono8")
            denoised_msg.header.stamp = self.get_clock().now().to_msg()
            self.denoised_pub.publish(denoised_msg)
            
            # 4️⃣ ROI 적용
            roi_frame = denoised_frame[self.roi_start_height:self.img_height, 0:self.img_width]
            roi_msg = self.bridge.cv2_to_imgmsg(roi_frame, encoding="mono8")
            roi_msg.header.stamp = self.get_clock().now().to_msg()
            self.roi_pub.publish(roi_msg)
            
            # 5️⃣ Canny Edge Detection
            edge_frame = cv2.Canny(
                roi_frame,
                self.canny_threshold1,
                self.canny_threshold2
            )
            edge_msg = self.bridge.cv2_to_imgmsg(edge_frame, encoding="mono8")
            edge_msg.header.stamp = self.get_clock().now().to_msg()
            self.edge_pub.publish(edge_msg)
            
            # 원근 변환 행렬 계산
            perspective_matrix, src_points = self.get_perspective_matrix()
            
            # 6️⃣ Birds-Eye View 변환
            bev_frame = cv2.warpPerspective(
                edge_frame,
                perspective_matrix,
                (self.img_width, self.roi_height),
                flags=cv2.INTER_LINEAR
            )
            bev_msg = self.bridge.cv2_to_imgmsg(bev_frame, encoding="mono8")
            bev_msg.header.stamp = self.get_clock().now().to_msg()
            self.bev_pub.publish(bev_msg)
            
            # ⭐ OpenCV 창에 표시
            
            # 1. ROI with Points - ROI 영역에 포인트 표시
            roi_display = cv2.cvtColor(roi_frame, cv2.COLOR_GRAY2BGR)
            pts = src_points.astype(np.int32)
            # 사다리꼴 그리기
            cv2.polylines(roi_display, [pts], True, (0, 255, 0), 2)
            # 각 포인트 표시
            for i, pt in enumerate(pts):
                cv2.circle(roi_display, tuple(pt), 5, (0, 0, 255), -1)
                label = ['LT', 'RT', 'RB', 'LB'][i]
                cv2.putText(roi_display, label, tuple(pt + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(roi_display, f'LT_X: {self.src_lt_x}, RT_X: {self.src_rt_x}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('ROI with Points', roi_display)
            
            # 2. Canny Edge
            edge_display = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(edge_display, f'T1: {self.canny_threshold1}, T2: {self.canny_threshold2}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Canny Edge', edge_display)
            
            # 3. Birds-Eye View
            bev_display = cv2.cvtColor(bev_frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(bev_display, 'Birds-Eye View',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Birds-Eye View', bev_display)
            
            # 키 입력 확인
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' 키가 눌렸습니다. 종료합니다...")
                rclpy.shutdown()
            elif key == ord('s'):
                self.get_logger().info("=" * 70)
                self.get_logger().info("현재 설정값:")
                self.get_logger().info("=" * 70)
                self.get_logger().info("Canny Edge:")
                self.get_logger().info(f"  self.canny_threshold1 = {self.canny_threshold1}")
                self.get_logger().info(f"  self.canny_threshold2 = {self.canny_threshold2}")
                self.get_logger().info("")
                self.get_logger().info("BEV Source Points (ROI 기준):")
                self.get_logger().info(f"  src_points = np.float32([")
                self.get_logger().info(f"      [{self.src_lt_x}, 0],              # 좌상")
                self.get_logger().info(f"      [{self.src_rt_x}, 0],              # 우상")
                self.get_logger().info(f"      [{self.img_width}, {self.roi_height}], # 우하")
                self.get_logger().info(f"      [0, {self.roi_height}]              # 좌하")
                self.get_logger().info(f"  ])")
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
        node = LaneDetectionBEVInteractive()
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