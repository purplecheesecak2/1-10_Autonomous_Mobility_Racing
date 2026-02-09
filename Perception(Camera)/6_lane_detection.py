#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LaneDetectionSlidingWindow(Node):
    def __init__(self):
        # ROS2 노드 초기화
        super().__init__('lane_detection_sliding_window_node')
        
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
        self.roi_start_height = self.img_height // 2
        self.roi_height = self.img_height - self.roi_start_height
        
        # Noise Reduction 파라미터
        self.gaussian_kernel_size = 5
        
        # Canny Edge 파라미터
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # BEV 원근 변환 포인트 초기값
        self.src_lt_x = 280  # 좌상 X
        self.src_rt_x = 360  # 우상 X
        
        self.src_points = np.float32([
            [self.src_lt_x, 0],
            [self.src_rt_x, 0],
            [640, self.roi_height],
            [0, self.roi_height]
        ])
        
        self.dst_points = np.float32([
            [150, 0],
            [490, 0],
            [490, self.roi_height],
            [150, self.roi_height]
        ])
        
        self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # ⭐ Sliding Window 파라미터
        self.nwindows = 9           # 윈도우 개수
        self.margin = 50            # 윈도우 너비의 절반
        self.minpix = 50            # 윈도우 안에 최소 픽셀 수
        
        # ⭐ OpenCV 창 생성 (3개)
        cv2.namedWindow('1. ROI + Canny Edge')
        cv2.namedWindow('2. Birds-Eye View')
        cv2.namedWindow('3. Lane Detection')
        
        # ⭐ 트랙바 생성 (각 창에 관련된 트랙바 배치)
        # 1번 창: Canny Edge 관련
        cv2.createTrackbar('Canny T1', '1. ROI + Canny Edge', self.canny_threshold1, 255, self.on_trackbar)
        cv2.createTrackbar('Canny T2', '1. ROI + Canny Edge', self.canny_threshold2, 255, self.on_trackbar)
        
        # 2번 창: BEV 관련
        cv2.createTrackbar('BEV Left', '2. Birds-Eye View', self.src_lt_x, 640, self.on_trackbar)
        cv2.createTrackbar('BEV Right', '2. Birds-Eye View', self.src_rt_x, 640, self.on_trackbar)
        
        # 3번 창: Sliding Window 관련
        cv2.createTrackbar('Windows', '3. Lane Detection', self.nwindows, 15, self.on_trackbar)
        cv2.createTrackbar('Margin', '3. Lane Detection', self.margin, 100, self.on_trackbar)
        
        # Publisher 설정
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 1)
        self.bev_pub = self.create_publisher(Image, '/camera/bev/image_raw', 1)
        self.lane_pub = self.create_publisher(Image, '/camera/lane/image_raw', 1)
        
        # 타이머 설정 (30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Sliding Window Lane Detection 노드가 시작되었습니다.")
        self.get_logger().info("=" * 70)
        self.get_logger().info("📺 OpenCV 창 (3개):")
        self.get_logger().info("  1. ROI + Canny Edge - Canny 파라미터 조정")
        self.get_logger().info("  2. Birds-Eye View - BEV 포인트 조정")
        self.get_logger().info("  3. Lane Detection - Sliding Window 결과")
        self.get_logger().info("=" * 70)
        self.get_logger().info("🎛️  트랙바:")
        self.get_logger().info("  [1번 창] Canny T1, T2: Canny Edge 임계값")
        self.get_logger().info("  [2번 창] BEV Left/Right: BEV 원근 변환 포인트")
        self.get_logger().info("  [3번 창] Windows, Margin: Sliding Window 파라미터")
        self.get_logger().info("=" * 70)
        self.get_logger().info("⌨️  's' - 설정 출력 | 'q' - 종료")
        self.get_logger().info("=" * 70)
    
    def on_trackbar(self, val):
        """트랙바 콜백"""
        pass
    
    def find_lane_pixels(self, binary_warped):
        """
        Sliding Window로 차선 픽셀 찾기
        
        Args:
            binary_warped: BEV 변환된 이진 이미지
            
        Returns:
            leftx, lefty, rightx, righty: 왼쪽/오른쪽 차선 픽셀 좌표
            out_img: 시각화 이미지
        """
        # 히스토그램 생성 (이미지 하단 절반)
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # 시각화용 컬러 이미지
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # 히스토그램 피크로 차선 시작점 찾기
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # 윈도우 높이
        window_height = int(binary_warped.shape[0] // self.nwindows)
        
        # 0이 아닌 픽셀의 x, y 좌표
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 현재 윈도우 중심
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # 차선 픽셀 인덱스 저장
        left_lane_inds = []
        right_lane_inds = []
        
        # 각 윈도우 순회
        for window in range(self.nwindows):
            # 윈도우 경계 계산
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            # 왼쪽 윈도우 경계
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            
            # 오른쪽 윈도우 경계
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # 윈도우 그리기 (시각화)
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                         (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                         (win_xright_high, win_y_high), (0, 255, 0), 2)
            
            # 윈도우 내 0이 아닌 픽셀 찾기
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # 인덱스 추가
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # 충분한 픽셀이 있으면 윈도우 중심 재조정
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # 인덱스 배열 연결
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # 차선을 찾지 못한 경우
            pass
        
        # 차선 픽셀 좌표 추출
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty, out_img
    
    def fit_polynomial(self, binary_warped, leftx, lefty, rightx, righty):
        """
        Polynomial Fit으로 차선 곡선 생성
        
        Args:
            binary_warped: BEV 이미지
            leftx, lefty, rightx, righty: 차선 픽셀 좌표
            
        Returns:
            result: 차선이 그려진 이미지
            left_fitx, right_fitx: 차선 x 좌표
            ploty: y 좌표
        """
        # 2차 다항식 피팅
        left_fit = np.polyfit(lefty, leftx, 3) if len(lefty) > 0 else None
        right_fit = np.polyfit(righty, rightx, 3) if len(righty) > 0 else None
        
        # y 좌표 생성
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        
        # 차선 x 좌표 계산
        if left_fit is not None:
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        else:
            left_fitx = np.zeros_like(ploty)
        
        if right_fit is not None:
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        else:
            right_fitx = np.zeros_like(ploty)
        
        return left_fitx, right_fitx, ploty
    
    def timer_callback(self):
        """타이머 콜백"""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning("프레임을 읽을 수 없습니다.", throttle_duration_sec=5.0)
            return
        
        try:
            # 트랙바 값 읽기
            self.canny_threshold1 = cv2.getTrackbarPos('Canny T1', '1. ROI + Canny Edge')
            self.canny_threshold2 = cv2.getTrackbarPos('Canny T2', '1. ROI + Canny Edge')
            self.src_lt_x = cv2.getTrackbarPos('BEV Left', '2. Birds-Eye View')
            self.src_rt_x = cv2.getTrackbarPos('BEV Right', '2. Birds-Eye View')
            self.nwindows = max(1, cv2.getTrackbarPos('Windows', '3. Lane Detection'))
            self.margin = max(10, cv2.getTrackbarPos('Margin', '3. Lane Detection'))
            
            if self.canny_threshold1 > self.canny_threshold2:
                self.canny_threshold2 = self.canny_threshold1 + 1
                cv2.setTrackbarPos('Canny T2', '1. ROI + Canny Edge', self.canny_threshold2)
            
            # BEV 포인트 유효성 검사
            if self.src_lt_x >= self.src_rt_x:
                self.src_rt_x = self.src_lt_x + 10
                cv2.setTrackbarPos('BEV Right', '2. Birds-Eye View', self.src_rt_x)
            
            # ⭐ BEV 원근 변환 행렬 재계산
            self.src_points = np.float32([
                [self.src_lt_x, 0],
                [self.src_rt_x, 0],
                [640, self.roi_height],
                [0, self.roi_height]
            ])
            self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            
            # 1. Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Noise Reduction
            blur = cv2.GaussianBlur(gray, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
            
            # 3. ROI
            roi = blur[self.roi_start_height:self.img_height, 0:self.img_width]
            
            # 4. Canny Edge
            edges = cv2.Canny(roi, self.canny_threshold1, self.canny_threshold2)
            
            # 5. BEV
            bev = cv2.warpPerspective(edges, self.perspective_matrix,
                                     (self.img_width, self.roi_height))
            
            # 6. Sliding Window로 차선 픽셀 찾기
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(bev)
            
            # 7. Polynomial Fit
            left_fitx, right_fitx, ploty = self.fit_polynomial(bev, leftx, lefty, rightx, righty)
            
            # 시각화
            result = out_img.copy()
            
            # 차선 영역 칠하기
            if len(left_fitx) > 0 and len(right_fitx) > 0:
                # 왼쪽 차선 픽셀 색칠 (빨강)
                if len(lefty) > 0:
                    result[lefty, leftx] = [255, 0, 0]
                
                # 오른쪽 차선 픽셀 색칠 (파랑)
                if len(righty) > 0:
                    result[righty, rightx] = [0, 0, 255]
                
                # Polynomial Fit 곡선 그리기
                left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                
                # 차선 사이 영역 칠하기 (초록색 반투명)
                pts = np.hstack((left_points, right_points)).astype(np.int32)
                cv2.fillPoly(result, pts, (0, 255, 0))
                
                # 원본 이미지와 블렌딩
                result = cv2.addWeighted(out_img, 0.7, result, 0.3, 0)
                
                # 차선 곡선 그리기 (노란색)
                for i in range(len(ploty) - 1):
                    cv2.line(result,
                            (int(left_fitx[i]), int(ploty[i])),
                            (int(left_fitx[i+1]), int(ploty[i+1])),
                            (0, 255, 255), 3)
                    cv2.line(result,
                            (int(right_fitx[i]), int(ploty[i])),
                            (int(right_fitx[i+1]), int(ploty[i+1])),
                            (0, 255, 255), 3)
            
            # ⭐ OpenCV 창에 표시 (3개 창)
            
            # 1번 창: ROI + Canny Edge
            roi_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(edges_display, f'Canny T1: {self.canny_threshold1}, T2: {self.canny_threshold2}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('1. ROI + Canny Edge', edges_display)
            
            # 2번 창: Birds-Eye View (BEV 포인트 표시)
            bev_display = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
            # ROI 이미지에 BEV 포인트 표시
            roi_with_points = roi_display.copy()
            pts = self.src_points.astype(np.int32)
            cv2.polylines(roi_with_points, [pts], True, (0, 255, 0), 2)
            for i, pt in enumerate(pts):
                cv2.circle(roi_with_points, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(roi_with_points, f'BEV Left: {self.src_lt_x}, Right: {self.src_rt_x}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # BEV와 ROI를 위아래로 합치기
            combined_bev = np.vstack([roi_with_points, bev_display])
            cv2.imshow('2. Birds-Eye View', combined_bev)
            
            # 3번 창: Lane Detection (Sliding Window + Polynomial Fit)
            # 정보 텍스트 추가
            cv2.putText(result, f'Windows: {self.nwindows}, Margin: {self.margin}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # OpenCV 창에 표시
            cv2.imshow('3. Lane Detection', result)
            
            # ROS2 토픽 발행
            color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            color_msg.header.stamp = self.get_clock().now().to_msg()
            self.color_pub.publish(color_msg)
            
            bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding="mono8")
            bev_msg.header.stamp = self.get_clock().now().to_msg()
            self.bev_pub.publish(bev_msg)
            
            lane_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            lane_msg.header.stamp = self.get_clock().now().to_msg()
            self.lane_pub.publish(lane_msg)
            
            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()
            elif key == ord('s'):
                self.get_logger().info("=" * 70)
                self.get_logger().info("현재 설정값:")
                self.get_logger().info("=" * 70)
                self.get_logger().info(f"Canny: T1={self.canny_threshold1}, T2={self.canny_threshold2}")
                self.get_logger().info(f"Sliding Window: nwindows={self.nwindows}, margin={self.margin}")
                self.get_logger().info(f"BEV: Left={self.src_lt_x}, Right={self.src_rt_x}")
                self.get_logger().info("=" * 70)
                self.get_logger().info("코드에 적용:")
                self.get_logger().info(f"self.canny_threshold1 = {self.canny_threshold1}")
                self.get_logger().info(f"self.canny_threshold2 = {self.canny_threshold2}")
                self.get_logger().info(f"self.nwindows = {self.nwindows}")
                self.get_logger().info(f"self.margin = {self.margin}")
                self.get_logger().info(f"self.src_lt_x = {self.src_lt_x}")
                self.get_logger().info(f"self.src_rt_x = {self.src_rt_x}")
                self.get_logger().info("=" * 70)
            
        except Exception as e:
            self.get_logger().error(f"에러: {e}")
    
    def destroy_node(self):
        self.get_logger().info("노드를 종료합니다...")
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LaneDetectionSlidingWindow()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
