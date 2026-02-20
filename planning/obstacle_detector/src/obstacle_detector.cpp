#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp> // 데이터 전송용
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

const float LANE_WIDTH = 0.6; 
const float CHECK_DIST = 2.0; 

class ObstacleDetector : public rclcpp::Node {
public:
    ObstacleDetector() : Node("obstacle_detector") {
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ObstacleDetector::scan_callback, this, std::placeholders::_1));

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacle_markers", 10);
        
        // Planning 팀에게 보낼 데이터 퍼블리셔
        info_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/obstacle_info", 10);
        
        RCLCPP_INFO(this->get_logger(), "Obstacle Detector (Publisher Mode) Started.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr info_pub_;

    // 거리 저장 변수 (초기값 99.0 = 장애물 없음)
    float front_min_dist_ = 99.0;
    float left_min_dist_ = 99.0;
    float right_min_dist_ = 99.0;

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // 매번 리셋
        front_min_dist_ = 99.0;
        left_min_dist_ = 99.0;
        right_min_dist_ = 99.0;

        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            float r = msg->ranges[i];
            if (std::isinf(r) || std::isnan(r) || r > CHECK_DIST || r < 0.1) continue;

            float angle = msg->angle_min + i * msg->angle_increment;
            float x = r * std::cos(angle);
            float y = r * std::sin(angle);

            if (x < 0) continue; // 후방 무시

            // 영역별 가장 가까운 거리(Min Distance) 찾기
            // (A) Center
            if (std::abs(y) < (LANE_WIDTH / 2.0)) {
                if (x < front_min_dist_) front_min_dist_ = x;
            }
            // (B) Left
            else if (y > (LANE_WIDTH / 2.0) && y < (LANE_WIDTH * 1.5)) {
                if (x < left_min_dist_) left_min_dist_ = x;
            }
            // (C) Right
            else if (y < -(LANE_WIDTH / 2.0) && y > -(LANE_WIDTH * 1.5)) {
                if (x < right_min_dist_) right_min_dist_ = x;
            }
        }
       

        publish_markers();
        publish_obstacle_info(); // 데이터 전송
    }

    void publish_obstacle_info() {
        std_msgs::msg::Float32MultiArray msg;
        // [0]:전방, [1]:왼쪽, [2]:오른쪽
        msg.data = {front_min_dist_, left_min_dist_, right_min_dist_};
        info_pub_->publish(msg);
    }

    void publish_markers() {
        visualization_msgs::msg::MarkerArray marker_array;
        auto create_box = [&](int id, float y_offset, float dist, std::string ns) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "ego_racecar/base_link"; 
            marker.header.stamp = this->now();
            marker.ns = ns; marker.id = id;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            marker.pose.position.x = CHECK_DIST / 2.0; 
            marker.pose.position.y = y_offset;
            marker.pose.position.z = 0.1;
            marker.scale.x = CHECK_DIST; marker.scale.y = LANE_WIDTH; marker.scale.z = 0.1;

            marker.color.a = 0.5; 
            // 2m 안에 장애물 있으면 빨강, 없으면 초록
            if (dist < CHECK_DIST) {
                marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0;
            } else {
                marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;
            }
            return marker;
        };

        marker_array.markers.push_back(create_box(0, 0.0, front_min_dist_, "center"));
        marker_array.markers.push_back(create_box(1, LANE_WIDTH, left_min_dist_, "left"));
        marker_array.markers.push_back(create_box(2, -LANE_WIDTH, right_min_dist_, "right"));

        marker_pub_->publish(marker_array);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObstacleDetector>());
    rclcpp::shutdown();
    return 0;
}
