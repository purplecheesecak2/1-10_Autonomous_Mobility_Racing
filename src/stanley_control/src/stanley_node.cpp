#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

// ★ 본인 계정명 확인 (louisdarong)
const std::string CSV_PATH = "/home/louisdarong/ros2_ws/waypoints.csv";

struct Waypoint {
    double x, y;
    double yaw; 
};

class StanleyController : public rclcpp::Node {
public:
    StanleyController() : Node("stanley_node") {
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        vis_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/stanley_marker", 10);
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&StanleyController::odom_callback, this, std::placeholders::_1));

        load_waypoints();
        RCLCPP_INFO(this->get_logger(), "Stanley Controller Started (Smooth Visual Version)");
    }

private:
    // [튜닝 파라미터]
    const double L = 0.33;           // 휠베이스
    const double k = 0.5;            // 제어 게인
    const double k_soft = 1.0;       // 저속 안정화 상수
    
    // 속도 설정
    const double MAX_STEER = 0.4;    
    const double MAX_SPEED = 5.0;    
    const double MIN_SPEED = 1.0;

    std::vector<Waypoint> path_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vis_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    void load_waypoints() {
        std::ifstream file(CSV_PATH);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "CSV 파일 오류! 경로: %s", CSV_PATH.c_str());
            return;
        }

        std::string line;
        std::vector<std::pair<double, double>> raw_points;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            if (row.size() >= 2) {
                raw_points.push_back({std::stod(row[0]), std::stod(row[1])});
            }
        }

        for (size_t i = 0; i < raw_points.size(); ++i) {
            double x = raw_points[i].first;
            double y = raw_points[i].second;
            
            size_t next_idx = (i + 1) % raw_points.size();
            double dx = raw_points[next_idx].first - x;
            double dy = raw_points[next_idx].second - y;
            double yaw = std::atan2(dy, dx);

            path_.push_back({x, y, yaw});
        }
        publish_path();
    }

    // 초록색 선 그리기 (매 프레임 갱신 + 시간 고정)
    void publish_path() {
        if (path_.empty()) return;

        auto marker = visualization_msgs::msg::Marker();
        marker.header.frame_id = "map";
        
        // ★ 핵심: 시간을 0으로 설정하여 어떤 시점이든 TF 변환이 즉시 되도록 함
        marker.header.stamp = rclcpp::Time(0); 

        marker.ns = "stanley_path"; 
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // 깜빡임 방지를 위해 수명(Lifetime)을 0(무한대)으로 설정
        marker.lifetime = rclcpp::Duration::from_seconds(0);

        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.1; 
        marker.color.a = 1.0; 
        marker.color.g = 1.0; // 초록색
        marker.color.b = 0.0;
        marker.color.r = 0.0;

        for (const auto& wp : path_) {
            geometry_msgs::msg::Point p;
            p.x = wp.x;
            p.y = wp.y;
            p.z = 0.0; 
            marker.points.push_back(p);
        }
        vis_pub_->publish(marker);
    }

    int get_closest_waypoint_index(double x, double y) {
        double min_dist = std::numeric_limits<double>::max();
        int idx = 0;
        for (size_t i = 0; i < path_.size(); ++i) {
            double dist = std::sqrt(std::pow(path_[i].x - x, 2) + std::pow(path_[i].y - y, 2));
            if (dist < min_dist) {
                min_dist = dist;
                idx = i;
            }
        }
        return idx;
    }

    void publish_debug_markers(double fx, double fy, double tx, double ty) {
        // 앞바퀴 (빨강)
        auto m1 = visualization_msgs::msg::Marker();
        m1.header.frame_id = "map"; 
        m1.header.stamp = rclcpp::Time(0); 
        m1.ns = "front_axle"; m1.id = 1;
        m1.type = visualization_msgs::msg::Marker::SPHERE;
        m1.action = visualization_msgs::msg::Marker::ADD;
        m1.pose.position.x = fx; m1.pose.position.y = fy; m1.pose.position.z = 0.2;
        m1.scale.x = 0.2; m1.scale.y = 0.2; m1.scale.z = 0.2;
        m1.color.a = 1.0; m1.color.r = 1.0; m1.color.g = 0.0; m1.color.b = 0.0;
        vis_pub_->publish(m1);

        // 목표점 (노랑)
        auto m2 = visualization_msgs::msg::Marker();
        m2.header.frame_id = "map"; 
        m2.header.stamp = rclcpp::Time(0); 
        m2.ns = "target_point"; m2.id = 2;
        m2.type = visualization_msgs::msg::Marker::SPHERE;
        m2.action = visualization_msgs::msg::Marker::ADD;
        m2.pose.position.x = tx; m2.pose.position.y = ty; m2.pose.position.z = 0.2;
        m2.scale.x = 0.3; m2.scale.y = 0.3; m2.scale.z = 0.3;
        m2.color.a = 1.0; m2.color.r = 1.0; m2.color.g = 1.0; m2.color.b = 0.0;
        vis_pub_->publish(m2);
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (path_.empty()) return;

        double rx = msg->pose.pose.position.x;
        double ry = msg->pose.pose.position.y;
        double v = msg->twist.twist.linear.x; 

        double siny_cosp = 2 * (msg->pose.pose.orientation.w * msg->pose.pose.orientation.z + 
                               msg->pose.pose.orientation.x * msg->pose.pose.orientation.y);
        double cosy_cosp = 1 - 2 * (msg->pose.pose.orientation.y * msg->pose.pose.orientation.y + 
                                   msg->pose.pose.orientation.z * msg->pose.pose.orientation.z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        double fx = rx + L * std::cos(yaw);
        double fy = ry + L * std::sin(yaw);

        int idx = get_closest_waypoint_index(fx, fy);
        Waypoint target = path_[idx];

        publish_debug_markers(fx, fy, target.x, target.y);

        double map_vec_x = std::cos(target.yaw);
        double map_vec_y = std::sin(target.yaw);
        double car_vec_x = fx - target.x;
        double car_vec_y = fy - target.y;

        double cross_prod = map_vec_x * car_vec_y - map_vec_y * car_vec_x;
        double error_magnitude = std::sqrt(car_vec_x*car_vec_x + car_vec_y*car_vec_y);

        double error_cross_track;
        if (cross_prod > 0) {
            error_cross_track = -error_magnitude; 
        } else {
            error_cross_track = error_magnitude;  
        }

        double error_heading = normalize_angle(target.yaw - yaw);
        double steering_angle = error_heading + std::atan2(k * error_cross_track, k_soft + v);

        steering_angle = std::max(-MAX_STEER, std::min(MAX_STEER, steering_angle));

        double steering_ratio = std::abs(steering_angle) / MAX_STEER;
        double target_speed = MAX_SPEED * (1.0 - steering_ratio) + MIN_SPEED * steering_ratio;
        target_speed = std::max(MIN_SPEED, target_speed);

        publish_drive(steering_angle, target_speed);
        
        // ★ [복구] 다시 매번 그립니다. (Base_link 시점에서도 부드럽게 보이기 위함)
        publish_path();
    }

    void publish_drive(double steer, double speed) {
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.header.stamp = this->now();
        drive_msg.drive.steering_angle = steer;
        drive_msg.drive.speed = speed;
        drive_pub_->publish(drive_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StanleyController>());
    rclcpp::shutdown();
    return 0;
}
