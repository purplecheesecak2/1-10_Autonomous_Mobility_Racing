#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

const std::string CSV_PATH = "/home/louisdarong/ros2_ws/waypoints.csv";

// ★ [수정 1] 차선 폭 조정 (0.7m는 너무 넓어서 벽에 닿음 -> 0.6m가 황금비율)
const double LANE_WIDTH = 0.6;

struct Waypoint {
    double x, y;
    double yaw;
};

class PurePursuit : public rclcpp::Node {
public:
    PurePursuit() : Node("pure_pursuit_node") {
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        path_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/lane_markers", 10);
        target_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/target_marker", 10);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&PurePursuit::odom_callback, this, std::placeholders::_1));

        obs_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/obstacle_info", 10, std::bind(&PurePursuit::obstacle_callback, this, std::placeholders::_1));

        load_waypoints();
        RCLCPP_INFO(this->get_logger(), "Pure Pursuit: Lane Keeping & Wall Ignore Mode");
    }

private:
    // [튜닝 파라미터]
    double L_d_base = 0.8;
    double k_v = 0.1;
    const double MAX_SPEED = 2.5;

    // ★ [수정 2] 감지 거리 축소 (2.5m -> 1.3m)
    // 너무 멀리 보면 커브길의 벽을 장애물로 착각해서 멈춥니다.
    // 1.3m 정도면 실제 장애물에는 반응하고, 벽은 무시합니다.
    const double ACC_DIST = 1.3;

    // 상태 변수
    float front_dist_ = 99.0;
    float left_dist_ = 99.0;
    float right_dist_ = 99.0;

    // 현재 차선 (0:Center, 1:Left, -1:Right)
    int current_lane_ = 0;

    // 정지 시간 기록용 타이머
    rclcpp::Time stop_start_time_ = rclcpp::Time(0);

    // 차선 변경 쿨다운 (빠른 진동 방지)
    rclcpp::Time last_lane_change_ = rclcpp::Time(0);
    const double LANE_CHANGE_COOLDOWN = 2.5;

    // 경로 데이터
    std::vector<Waypoint> path_center_;
    std::vector<Waypoint> path_left_;
    std::vector<Waypoint> path_right_;

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr obs_sub_;

    void load_waypoints() {
        std::ifstream file(CSV_PATH);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "CSV 파일 오류! 경로를 확인하세요.");
            return;
        }

        std::string line;
        std::vector<std::pair<double, double>> raw;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            while (std::getline(ss, cell, ',')) row.push_back(cell);
            if (row.size() >= 2) raw.push_back({std::stod(row[0]), std::stod(row[1])});
        }

        for (size_t i = 0; i < raw.size(); ++i) {
            size_t next = (i + 1) % raw.size();
            double dx = raw[next].first - raw[i].first;
            double dy = raw[next].second - raw[i].second;
            double yaw = std::atan2(dy, dx);
            path_center_.push_back({raw[i].first, raw[i].second, yaw});
        }

        for (const auto& wp : path_center_) {
            double lx = wp.x + LANE_WIDTH * -std::sin(wp.yaw);
            double ly = wp.y + LANE_WIDTH * std::cos(wp.yaw);
            path_left_.push_back({lx, ly, wp.yaw});

            double rx = wp.x + (-LANE_WIDTH) * -std::sin(wp.yaw);
            double ry = wp.y + (-LANE_WIDTH) * std::cos(wp.yaw);
            path_right_.push_back({rx, ry, wp.yaw});
        }
        publish_lanes();
    }

    void publish_lanes() {
        visualization_msgs::msg::MarkerArray marker_array;
        auto create_strip = [&](int id, const std::vector<Waypoint>& path, float r, float g, float b, std::string ns) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = rclcpp::Time(0);
            marker.ns = ns; marker.id = id;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.05;
            marker.color.a = 1.0; marker.color.r = r; marker.color.g = g; marker.color.b = b;
            for (const auto& wp : path) {
                geometry_msgs::msg::Point p;
                p.x = wp.x; p.y = wp.y; p.z = 0.0;
                marker.points.push_back(p);
            }
            return marker;
        };
        marker_array.markers.push_back(create_strip(0, path_center_, 0.0, 1.0, 0.0, "center_lane"));
        marker_array.markers.push_back(create_strip(1, path_left_, 0.0, 0.5, 1.0, "left_lane"));
        marker_array.markers.push_back(create_strip(2, path_right_, 1.0, 0.0, 0.0, "right_lane"));
        path_pub_->publish(marker_array);
    }

    void obstacle_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (msg->data.size() < 3) return;
        front_dist_ = msg->data[0];
        left_dist_  = msg->data[1];
        right_dist_ = msg->data[2];
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (path_center_.empty()) return;

        double car_x = msg->pose.pose.position.x;
        double car_y = msg->pose.pose.position.y;

        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        double car_yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
        double current_speed = msg->twist.twist.linear.x;

        // ★ [State Machine] Lane Keeping & Infinite Dodge
        double target_speed = MAX_SPEED;

        // 1. 긴급 제동 (0.35m 이내 - 진짜 충돌 직전)
        if (front_dist_ < 0.35) {
            target_speed = 0.0;
            if (stop_start_time_.nanoseconds() == 0) {
                stop_start_time_ = this->now();
            }
            double elapsed_time = (this->now() - stop_start_time_).seconds();
            if (elapsed_time < 5.0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "🛑 EMERGENCY STOP! (Stopped for %.1f sec)", elapsed_time);
            }
        }

        // 2. 장애물 회피 로직
        else if (front_dist_ < ACC_DIST) {
            stop_start_time_ = rclcpp::Time(0);

            bool can_change = (last_lane_change_.nanoseconds() == 0 ||
                               (this->now() - last_lane_change_).seconds() > LANE_CHANGE_COOLDOWN);

            if (can_change) {
                // [A] 중앙(0) 주행 중 -> 좌/우 회피
                if (current_lane_ == 0) {
                    if (left_dist_ > 1.2) {
                        current_lane_ = 1;
                        last_lane_change_ = this->now();
                        RCLCPP_INFO(this->get_logger(), "Center -> Left 🔵");
                    }
                    else if (right_dist_ > 1.2) {
                        current_lane_ = -1;
                        last_lane_change_ = this->now();
                        RCLCPP_INFO(this->get_logger(), "Center -> Right 🔴");
                    }
                    else {
                        target_speed *= 0.5; // 갇히면 감속
                    }
                }
                // [B] 왼쪽(1) 주행 중 -> 오른쪽(중앙)으로 회피
                else if (current_lane_ == 1) {
                    if (right_dist_ > 1.2) {
                        current_lane_ = 0;
                        last_lane_change_ = this->now();
                        RCLCPP_INFO(this->get_logger(), "Left -> Center 🟢");
                    }
                    else {
                        target_speed *= 0.8;
                    }
                }
                // [C] 오른쪽(-1) 주행 중 -> 왼쪽(중앙)으로 회피
                else if (current_lane_ == -1) {
                    if (left_dist_ > 1.2) {
                        current_lane_ = 0;
                        last_lane_change_ = this->now();
                        RCLCPP_INFO(this->get_logger(), "Right -> Center 🟢");
                    }
                    else {
                        target_speed *= 0.8;
                    }
                }
            }
            else {
                // 쿨다운 중: 현재 차선 유지, 속도 약간 감속
                target_speed *= 0.8;
            }
        }
        else {
            // 3. 장애물이 없을 때 -> 중앙 복귀
            stop_start_time_ = rclcpp::Time(0);
            bool can_change = (last_lane_change_.nanoseconds() == 0 ||
                               (this->now() - last_lane_change_).seconds() > LANE_CHANGE_COOLDOWN);
            if (can_change && current_lane_ != 0) {
                current_lane_ = 0;
                last_lane_change_ = this->now();
                RCLCPP_INFO(this->get_logger(), "Returning to Center 🟢");
            }
        }

        // 차선 선택
        const std::vector<Waypoint>* current_path;
        if (current_lane_ == 1) current_path = &path_left_;
        else if (current_lane_ == -1) current_path = &path_right_;
        else current_path = &path_center_;

        // 조향 로직
        double current_L_d_base = (current_lane_ != 0) ? 0.4 : L_d_base;
        double L_d = current_L_d_base + k_v * current_speed;

        int closest_idx = -1;
        double min_dist = 1e9;
        for(size_t i=0; i<current_path->size(); ++i) {
            double d = std::hypot((*current_path)[i].x - car_x, (*current_path)[i].y - car_y);
            if(d < min_dist) { min_dist = d; closest_idx = i; }
        }

        int target_idx = closest_idx;
        for(size_t i = closest_idx; i < current_path->size() + closest_idx; ++i) {
            int idx = i % current_path->size();
            double d = std::hypot((*current_path)[idx].x - car_x, (*current_path)[idx].y - car_y);
            if (d > L_d) {
                target_idx = idx;
                break;
            }
        }

        Waypoint target = (*current_path)[target_idx];

        double dx = target.x - car_x;
        double dy = target.y - car_y;
        double alpha = std::atan2(dy, dx) - car_yaw;
        double steering_angle = std::atan2(2.0 * 0.33 * std::sin(alpha), L_d);

        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.drive.steering_angle = steering_angle;
        drive_msg.drive.speed = target_speed;
        drive_pub_->publish(drive_msg);

        auto marker = visualization_msgs::msg::Marker();
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = target.x;
        marker.pose.position.y = target.y;
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3;
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0;
        target_pub_->publish(marker);

        publish_lanes();
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}
