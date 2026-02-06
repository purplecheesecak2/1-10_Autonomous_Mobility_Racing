#include "rclcpp/rclcpp.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

using std::placeholders::_1;

class ReactiveFollowGap : public rclcpp::Node {

public:
    ReactiveFollowGap() : Node("reactive_node")
    {
        drive_publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ReactiveFollowGap::scan_callback, this, _1));
    }

private:
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;

    double prev_angle = 0.0;

    void preprocess_lidar(std::vector<float> &ranges, double angle_increment)
    {
        int n = ranges.size();
        double min_dist = 100.0;
        int min_idx = -1;

        double max_lookahead = 3.0; 

        for (int i = 0; i < n; i++)
        {
            // 시야각을 최대로 넓힘 (왼쪽 구석까지 보기 위함)
            if (i < n / 10 || i > n - (n / 10)) {
                ranges[i] = 0.0;
                continue;
            }

            if (std::isinf(ranges[i])) ranges[i] = max_lookahead;
            else if (std::isnan(ranges[i])) ranges[i] = 0.0;
            
            if (ranges[i] > max_lookahead) ranges[i] = max_lookahead;

            if (ranges[i] > 0.0 && ranges[i] < min_dist) {
                min_dist = ranges[i];
                min_idx = i;
            }
        }

        if (min_idx != -1) 
        {
            // [설정] 버블 최소화 (0.25m)
            // 왼쪽 틈이 아주 좁아 보여도 "갈 수 있다"고 인식시키기 위해 극한으로 줄임
            double bubble_radius = 0.25; 
            double angle_extent = std::atan2(bubble_radius, min_dist);
            int idx_extent = static_cast<int>(angle_extent / angle_increment);

            int start = std::max(0, min_idx - idx_extent);
            int end = std::min(n - 1, min_idx + idx_extent);

            for (int i = start; i <= end; i++) {
                ranges[i] = 0.0;
            }
        }
    }

    // 2. 최대 틈새(Gap) 찾기 - [수정: 점수제 폐지, 크기 우선]
    void find_max_gap(std::vector<float> &ranges, int &start_idx, int &end_idx)
    {
        int current_start = 0;
        int current_len = 0; 
        int max_len = 0;
        int n = ranges.size();

        // 초기화: 못 찾으면 중앙(직진)을 보게 함 (패닉 스핀 방지 유지)
        start_idx = n / 2; 
        end_idx = n / 2;

        for (int i = 0; i < n; i++)
        {
            if (ranges[i] > 0.0) 
            {
                if (current_len == 0) current_start = i;
                current_len++;
            }
            else 
            {
                // [수정] 점수(Score) 계산 로직 삭제 -> 단순 길이(Length) 비교로 복귀
                // 이제 위치 상관없이 무조건 '가장 넓은 틈'을 선택합니다.
                if (current_len > max_len && current_len > 10)
                {
                    max_len = current_len;
                    start_idx = current_start;
                    end_idx = i - 1;
                }
                current_len = 0;
            }
        }
        
        // 마지막 구간 체크
        if (current_len > max_len && current_len > 10)
        {
            start_idx = current_start;
            end_idx = n - 1;
        }
    }
    int find_best_point(std::vector<float> &ranges, int start_idx, int end_idx)
    {
        return (start_idx + end_idx) / 2;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {
        std::vector<float> ranges = scan_msg->ranges;
        int n = ranges.size();

        preprocess_lidar(ranges, scan_msg->angle_increment);

        int start_idx = 0, end_idx = 0;
        find_max_gap(ranges, start_idx, end_idx);

        int target_idx = find_best_point(ranges, start_idx, end_idx);

        double current_angle = scan_msg->angle_min + target_idx * scan_msg->angle_increment;

        double alpha = 0.60; 
        double final_angle = (alpha * current_angle) + ((1.0 - alpha) * prev_angle);
        prev_angle = final_angle;

        double kp = 0.40; 
        double steering_angle = final_angle * kp;

        double max_speed = 6.0; 
        double speed = max_speed - (3.0 * std::abs(final_angle));

        int center_idx = n / 2;
        double front_dist = (ranges[center_idx] + ranges[center_idx-1] + ranges[center_idx+1]) / 3.0;
        
        if (front_dist < 1.0) {
            speed = std::min(speed, 2.0);
        }

        if (speed < 1.0) speed = 1.0;

        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.header.stamp = this->now();
        drive_msg.drive.steering_angle = steering_angle;
        drive_msg.drive.speed = speed;

        drive_publisher_->publish(drive_msg);
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}