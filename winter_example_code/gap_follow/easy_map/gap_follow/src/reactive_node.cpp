#include "rclcpp/rclcpp.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

class ReactiveFollowGap : public rclcpp::Node {

public:
    ReactiveFollowGap() : Node("reactive_node")
    {
      
        drive_pub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        scan_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ReactiveFollowGap::lidar_callback, this, std::placeholders::_1));
    }

private:
    std::string lidarscan_topic = "/scan";
    std::string drive_topic = "/drive";

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub;

   
    double prev_steering_angle = 0.0;

    void preprocess_lidar(std::vector<float>& ranges)
    {   
        const float MAX_RANGE = 10.0; 

        for (size_t i = 0; i < ranges.size(); i++) {
            if (std::isnan(ranges[i]) || std::isinf(ranges[i]) || ranges[i] > MAX_RANGE) {
                ranges[i] = MAX_RANGE;
            }
        }
        
       
        std::vector<float> smoothed = ranges;
        int window_size = 5;
        int half = window_size / 2;

        for (size_t i = half; i < (int)ranges.size() - half; i++) {
            float sum = 0.0;
            for (int j = -half; j <= half; j++) {
                sum += ranges[i + j];
            }
            smoothed[i] = sum / window_size;
        }
        ranges = smoothed;
    }

    void find_max_gap(const std::vector<float>& ranges, int start_i, int end_i, int& gap_start, int& gap_end)
    {
        int max_len = 0;        
        int current_start = -1; 
        int current_len = 0;    

        gap_start = start_i;
        gap_end = start_i;

        for (int i = start_i; i <= end_i; i++) {
            if (ranges[i] > 0.1) {
                if (current_len == 0) current_start = i;
                current_len++;
            }
            else {
                if (current_len > max_len) {
                    max_len = current_len;
                    gap_start = current_start;
                    gap_end = i - 1;
                }
                current_len = 0;
            }
        }

        if (current_len > max_len) {
            gap_start = current_start;
            gap_end = end_i;
        }
    }

    void find_best_point(const std::vector<float>& ranges, int gap_start, int gap_end, int& best_i)
    {   
        
        int furthest_i = gap_start;
        float max_dist = -1.0;
        for (int i = gap_start; i <= gap_end; i++) {
            if (ranges[i] > max_dist) {
                max_dist = ranges[i];
                furthest_i = i;
            }
        }
        
        int gap_center_i = (gap_start + gap_end) / 2;

       
        best_i = (int)(furthest_i * 0.2 + gap_center_i * 0.8);
    }

    void lidar_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {   
        std::vector<float> ranges = scan_msg->ranges;
        
        int num_points = (int)ranges.size();
        int start_idx = num_points / 5;
        int end_idx = 4 * num_points / 5;
        
        preprocess_lidar(ranges);

        
        float bubble_radius = 0.25; 
        float threshold = 1.5;      

        std::vector<float> ranges_with_bubbles = ranges; 

        for (int i = start_idx; i <= end_idx; i++) {
            if (ranges[i] < threshold && ranges[i] > 0.0) {
                int b_range = std::atan2(bubble_radius, ranges[i]) / scan_msg->angle_increment;
                for (int j = i - b_range; j <= i + b_range; j++) {
                    if (j >= start_idx && j <= end_idx) {
                        ranges_with_bubbles[j] = 0.0; 
                    }
                }
            }
        }
        ranges = ranges_with_bubbles; 

        int gap_start, gap_end;
        find_max_gap(ranges, start_idx, end_idx, gap_start, gap_end);
        
        int best_i;
        find_best_point(ranges, gap_start, gap_end, best_i);

        
        float raw_angle = scan_msg->angle_min + best_i * scan_msg->angle_increment;
        
        
        double alpha = 0.3; 
        double smooth_angle = (alpha * raw_angle) + ((1.0 - alpha) * prev_steering_angle);
        prev_steering_angle = smooth_angle; 
        
        float final_angle = smooth_angle * 1.2; 

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp = this->now();
        drive_msg.header.frame_id = "laser";
        drive_msg.drive.steering_angle = final_angle;

        
        float abs_angle = std::abs(final_angle);
        if (abs_angle > 20.0 * M_PI / 180.0) {          
            drive_msg.drive.speed = 1.5; 
        } else if (abs_angle > 10.0 * M_PI / 180.0) {   
            drive_msg.drive.speed = 3.5;
        } else {                                        
            drive_msg.drive.speed = 5.5;                
        }
        
        drive_pub->publish(drive_msg);
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}