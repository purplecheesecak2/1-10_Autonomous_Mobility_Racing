#include "rclcpp/rclcpp.hpp"
#include <string>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

class WallFollow : public rclcpp::Node {

public:
    WallFollow() : Node("wall_follow_node")
    {
      
        drive_publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        scan_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&WallFollow::scan_callback, this, std::placeholders::_1));
    }

private:

    double servo_offset = 0.0;
    double prev_error = 0.0;
    double error = 0.0;
    double integral = 0.0;
    double kp = 1.5;
    double kd = 1.2;
    double ki = 0.0;


    std::string lidarscan_topic = "/scan";
    std::string drive_topic = "/drive";
   
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscriber_;
    


    double get_range(const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg, double angle)
    {
        if (std::isnan(angle) || std::isinf(angle)) return 0.0;

 
        double angle_min = scan_msg->angle_min;
        double angle_increment = scan_msg->angle_increment;

    
        int index = static_cast<int>((angle - angle_min) / angle_increment);
        
      
        int array_size = static_cast<int>(scan_msg->ranges.size());

       
        if (index >= 0 && index < array_size) {
            float val = scan_msg->ranges[index];
            if (!std::isnan(val) && !std::isinf(val)) {
                return val;
            }
        }
        return 0.0; 
    }


    double get_error(const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg, double desired_dist)
    {
   
        double b_angle = -90.0 * M_PI / 180.0; 
        
     
        double theta = 57.0 * M_PI / 180.0; 
        
    
        double a_angle = b_angle + theta; 

       
        double b = get_range(scan_msg, b_angle);
        double a = get_range(scan_msg, a_angle);

  
        if (a <= 0.0 || b <= 0.0) {
            return prev_error; 
        }

        double alpha = std::atan2(a * std::cos(theta) - b, a * std::sin(theta));
        
        double D_t = b * std::cos(alpha);
        
        double lookahead_dist = 1.0; 
        double D_next = D_t + lookahead_dist * std::sin(alpha);

        return desired_dist - D_next;
    }



void pid_control(double error, double velocity)
    {

        double dt = 0.05; 
     
        double diff_error = error - prev_error;

        integral += error * dt;

        double derivative = diff_error / dt;

        double angle = (kp * error) + (kd * derivative) + (ki * integral);

        prev_error = error;

        if (angle > 0.4) angle = 0.4;
        if (angle < -0.4) angle = -0.4;

        double current_velocity = velocity;
        if (std::abs(angle) > 0.17){ 
            current_velocity = 3.0;   
        }

        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.header.stamp = this->now();
        drive_msg.drive.steering_angle = angle;
        drive_msg.drive.speed = current_velocity;

        drive_publisher_->publish(drive_msg);
    }


void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) 
    {

        double current_error = get_error(scan_msg, 1.0); 
        pid_control(current_error, 5.0); 
    }

};
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WallFollow>());
    rclcpp::shutdown();
    return 0;
}