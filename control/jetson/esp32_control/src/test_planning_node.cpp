/*
 * Test Planning Node
 * Publishes to /autonomous/drive for testing the full pipeline
 * Parameters:
 *   speed       : target speed in m/s (default 0.5)
 *   steering    : target steering angle in deg (default 0.0)
 *   mode        : "constant" or "sine" (default constant)
 */

#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <cmath>

class TestPlanning : public rclcpp::Node {
public:
    TestPlanning() : Node("test_planning"), t_(0.0) {
        declare_parameter("speed",    0.5);
        declare_parameter("steering", 0.0);
        declare_parameter("mode",     std::string("constant"));

        speed_    = get_parameter("speed").as_double();
        steering_ = get_parameter("steering").as_double();
        mode_     = get_parameter("mode").as_string();

        pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/autonomous/drive", 10);

        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),   // 10Hz
            std::bind(&TestPlanning::timerCb, this));

        RCLCPP_INFO(get_logger(), "Test Planning Node started");
        RCLCPP_INFO(get_logger(), "  mode=%s  speed=%.2f m/s  steering=%.1f deg",
                    mode_.c_str(), speed_, steering_);
    }

private:
    void timerCb() {
        double spd, steer_rad;

        if (mode_ == "sine") {
            // 조향각을 사인파로 변경 (±steering_deg 범위)
            steer_rad = steering_ * M_PI / 180.0 * std::sin(t_);
            spd = speed_;
            t_ += 0.1;
        } else {
            // constant: 파라미터 값 그대로
            spd       = speed_;
            steer_rad = steering_ * M_PI / 180.0;
        }

        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.header.stamp    = now();
        msg.header.frame_id = "base_link";
        msg.drive.speed          = spd;
        msg.drive.steering_angle = steer_rad;

        pub_->publish(msg);
    }

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    double speed_, steering_, t_;
    std::string mode_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestPlanning>());
    rclcpp::shutdown();
    return 0;
}
