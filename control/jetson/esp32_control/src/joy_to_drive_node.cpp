/*
 * Joy to Drive Node
 * Logitech F710 (X mode) → AckermannDriveStamped
 *
 * 조작법:
 *   왼쪽 스틱 좌우 (axis 0) → 조향 (-45° ~ +45°)
 *   오른쪽 스틱 상하 (axis 4) → 속도 (위=전진, 아래=후진)
 *   RB 버튼 (button 5)       → 데드맨 스위치 (누르는 동안만 이동)
 *
 * F710 X모드 축 번호:
 *   axis 0: 왼쪽 스틱 좌우  (좌=-1, 우=+1)
 *   axis 1: 왼쪽 스틱 상하  (위=+1, 아래=-1)
 *   axis 3: 오른쪽 스틱 좌우
 *   axis 4: 오른쪽 스틱 상하 (위=+1, 아래=-1)
 *   axis 2: LT  (안누름=+1, 누름=-1)
 *   axis 5: RT  (안누름=+1, 누름=-1)
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <cmath>

class JoyToDriveNode : public rclcpp::Node {
public:
    JoyToDriveNode() : Node("joy_to_drive") {
        // Parameters
        this->declare_parameter<int>("steering_axis",   0);     // 왼쪽 스틱 좌우
        this->declare_parameter<int>("speed_axis",      4);     // 오른쪽 스틱 상하
        this->declare_parameter<int>("deadman_button",  4);     // LB 버튼
        this->declare_parameter<double>("max_speed",    3.0);   // m/s
        this->declare_parameter<double>("max_steering", 45.0);  // degrees
        this->declare_parameter<double>("deadzone",     0.1);
        this->declare_parameter<bool>("invert_steering", false);
        this->declare_parameter<bool>("invert_speed",    false);
        this->declare_parameter<bool>("use_deadman",     true); // false로 하면 RB 없이도 동작

        steering_axis_    = this->get_parameter("steering_axis").as_int();
        speed_axis_       = this->get_parameter("speed_axis").as_int();
        deadman_button_   = this->get_parameter("deadman_button").as_int();
        max_speed_        = this->get_parameter("max_speed").as_double();
        max_steering_     = this->get_parameter("max_steering").as_double();
        deadzone_         = this->get_parameter("deadzone").as_double();
        invert_steering_  = this->get_parameter("invert_steering").as_bool();
        invert_speed_     = this->get_parameter("invert_speed").as_bool();
        use_deadman_      = this->get_parameter("use_deadman").as_bool();

        joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 10,
            std::bind(&JoyToDriveNode::joyCallback, this, std::placeholders::_1));

        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);

        RCLCPP_INFO(this->get_logger(), "Joy to Drive Node started");
        RCLCPP_INFO(this->get_logger(), "  왼쪽 스틱 좌우 → 조향 (axis %d)", steering_axis_);
        RCLCPP_INFO(this->get_logger(), "  오른쪽 스틱 상하 → 속도 (axis %d)", speed_axis_);
        if (use_deadman_) {
            RCLCPP_INFO(this->get_logger(), "  RB 버튼 누르는 동안만 이동 (button %d)", deadman_button_);
        } else {
            RCLCPP_INFO(this->get_logger(), "  데드맨 스위치 OFF - 항상 활성");
        }
    }

private:
    int steering_axis_, speed_axis_, deadman_button_;
    double max_speed_, max_steering_, deadzone_;
    bool invert_steering_, invert_speed_, use_deadman_;

    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

    double applyDeadzone(double value) {
        if (std::abs(value) < deadzone_) return 0.0;
        // 데드존 이후 0부터 선형 보간
        double sign = (value > 0) ? 1.0 : -1.0;
        return sign * (std::abs(value) - deadzone_) / (1.0 - deadzone_);
    }

    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg) {
        // 데드맨 스위치 확인
        bool deadman_ok = !use_deadman_;
        if (use_deadman_) {
            if (deadman_button_ < (int)msg->buttons.size()) {
                deadman_ok = (msg->buttons[deadman_button_] == 1);
            }
        }

        double steering_raw = 0.0;
        double speed_raw    = 0.0;

        if (steering_axis_ < (int)msg->axes.size())
            steering_raw = msg->axes[steering_axis_];
        if (speed_axis_ < (int)msg->axes.size())
            speed_raw = msg->axes[speed_axis_];

        // 반전 적용
        if (invert_steering_) steering_raw = -steering_raw;
        if (invert_speed_)    speed_raw    = -speed_raw;

        // 데드존 적용
        double steering_out = applyDeadzone(steering_raw);
        double speed_out    = applyDeadzone(speed_raw);

        // 데드맨 해제 시 정지
        if (!deadman_ok) {
            steering_out = 0.0;
            speed_out    = 0.0;
        }

        // 스케일 변환
        double steering_deg = steering_out * max_steering_;
        double speed_mps    = speed_out    * max_speed_;

        // Publish
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.header.stamp    = this->now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = steering_deg * M_PI / 180.0;  // rad
        drive_msg.drive.speed          = speed_mps;

        drive_pub_->publish(drive_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JoyToDriveNode>());
    rclcpp::shutdown();
    return 0;
}
