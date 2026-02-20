/*
 * Joy to Drive Node (with Autonomous/Manual Mode Switch)
 * Logitech F710 (X mode)
 *
 * 모드:
 *   기본값 = 자율주행 모드 (플래닝 노드의 /autonomous/drive 명령 전달)
 *   A 버튼 (button 0) 누르면 수동/자율 모드 토글
 *
 * 수동 모드 조작:
 *   LB (button 4)         → 데드맨 스위치 (누르는 동안만 이동)
 *   왼쪽 스틱 좌우 (axis 0) → 조향 (-45° ~ +45°)
 *   오른쪽 스틱 상하 (axis 4) → 속도 (위=전진, 최대 3m/s)
 *
 * 토픽:
 *   subscribe: /joy                        (조이스틱 입력)
 *   subscribe: /autonomous/drive           (플래닝 노드 명령)
 *   publish:   /drive (AckermannDriveStamped)
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <cmath>

enum class DriveMode { AUTONOMOUS, MANUAL };

class JoyToDriveNode : public rclcpp::Node {
public:
    JoyToDriveNode() : Node("joy_to_drive"), mode_(DriveMode::AUTONOMOUS) {
        // Parameters
        this->declare_parameter<int>("steering_axis",    0);
        this->declare_parameter<int>("speed_axis",       4);
        this->declare_parameter<int>("deadman_button",   4);   // LB
        this->declare_parameter<int>("mode_toggle_button", 0); // A 버튼
        this->declare_parameter<double>("max_speed",     3.0);
        this->declare_parameter<double>("max_steering",  45.0);
        this->declare_parameter<double>("deadzone",      0.1);
        this->declare_parameter<bool>("invert_steering", false);
        this->declare_parameter<bool>("invert_speed",    false);

        steering_axis_       = this->get_parameter("steering_axis").as_int();
        speed_axis_          = this->get_parameter("speed_axis").as_int();
        deadman_button_      = this->get_parameter("deadman_button").as_int();
        mode_toggle_button_  = this->get_parameter("mode_toggle_button").as_int();
        max_speed_           = this->get_parameter("max_speed").as_double();
        max_steering_        = this->get_parameter("max_steering").as_double();
        deadzone_            = this->get_parameter("deadzone").as_double();
        invert_steering_     = this->get_parameter("invert_steering").as_bool();
        invert_speed_        = this->get_parameter("invert_speed").as_bool();

        // Subscribers
        joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 10,
            std::bind(&JoyToDriveNode::joyCallback, this, std::placeholders::_1));

        auto_sub_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
            "/autonomous/drive", 10,
            std::bind(&JoyToDriveNode::autoCallback, this, std::placeholders::_1));

        // Publisher
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);

        RCLCPP_INFO(this->get_logger(), "=== Drive Mux Node Started ===");
        RCLCPP_INFO(this->get_logger(), "현재 모드: [자율주행]");
        RCLCPP_INFO(this->get_logger(), "A 버튼으로 자율/수동 모드 전환");
        RCLCPP_INFO(this->get_logger(), "수동 모드: LB 누른 채로 오른쪽 스틱 조작");
    }

private:
    DriveMode mode_;
    bool prev_toggle_button_ = false;

    int steering_axis_, speed_axis_, deadman_button_, mode_toggle_button_;
    double max_speed_, max_steering_, deadzone_;
    bool invert_steering_, invert_speed_;

    ackermann_msgs::msg::AckermannDriveStamped latest_auto_cmd_;

    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr auto_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

    // 자율주행 명령 저장
    void autoCallback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
        latest_auto_cmd_ = *msg;

        // 자율주행 모드일 때만 전달
        if (mode_ == DriveMode::AUTONOMOUS) {
            drive_pub_->publish(latest_auto_cmd_);
        }
    }

    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg) {
        // A 버튼으로 모드 토글 (버튼을 눌렀다 뗄 때 전환)
        bool toggle_pressed = (mode_toggle_button_ < (int)msg->buttons.size()) &&
                              (msg->buttons[mode_toggle_button_] == 1);

        if (toggle_pressed && !prev_toggle_button_) {
            if (mode_ == DriveMode::AUTONOMOUS) {
                mode_ = DriveMode::MANUAL;
                RCLCPP_WARN(this->get_logger(), ">>> 모드 전환: [수동 조작] (LB + 오른쪽 스틱)");
            } else {
                mode_ = DriveMode::AUTONOMOUS;
                RCLCPP_INFO(this->get_logger(), ">>> 모드 전환: [자율주행]");
                // 자율주행 복귀 시 정지 명령 한 번 보내고 자율 명령 기다림
                publishStop();
            }
        }
        prev_toggle_button_ = toggle_pressed;

        // 수동 모드일 때만 조이스틱 처리
        if (mode_ == DriveMode::MANUAL) {
            processManual(msg);
        }
    }

    void processManual(const sensor_msgs::msg::Joy::SharedPtr msg) {
        // 데드맨 확인 (LB)
        bool deadman_ok = (deadman_button_ < (int)msg->buttons.size()) &&
                          (msg->buttons[deadman_button_] == 1);

        double steering_raw = 0.0;
        double speed_raw    = 0.0;

        if (steering_axis_ < (int)msg->axes.size())
            steering_raw = msg->axes[steering_axis_];
        if (speed_axis_ < (int)msg->axes.size())
            speed_raw = msg->axes[speed_axis_];

        if (invert_steering_) steering_raw = -steering_raw;
        if (invert_speed_)    speed_raw    = -speed_raw;

        double steering_out = deadman_ok ? applyDeadzone(steering_raw) : 0.0;
        double speed_out    = deadman_ok ? applyDeadzone(speed_raw)    : 0.0;

        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.header.stamp    = this->now();
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = steering_out * max_steering_ * M_PI / 180.0;
        drive_msg.drive.speed          = speed_out * max_speed_;

        drive_pub_->publish(drive_msg);
    }

    void publishStop() {
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.header.stamp    = this->now();
        msg.header.frame_id = "base_link";
        msg.drive.steering_angle = 0.0;
        msg.drive.speed          = 0.0;
        drive_pub_->publish(msg);
    }

    double applyDeadzone(double value) {
        if (std::abs(value) < deadzone_) return 0.0;
        double sign = (value > 0) ? 1.0 : -1.0;
        return sign * (std::abs(value) - deadzone_) / (1.0 - deadzone_);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JoyToDriveNode>());
    rclcpp::shutdown();
    return 0;
}
