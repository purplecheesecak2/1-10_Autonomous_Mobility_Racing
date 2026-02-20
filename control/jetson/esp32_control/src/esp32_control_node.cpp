/*
 * Jetson Control Node for ESP32
 * - Receives AckermannDriveStamped from ackermann_mux
 * - Applies safety processing
 * - Sends commands to ESP32 via Serial
 * - Receives encoder feedback from ESP32
 *
 * [수정사항]
 * 1. serial/serial.h → POSIX termios (추가 라이브러리 불필요)
 * 2. 속도 감쇄 로직을 Jetson에서만 처리 (ESP32 측 제거에 대응)
 * 3. readline() 블로킹 → available() + read() 논블로킹 방식으로 교체
 * 4. /odom → /ego_racecar/odom (planning node 토픽 이름 맞춤)
 */

#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>

// POSIX serial
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>

class ESP32ControlNode : public rclcpp::Node {
public:
    ESP32ControlNode() : Node("esp32_control_node"), serial_fd_(-1) {
        // Parameters
        this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
        this->declare_parameter<int>("serial_baudrate", 115200);
        this->declare_parameter<double>("max_steering", 45.0);
        this->declare_parameter<double>("min_steering", -45.0);
        this->declare_parameter<double>("max_speed", 3.0);
        this->declare_parameter<double>("min_speed", -3.0);
        this->declare_parameter<double>("steering_filter_alpha", 0.3);
        this->declare_parameter<double>("high_steering_threshold", 20.0);
        this->declare_parameter<double>("speed_reduction_factor", 0.5);
        this->declare_parameter<double>("timeout_threshold", 0.5);

        serial_port_             = this->get_parameter("serial_port").as_string();
        serial_baudrate_         = this->get_parameter("serial_baudrate").as_int();
        max_steering_            = this->get_parameter("max_steering").as_double();
        min_steering_            = this->get_parameter("min_steering").as_double();
        max_speed_               = this->get_parameter("max_speed").as_double();
        min_speed_               = this->get_parameter("min_speed").as_double();
        steering_filter_alpha_   = this->get_parameter("steering_filter_alpha").as_double();
        high_steering_threshold_ = this->get_parameter("high_steering_threshold").as_double();
        speed_reduction_factor_  = this->get_parameter("speed_reduction_factor").as_double();
        timeout_threshold_       = this->get_parameter("timeout_threshold").as_double();

        // Serial 초기화
        if (!openSerial(serial_port_, serial_baudrate_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open serial port: %s", serial_port_.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Serial port opened: %s at %d baud",
                        serial_port_.c_str(), serial_baudrate_);
        }

        // Subscriber (from ackermann_mux)
        drive_sub_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10,
            std::bind(&ESP32ControlNode::driveCallback, this, std::placeholders::_1));

        // Publisher
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/ego_racecar/odom", 10);

        // 제어 루프 (50Hz)
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&ESP32ControlNode::controlLoop, this));

        // 시리얼 읽기 (100Hz, 논블로킹)
        serial_read_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&ESP32ControlNode::readSerialFeedback, this));

        filtered_steering_ = 0.0;
        current_speed_     = 0.0;

        RCLCPP_INFO(this->get_logger(), "ESP32 Control Node Initialized");
    }

    ~ESP32ControlNode() {
        if (serial_fd_ >= 0) {
            sendToESP32(0.0, 0.0);
            close(serial_fd_);
        }
    }

private:
    int serial_fd_;
    std::string serial_port_;
    int serial_baudrate_;

    rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr serial_read_timer_;

    double max_steering_, min_steering_;
    double max_speed_, min_speed_;
    double steering_filter_alpha_;
    double high_steering_threshold_;
    double speed_reduction_factor_;
    double timeout_threshold_;

    double desired_steering_ = 0.0;
    double target_speed_     = 0.0;
    double filtered_steering_;
    double current_speed_;

    rclcpp::Time last_drive_time_;
    bool received_drive_ = false;

    // 논블로킹 시리얼 파싱용 버퍼
    std::string serial_buffer_;

    // ========== POSIX SERIAL ==========
    bool openSerial(const std::string& port, int baudrate) {
        serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (serial_fd_ < 0) return false;

        struct termios tty;
        if (tcgetattr(serial_fd_, &tty) != 0) {
            close(serial_fd_);
            serial_fd_ = -1;
            return false;
        }

        speed_t speed = B115200;
        if      (baudrate == 9600)   speed = B9600;
        else if (baudrate == 57600)  speed = B57600;
        else if (baudrate == 115200) speed = B115200;

        cfsetospeed(&tty, speed);
        cfsetispeed(&tty, speed);

        // 8N1, no hardware flow control
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cflag |= CREAD | CLOCAL;

        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;
        tty.c_oflag &= ~ONLCR;

        tty.c_cc[VMIN]  = 0;
        tty.c_cc[VTIME] = 0;

        tcsetattr(serial_fd_, TCSANOW, &tty);
        return true;
    }

    int serialAvailable() {
        if (serial_fd_ < 0) return 0;
        int bytes = 0;
        ioctl(serial_fd_, FIONREAD, &bytes);
        return bytes;
    }

    std::string serialRead(int n) {
        std::vector<char> buf(n);
        int count = ::read(serial_fd_, buf.data(), n);
        if (count <= 0) return "";
        return std::string(buf.data(), count);
    }

    void serialWrite(const std::string& s) {
        if (serial_fd_ < 0) return;
        ::write(serial_fd_, s.c_str(), s.size());
    }

    // ========== CALLBACKS ==========
    void driveCallback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
        if (std::isnan(msg->drive.steering_angle) || std::isinf(msg->drive.steering_angle) ||
            std::isnan(msg->drive.speed)          || std::isinf(msg->drive.speed)) {
            RCLCPP_WARN(this->get_logger(), "Invalid drive command (NaN/Inf)");
            return;
        }

        desired_steering_ = msg->drive.steering_angle * 180.0 / M_PI;
        target_speed_     = msg->drive.speed;
        last_drive_time_  = this->now();
        received_drive_   = true;
    }

    // 논블로킹 시리얼 읽기: 1바이트씩 읽어 \n 감지 시 파싱
    void readSerialFeedback() {
        if (serial_fd_ < 0) return;

        int avail = serialAvailable();
        if (avail == 0) return;

        std::string chunk = serialRead(avail);
        for (char c : chunk) {
            if (c == '\n') {
                parseFeedbackLine(serial_buffer_);
                serial_buffer_.clear();
            } else {
                serial_buffer_ += c;
                if (serial_buffer_.size() > 64) {
                    serial_buffer_.clear();
                }
            }
        }
    }

    void parseFeedbackLine(const std::string& line) {
        if (line.find("SPEED:") == 0) {
            try {
                float measured_speed = std::stof(line.substr(6));
                if (!std::isnan(measured_speed) && !std::isinf(measured_speed)) {
                    current_speed_ = measured_speed;
                    publishOdometry();
                }
            } catch (...) {
                // 파싱 실패 무시
            }
        }
    }

    void publishOdometry() {
        auto msg = nav_msgs::msg::Odometry();
        msg.header.stamp    = this->now();
        msg.header.frame_id = "odom";
        msg.child_frame_id  = "base_link";
        msg.twist.twist.linear.x  = current_speed_;
        msg.twist.twist.linear.y  = 0.0;
        msg.twist.twist.angular.z = 0.0;
        odom_pub_->publish(msg);
    }

    // ========== SAFETY FUNCTIONS ==========
    double limitSteering(double s) {
        return std::max(min_steering_, std::min(max_steering_, s));
    }

    double limitSpeed(double v) {
        return std::max(min_speed_, std::min(max_speed_, v));
    }

    double filterSteering(double raw) {
        filtered_steering_ = steering_filter_alpha_ * raw +
                             (1.0 - steering_filter_alpha_) * filtered_steering_;
        return filtered_steering_;
    }

    // 속도 감쇄: Jetson에서만 처리 (선형 감쇄)
    double adjustSpeedForSteering(double speed, double steering) {
        double abs_steering = std::abs(steering);
        if (abs_steering <= high_steering_threshold_) return speed;

        double max_steering = 45.0;
        double ratio = (abs_steering - high_steering_threshold_) /
                       (max_steering - high_steering_threshold_);
        ratio = std::max(0.0, std::min(1.0, ratio));

        double factor = 1.0 - ratio * (1.0 - speed_reduction_factor_);
        return speed * factor;
    }

    bool checkTimeout() {
        if (!received_drive_) return false;
        double dt = (this->now() - last_drive_time_).seconds();
        if (dt > timeout_threshold_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Drive command timeout! Safe stop.");
            return true;
        }
        return false;
    }

    // ========== SERIAL SEND ==========
    void sendToESP32(double steering_deg, double speed_mps) {
        if (serial_fd_ < 0) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                  "Serial port not open!");
            return;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << steering_deg << "," << speed_mps << "\n";

        serialWrite(oss.str());

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                              "Sent to ESP32: %s", oss.str().c_str());
    }

    // ========== CONTROL LOOP ==========
    void controlLoop() {
        double final_steering = 0.0;
        double final_speed    = 0.0;

        if (checkTimeout()) {
            // Timeout: safe stop
        } else if (received_drive_) {
            double limited_steering = limitSteering(desired_steering_);
            double filtered         = filterSteering(limited_steering);
            double limited_speed    = limitSpeed(target_speed_);
            double adjusted_speed   = adjustSpeedForSteering(limited_speed, filtered);

            final_steering = filtered;
            final_speed    = adjusted_speed;
        }

        sendToESP32(final_steering, final_speed);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ESP32ControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
