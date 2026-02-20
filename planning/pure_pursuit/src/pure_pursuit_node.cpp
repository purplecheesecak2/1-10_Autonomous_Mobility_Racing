#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// ─────────────────────────────────────────────────────────────
//  Pure Pursuit + 4-State Overtaking State Machine
//
//  States:
//    LANE_KEEPING  : normal center-lane following
//    LANE_CHANGE   : executing move to target side lane
//    OVERTAKING    : following side lane past the obstacle
//    RETURNING     : moving back to center lane
//
//  Corner detection:
//    κ = |Δyaw / Δs| per waypoint
//    κ > CORNER_KAPPA → no lane change (slow down instead)
//
//  Perception team interface (future):
//    Currently uses waypoints.csv to generate 3 lanes.
//    When perception team is ready, subscribe to their topics
//    and replace path_center_ / path_left_ / path_right_ with
//    their lane coordinate arrays.
// ─────────────────────────────────────────────────────────────

const std::string CSV_PATH  = "/home/louisdarong/ros2_ws/waypoints.csv";
const double      WHEELBASE = 0.33;   // m
const double      MAX_STEER = 0.4189; // rad (~24°)
const double      LANE_WIDTH = 0.6;   // m – lateral offset per lane

struct Waypoint {
    double x, y, yaw, curvature;
};

class PurePursuit : public rclcpp::Node {
public:
    // ── Driving states ──────────────────────────────────────
    enum class State {
        LANE_KEEPING,  // follow center, watch for obstacles
        LANE_CHANGE,   // moving laterally to side lane
        OVERTAKING,    // following side lane past obstacle
        RETURNING      // returning to center lane
    };

    PurePursuit() : Node("pure_pursuit_node") {
        drive_pub_  = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        lane_pub_   = create_publisher<visualization_msgs::msg::MarkerArray>("/lane_markers", 10);
        target_pub_ = create_publisher<visualization_msgs::msg::Marker>("/target_marker", 10);
        state_pub_  = create_publisher<visualization_msgs::msg::Marker>("/driving_state_marker", 10);

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10,
            std::bind(&PurePursuit::odom_callback, this, std::placeholders::_1));

        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&PurePursuit::scan_callback, this, std::placeholders::_1));

        load_waypoints();
        RCLCPP_INFO(get_logger(),
            "Pure Pursuit: 4-State Machine + curvature corner detection. Ready.");
    }

private:
    // ── Tuning parameters ────────────────────────────────────
    const double MAX_SPEED      = 2.5;   // m/s – straight line
    const double MIN_SPEED      = 0.5;   // m/s – minimum
    const double CORNER_SPEED   = 1.2;   // m/s – max in corner zone
    const double LD_BASE        = 0.8;   // m   – lookahead base
    const double K_V            = 0.1;   //     – speed → lookahead gain

    // Obstacle thresholds (LiDAR based)
    const double DETECT_DIST    = 2.5;   // m: start considering overtake
    const double CLEAR_DIST     = 3.5;   // m: obstacle considered passed
    const double EMERG_DIST     = 0.30;  // m: emergency stop

    // Half-angle for front distance check (deg)
    const double FRONT_DEG      = 15.0;
    // Minimum side clearance to allow lane change (m)
    const double SIDE_CLEAR_MIN = 1.2;

    // Corner detection: no overtaking if κ > this
    // κ = |Δyaw / Δs|  (rad/m)
    //   < 0.15: straight
    //   0.15–0.3: gentle curve
    //   > 0.3: sharp corner
    const double CORNER_KAPPA   = 0.3;

    // Lane-settling threshold: considered "in lane" when this close (m)
    const double LANE_SETTLE_DIST = 0.35;

    // Cooldown between lane change decisions (s)
    const double CHANGE_COOLDOWN = 2.0;

    // ── State ────────────────────────────────────────────────
    State  state_       = State::LANE_KEEPING;
    int    target_lane_ = 0;   // 0:center, 1:left, -1:right

    double car_x_   = 0, car_y_  = 0;
    double car_yaw_ = 0, car_spd_ = 0;
    bool   has_odom_ = false;
    int    nearest_idx_ = 0;
    bool   nearest_initialized_ = false;

    sensor_msgs::msg::LaserScan::SharedPtr scan_;
    bool has_scan_ = false;

    rclcpp::Time last_change_ = rclcpp::Time(0);

    // ── Paths ────────────────────────────────────────────────
    // ★ Perception team interface note:
    //   In the future, these will be filled by subscribing to
    //   the perception team's lane coordinate topic instead of
    //   loading from CSV.
    std::vector<Waypoint> path_center_;
    std::vector<Waypoint> path_left_;
    std::vector<Waypoint> path_right_;

    // ── Publishers / Subscribers ─────────────────────────────
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr       lane_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr            target_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr            state_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr  odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // ─────────────────────────────────────────────────────────
    void load_waypoints() {
        std::ifstream file(CSV_PATH);
        if (!file.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open waypoints: %s", CSV_PATH.c_str());
            return;
        }

        std::string line;
        std::vector<std::pair<double, double>> raw;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            while (std::getline(ss, cell, ',')) row.push_back(cell);
            if (row.size() >= 2)
                raw.push_back({std::stod(row[0]), std::stod(row[1])});
        }

        size_t n = raw.size();
        if (n == 0) { RCLCPP_ERROR(get_logger(), "Empty waypoints file!"); return; }

        // ── Compute yaw for each waypoint ─────────────────────
        std::vector<double> yaws(n);
        for (size_t i = 0; i < n; ++i) {
            size_t nxt = (i + 1) % n;
            yaws[i] = std::atan2(raw[nxt].second - raw[i].second,
                                 raw[nxt].first  - raw[i].first);
        }

        // ── Compute curvature κ = |Δyaw| / Δs ─────────────────
        std::vector<double> kappas(n);
        for (size_t i = 0; i < n; ++i) {
            size_t prv = (i + n - 1) % n;
            double ds = std::hypot(raw[i].first  - raw[prv].first,
                                   raw[i].second - raw[prv].second);
            double dyaw = yaws[i] - yaws[prv];
            while (dyaw >  M_PI) dyaw -= 2 * M_PI;
            while (dyaw < -M_PI) dyaw += 2 * M_PI;
            kappas[i] = (ds > 0.001) ? std::abs(dyaw) / ds : 0.0;
        }

        // ── Build center / left / right paths ─────────────────
        for (size_t i = 0; i < n; ++i) {
            double x = raw[i].first, y = raw[i].second;
            double yaw = yaws[i], kappa = kappas[i];

            path_center_.push_back({x, y, yaw, kappa});

            // Left: +90° perpendicular offset
            path_left_.push_back({
                x + LANE_WIDTH * (-std::sin(yaw)),
                y + LANE_WIDTH * ( std::cos(yaw)),
                yaw, kappa
            });

            // Right: -90° perpendicular offset
            path_right_.push_back({
                x + LANE_WIDTH * ( std::sin(yaw)),
                y + LANE_WIDTH * (-std::cos(yaw)),
                yaw, kappa
            });
        }

        RCLCPP_INFO(get_logger(), "Loaded %zu waypoints (with curvature)", n);
        publish_lanes();
    }

    // ─────────────────────────────────────────────────────────
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        car_x_   = msg->pose.pose.position.x;
        car_y_   = msg->pose.pose.position.y;
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        car_yaw_ = std::atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz));
        car_spd_ = msg->twist.twist.linear.x;
        has_odom_ = true;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        scan_     = msg;
        has_scan_ = true;
        if (has_odom_) compute_drive();
    }

    // ─────────────────────────────────────────────────────────
    // Minimum range in a forward angular window (±FRONT_DEG)
    double front_distance() {
        if (!has_scan_) return 99.0;
        const auto& s = *scan_;
        int n  = (int)s.ranges.size();
        int ci = (int)((0.0 - s.angle_min) / s.angle_increment);
        int hw = (int)((FRONT_DEG * M_PI / 180.0) / s.angle_increment);
        double mn = s.range_max;
        for (int i = ci - hw; i <= ci + hw; ++i) {
            if (i < 0 || i >= n) continue;
            float r = s.ranges[i];
            if (std::isfinite(r) && r > s.range_min)
                mn = std::min(mn, (double)r);
        }
        return mn;
    }

    // Minimum range within [deg_lo, deg_hi] (0° = forward, CCW positive)
    double side_distance(double deg_lo, double deg_hi) {
        if (!has_scan_) return 0.0;
        const auto& s = *scan_;
        int n  = (int)s.ranges.size();
        double rad_lo = deg_lo * M_PI / 180.0;
        double rad_hi = deg_hi * M_PI / 180.0;
        int si = std::clamp((int)((rad_lo - s.angle_min) / s.angle_increment), 0, n-1);
        int ei = std::clamp((int)((rad_hi - s.angle_min) / s.angle_increment), 0, n-1);
        double mn = s.range_max;
        for (int i = si; i <= ei; ++i) {
            float r = s.ranges[i];
            if (std::isfinite(r) && r > s.range_min)
                mn = std::min(mn, (double)r);
        }
        return mn;
    }

    // Curvature κ at the current nearest waypoint
    double current_curvature() {
        if (path_center_.empty()) return 0.0;
        return path_center_[nearest_idx_].curvature;
    }

    bool in_corner() { return current_curvature() > CORNER_KAPPA; }

    // Lateral distance from car to nearest waypoint of given lane
    double dist_to_lane(int lane) {
        const std::vector<Waypoint>& path =
            (lane ==  1) ? path_left_  :
            (lane == -1) ? path_right_ : path_center_;
        if (path.empty()) return 99.0;
        double best = 99.0;
        int sz      = (int)path.size();
        int window  = std::min(sz, 60);
        for (int k = -window/2; k <= window/2; ++k) {
            int idx = ((nearest_idx_ + k) % sz + sz) % sz;
            double d = std::hypot(path[idx].x - car_x_, path[idx].y - car_y_);
            best = std::min(best, d);
        }
        return best;
    }

    bool cooldown_ok() {
        return (last_change_.nanoseconds() == 0) ||
               ((this->now() - last_change_).seconds() > CHANGE_COOLDOWN);
    }

    // ─────────────────────────────────────────────────────────
    void compute_drive() {
        if (path_center_.empty()) return;

        update_nearest();
        double fd = front_distance();

        // ── Emergency stop ────────────────────────────────────
        if (fd < EMERG_DIST) {
            publish_drive(0.0, 0.0);
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "EMERGENCY STOP  front=%.2fm", fd);
            publish_state_marker();
            return;
        }

        // ── State Machine ─────────────────────────────────────
        switch (state_) {

            // ────────────────────────────────────────────────
            case State::LANE_KEEPING:
                target_lane_ = 0;   // ensure center
                if (fd < DETECT_DIST && cooldown_ok()) {
                    if (in_corner()) {
                        // Corner: cannot overtake → just slow down
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                            "[CORNER] Obstacle %.2fm ahead, κ=%.3f. Slowing.",
                            fd, current_curvature());
                    } else {
                        // Straight: pick a free side lane
                        double lc = side_distance( 60.0,  90.0);  // left
                        double rc = side_distance(270.0, 300.0);  // right
                        RCLCPP_INFO(get_logger(),
                            "Obstacle %.2fm | left_clear=%.2fm right_clear=%.2fm κ=%.3f",
                            fd, lc, rc, current_curvature());

                        if (lc > SIDE_CLEAR_MIN) {
                            target_lane_  = 1;   // Left
                            state_        = State::LANE_CHANGE;
                            last_change_  = this->now();
                            RCLCPP_INFO(get_logger(), "LANE_KEEPING → LANE_CHANGE (Left)");
                        } else if (rc > SIDE_CLEAR_MIN) {
                            target_lane_  = -1;  // Right
                            state_        = State::LANE_CHANGE;
                            last_change_  = this->now();
                            RCLCPP_INFO(get_logger(), "LANE_KEEPING → LANE_CHANGE (Right)");
                        } else {
                            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                "Both sides blocked! Staying put.");
                        }
                    }
                }
                break;

            // ────────────────────────────────────────────────
            case State::LANE_CHANGE:
                // Wait until car has settled into the target lane
                if (dist_to_lane(target_lane_) < LANE_SETTLE_DIST) {
                    state_ = State::OVERTAKING;
                    RCLCPP_INFO(get_logger(), "LANE_CHANGE → OVERTAKING (lane=%d)", target_lane_);
                }
                break;

            // ────────────────────────────────────────────────
            case State::OVERTAKING:
                // Obstacle cleared → start returning
                if (fd > CLEAR_DIST) {
                    state_       = State::RETURNING;
                    last_change_ = this->now();
                    RCLCPP_INFO(get_logger(), "OVERTAKING → RETURNING");
                }
                break;

            // ────────────────────────────────────────────────
            case State::RETURNING:
                if (cooldown_ok()) {
                    target_lane_ = 0;
                    if (dist_to_lane(0) < LANE_SETTLE_DIST) {
                        state_ = State::LANE_KEEPING;
                        RCLCPP_INFO(get_logger(), "RETURNING → LANE_KEEPING");
                    }
                }
                break;
        }

        // ── Follow selected path ──────────────────────────────
        const std::vector<Waypoint>& path =
            (target_lane_ ==  1) ? path_left_  :
            (target_lane_ == -1) ? path_right_ : path_center_;

        double steer, speed;
        pure_pursuit(path, steer, speed);

        // Reduce speed in corners regardless of state
        if (in_corner()) speed = std::min(speed, CORNER_SPEED);

        // Extra slow-down: obstacle ahead + corner → can't dodge
        if (state_ == State::LANE_KEEPING && fd < DETECT_DIST && in_corner())
            speed = std::min(speed, MIN_SPEED);

        publish_drive(steer, speed);
        publish_lanes();
        publish_state_marker();
    }

    // ─────────────────────────────────────────────────────────
    // Update nearest_idx_: full O(n) search on first call, window search after
    void update_nearest() {
        int sz = (int)path_center_.size();
        if (!nearest_initialized_) {
            // Full scan to find the true closest waypoint at startup
            double best = 1e9;
            for (int i = 0; i < sz; ++i) {
                double d = std::hypot(path_center_[i].x - car_x_,
                                      path_center_[i].y - car_y_);
                if (d < best) { best = d; nearest_idx_ = i; }
            }
            nearest_initialized_ = true;
            RCLCPP_INFO(get_logger(), "Initial nearest waypoint: idx=%d", nearest_idx_);
            return;
        }
        // Window search forward from cached index
        int window = std::min(sz, 200);
        double best = std::hypot(path_center_[nearest_idx_].x - car_x_,
                                 path_center_[nearest_idx_].y - car_y_);
        for (int k = 1; k <= window; ++k) {
            int idx = (nearest_idx_ + k) % sz;
            double d = std::hypot(path_center_[idx].x - car_x_,
                                  path_center_[idx].y - car_y_);
            if (d < best) { best = d; nearest_idx_ = idx; }
        }
    }

    // Pure pursuit geometry on a given path
    void pure_pursuit(const std::vector<Waypoint>& path,
                      double& steer, double& speed) {
        double ld  = LD_BASE + K_V * std::abs(car_spd_);
        int    sz  = (int)path.size();

        // Find lookahead point
        int target_idx = nearest_idx_;
        for (int i = 0; i < sz; ++i) {
            int idx = (nearest_idx_ + i) % sz;
            double d = std::hypot(path[idx].x - car_x_, path[idx].y - car_y_);
            if (d > ld) { target_idx = idx; break; }
        }

        const auto& wp = path[target_idx];
        double dx    = wp.x - car_x_;
        double dy    = wp.y - car_y_;
        double alpha = std::atan2(dy, dx) - car_yaw_;
        while (alpha >  M_PI) alpha -= 2 * M_PI;
        while (alpha < -M_PI) alpha += 2 * M_PI;

        steer = std::atan2(2.0 * WHEELBASE * std::sin(alpha), ld);
        steer = std::clamp(steer, -MAX_STEER, MAX_STEER);

        // Speed: reduce for sharp steering
        double sf = 1.0 - std::abs(steer) / MAX_STEER * 0.4;
        speed = std::max(MIN_SPEED, MAX_SPEED * sf);

        // Target marker colour by lane
        float r = 1.0f, g = 1.0f, b = 0.0f;   // center → yellow
        if (target_lane_ ==  1) { r = 0.0f; g = 0.5f; b = 1.0f; }  // left → blue
        if (target_lane_ == -1) { r = 1.0f; g = 0.3f; b = 0.0f; }  // right → orange
        publish_target_marker(wp.x, wp.y, r, g, b);
    }

    // ─────────────────────────────────────────────────────────
    void publish_drive(double steer, double speed) {
        auto msg = ackermann_msgs::msg::AckermannDriveStamped();
        msg.drive.steering_angle = steer;
        msg.drive.speed          = speed;
        drive_pub_->publish(msg);
    }

    void publish_lanes() {
        visualization_msgs::msg::MarkerArray arr;

        auto make_line = [&](int id,
                             const std::vector<Waypoint>& path,
                             float r, float g, float b,
                             const std::string& ns)
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "map";
            m.header.stamp    = rclcpp::Time(0);
            m.ns = ns; m.id = id;
            m.type   = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.orientation.w = 1.0;
            m.scale.x = 0.05;
            m.color.a = 1.0f; m.color.r = r; m.color.g = g; m.color.b = b;
            for (const auto& wp : path) {
                geometry_msgs::msg::Point p;
                p.x = wp.x; p.y = wp.y; p.z = 0.0;
                m.points.push_back(p);
            }
            return m;
        };

        // Green = current lane (Perception team: 현재 차선)
        // Yellow = adjacent overtake lane (Perception team: 이동 가능 차선)
        arr.markers.push_back(make_line(0, path_center_, 0.0f, 1.0f, 0.0f, "center_lane"));
        arr.markers.push_back(make_line(1, path_left_,   1.0f, 1.0f, 0.0f, "left_lane"));
        arr.markers.push_back(make_line(2, path_right_,  1.0f, 0.5f, 0.0f, "right_lane"));
        lane_pub_->publish(arr);
    }

    void publish_target_marker(double x, double y, float r, float g, float b) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp    = get_clock()->now();
        m.ns = "target"; m.id = 0;
        m.type   = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = x; m.pose.position.y = y;
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 0.3;
        m.color.a = 1.0f; m.color.r = r; m.color.g = g; m.color.b = b;
        target_pub_->publish(m);
    }

    void publish_state_marker() {
        // State label displayed above the car in RViz
        std::string label;
        float r = 1.0f, g = 1.0f, b = 1.0f;

        switch (state_) {
            case State::LANE_KEEPING:
                label = "LANE_KEEPING"; r = 0.0f; g = 1.0f; b = 0.0f; break;
            case State::LANE_CHANGE:
                label = (target_lane_ == 1) ? "LANE_CHANGE L" : "LANE_CHANGE R";
                r = 1.0f; g = 1.0f; b = 0.0f; break;
            case State::OVERTAKING:
                label = "OVERTAKING"; r = 0.0f; g = 0.5f; b = 1.0f; break;
            case State::RETURNING:
                label = "RETURNING"; r = 1.0f; g = 0.5f; b = 0.0f; break;
        }

        if (in_corner()) {
            char buf[8]; std::snprintf(buf, sizeof(buf), "%.2f", current_curvature());
            label += " [CORNER k=" + std::string(buf) + "]";
            r = 1.0f; g = 0.2f; b = 0.2f;
        }

        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp    = get_clock()->now();
        m.ns = "state_text"; m.id = 0;
        m.type   = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = car_x_;
        m.pose.position.y = car_y_;
        m.pose.position.z = 1.0;
        m.pose.orientation.w = 1.0;
        m.scale.z = 0.5;
        m.color.a = 1.0f; m.color.r = r; m.color.g = g; m.color.b = b;
        m.text = label;
        state_pub_->publish(m);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}
