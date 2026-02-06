cmake_minimum_required(VERSION 3.8)
project(gap_follow_hard)

cmake_minimum_required(VERSION 3.8)
project(gap_follow_hard)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 의존성 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)

# 1. 실행 파일 생성 (이름을 reactive_node_hard로 지정)
# 소스 파일(src/reactive_node.cpp)은 그대로 둡니다.
add_executable(reactive_node_hard src/reactive_node.cpp)

# 2. 의존성 연결 (위에서 만든 reactive_node_hard 에 연결)
ament_target_dependencies(reactive_node_hard
  rclcpp
  sensor_msgs
  ackermann_msgs
)

# 3. 설치 설정 (reactive_node_hard 를 설치)
install(TARGETS reactive_node_hard
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
