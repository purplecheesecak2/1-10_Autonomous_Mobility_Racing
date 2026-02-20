cmake_minimum_required(VERSION 3.8)
project(obstacle_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 의존성 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# 2. 실행 파일 생성 (소스 코드 지정)
add_executable(obstacle_detector src/obstacle_detector.cpp)

# 3. 헤더 파일 경로 포함 (필수)
target_include_directories(obstacle_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# 4. 라이브러리 링크 (의존성 연결)
ament_target_dependencies(obstacle_detector
  rclcpp
  sensor_msgs
  visualization_msgs
)

# 5. 설치 규칙 (ros2 run으로 실행하기 위해)
install(TARGETS obstacle_detector
  DESTINATION lib/${PROJECT_NAME})

ament_package()
