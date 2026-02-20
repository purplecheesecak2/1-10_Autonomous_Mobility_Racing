import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import sys, select, termios, tty

settings = termios.tcgetattr(sys.stdin)

class TeleopDriver(Node):
    def __init__(self):
        super().__init__('teleop_driver')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        print("키보드로 운전하세요: W(전진), A(좌), D(우), S(정지), X(후진)")
        print("속도/조향각은 고정값입니다. (Ctrl+C로 종료)")

    def publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.publisher_.publish(msg)

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    rclpy.init()
    node = TeleopDriver()
    speed = 0.0
    steering = 0.0
    
    try:
        while True:
            key = getKey()
            if key == 'w':
                speed = 1.5      # 전진 속도
                steering = 0.0
            elif key == 'a':
                steering = 0.4   # 좌회전
            elif key == 'd':
                steering = -0.4  # 우회전
            elif key == 's':
                speed = 0.0      # 정지
                steering = 0.0
            elif key == 'x':
                speed = -1.0     # 후진
            elif key == '\x03':  # Ctrl+C
                break
            
            # 키를 떼면 조향을 풀고 싶다면 로직 추가 필요하지만, 
            # 일단 누를 때마다 명령을 보냄
            node.publish_drive(speed, steering)

    except Exception as e:
        print(e)
    finally:
        node.publish_drive(0.0, 0.0) # 종료 시 정지
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
