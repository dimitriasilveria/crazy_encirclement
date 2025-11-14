import rclpy
import numpy as np
from rclpy.node import Node
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import Position
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration


class GPSNode(Node):
    def __init__(self):
        super().__init__('gps_node')
        self.declare_parameter('robot', 'C01')
        self.declare_parameter('V', [0.1, 0.1, 0.1])
        self.declare_parameter('rate', 10.0)

        self.robot = self.get_parameter('robot').get_parameter_value().string_value
        self.V = self.get_parameter('V').get_parameter_value().double_array_value
        self.rate = self.get_parameter('rate').get_parameter_value().double_value

        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))

        self.create_subscription(
            NamedPoseArray, "/poses",
            self._callback, qos_profile
        )

        self.position = Position()
        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)
        self.position_pub = self.create_publisher(Position, f'/{self.robot}/gps_position', 10)

    def _callback(self, msg: NamedPoseArray):
        '''Callback function to process incoming NamedPoseArray messages from Vicon and publish noisy GPS positions.'''
        self.position = Position()
        self.position.header.stamp = self.get_clock().now().to_msg()
    
        # Add Gaussian noise to the position of the specified robot
        for pose in msg.poses:
            if pose.name == self.robot:
                noise = np.random.multivariate_normal(np.zeros(3), np.diag(np.square(self.V)))
                self.position.x = pose.position.x + noise[0]
                self.position.y = pose.position.y + noise[1]
                self.position.z = pose.position.z + noise[2]

    def timer_callback(self):
        '''Timer callback to publish the noisy GPS position at the specified rate.'''
        self.position_pub.publish(self.position)


if __name__ == '__main__':
    rclpy.init()
    gps_node = GPSNode()
    rclpy.spin(gps_node)
    gps_node.destroy_node()
    rclpy.shutdown()