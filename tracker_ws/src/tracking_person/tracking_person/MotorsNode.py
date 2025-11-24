#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped, PointStamped

class MotorsNode(Node):
    def __init__(self):
        super().__init__('motors_node')

        # Parámetros
        self.kp = self.declare_parameter('kp', 1.5).get_parameter_value().double_value
        self.deadband = self.declare_parameter('deadband', 0.05).get_parameter_value().double_value
        self.max_w = self.declare_parameter('max_w', 1.2).get_parameter_value().double_value
        self.timeout_s = self.declare_parameter('timeout_s', 1.0).get_parameter_value().double_value
        # sin búsqueda por defecto
        self.search_on_no_target = self.declare_parameter('search_on_no_target', False).get_parameter_value().bool_value

        # Subscripción al target
        self.sub = self.create_subscription(PointStamped, 'tracking_person/target', self.target_cb, 10)

        # Publisher /cmd_vel (TwistStamped) QoS BEST_EFFORT
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile_sensor_data)

        self.last_target = None
        self.last_target_time = None
        self.have_target = False

        # Control loop @ 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('🛞 motors_node online — TwistStamped a /cmd_vel (BEST_EFFORT), sin búsqueda.')

    def target_cb(self, msg: PointStamped):
        self.last_target = msg
        self.last_target_time = self.get_clock().now()
        self.have_target = True

    def publish_twist(self, lin_x: float, ang_z: float):
        now = self.get_clock().now()
        ts = TwistStamped()
        ts.header.stamp = now.to_msg()
        ts.header.frame_id = 'base_link'

        ts.twist.linear.x = float(lin_x)
        # Don't move forward/backward
        ts.twist.linear.y = 0.0
        ts.twist.linear.z = 0.0
        ts.twist.angular.x = 0.0
        ts.twist.angular.y = 0.0
        ts.twist.angular.z = float(ang_z)

        self.cmd_pub.publish(ts)


    def control_loop(self):
        now = self.get_clock().now()

        # ¿Target fresco?
        fresh = self.have_target and self.last_target_time and \
                (now - self.last_target_time) <= Duration(seconds=self.timeout_s)

        if not fresh:
            self.have_target = False
            # 👉 sin búsqueda: zeros
            self.publish_twist(0.0, 0.0)
            return

        # Control yaw por error horizontal
        nx = float(self.last_target.point.x)  # 0..1
        err = nx - 0.5

        if abs(err) < self.deadband:
            omega = 0.0
        else:
            omega = self.kp * (0.5 - nx)
            omega = max(-self.max_w, min(self.max_w, omega))

        self.publish_twist(0.0, omega)

def main():
    rclpy.init()
    node = MotorsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
