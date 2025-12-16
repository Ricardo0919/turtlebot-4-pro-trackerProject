#!/usr/bin/env python3
# ===============================================================================
# Project: Turtlebot 4 Pro - Person Tracker - Motors Node - Yaw + Distance Control
# Student Project
# Date: December 7th, 2025
# Students:
#   - Sergi Fernandez Mendez
#   - Ricardo Sierra Roa
# ===============================================================================

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import TwistStamped, PointStamped
from std_msgs.msg import Float32


class MotorsNode(Node):
    """
    MotorsNode
    ----------
    ROS2 node for Turtlebot 4 Pro.

    Responsibilities:
      - Subscribes to:
          * tracking_person/target      (geometry_msgs/PointStamped)
          * tracking_person/size_ratio  (std_msgs/Float32)
      - Publishes:
          * /cmd_vel (geometry_msgs/TwistStamped)

    Behavior:
      - Controls yaw using the horizontal position of the target.
      - Controls distance using the size_ratio (forward-only approach, no reverse).
    """

    def __init__(self):
        super().__init__('motors_node')

        # =========================== Basic yaw control parameters ===========================

        # kp: proportional gain for yaw control
        self.kp = self.declare_parameter('kp', 1.5).get_parameter_value().double_value
        # deadband: horizontal error range around the image center where we do not rotate
        self.deadband = self.declare_parameter('deadband', 0.05).get_parameter_value().double_value
        # max_w: maximum angular velocity (rad/s)
        self.max_w = self.declare_parameter('max_w', 1.5).get_parameter_value().double_value
        # timeout_s: maximum age of target/size messages before considering them stale
        self.timeout_s = self.declare_parameter('timeout_s', 0.5).get_parameter_value().double_value
        # search_on_no_target is kept for compatibility, not used in the control loop
        self.search_on_no_target = self.declare_parameter(
            'search_on_no_target', False
        ).get_parameter_value().bool_value

        # ===================== Distance control parameters (size_ratio) =====================

        # near_threshold is kept for compatibility but NOT used anymore.
        self.near_threshold = self.declare_parameter(
            'near_threshold', 0.60
        ).get_parameter_value().double_value

        # If size_ratio < far_threshold (0.65) → move forward.
        # If size_ratio >= far_threshold → hold position (stop).
        self.far_threshold = self.declare_parameter(
            'far_threshold', 0.65
        ).get_parameter_value().double_value

        # Constant forward linear speed while approaching the target
        self.linear_speed = self.declare_parameter(
            'linear_speed', 0.15  # m/s
        ).get_parameter_value().double_value

        # ================================== Subscriptions ==================================

        # Target position in normalized image coordinates (0..1 in x)
        self.sub_target = self.create_subscription(
            PointStamped,
            'tracking_person/target',
            self.target_cb,
            10,
        )

        # Size ratio of the detected person in the image (proxy for distance)
        self.sub_size = self.create_subscription(
            Float32,
            'tracking_person/size_ratio',
            self.size_cb,
            10,
        )

        # ======================== Publisher: /cmd_vel (TwistStamped) ========================

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            qos_profile_sensor_data,
        )

        # Last received target and timing
        self.last_target = None
        self.last_target_time = None
        self.have_target = False

        # Last received size_ratio and timing
        self.last_size_ratio = None
        self.last_size_time = None

        # Throttled logging to avoid spamming the console
        self.last_log_time = self.get_clock().now()

        # Main control timer @ 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)

        # Dynamic parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(
            'motors_node online — yaw + distance (forward-only) using size_ratio → /cmd_vel (TwistStamped).'
        )

    # ================================ Subscription callbacks ================================
    def target_cb(self, msg: PointStamped):
        """Callback for the tracked target position (normalized image coordinates)."""
        self.last_target = msg
        self.last_target_time = self.get_clock().now()
        self.have_target = True

    def size_cb(self, msg: Float32):
        """Callback for the size_ratio of the detected person."""
        self.last_size_ratio = float(msg.data)
        self.last_size_time = self.get_clock().now()

    # ==================================== Publish helper ====================================
    def publish_twist(self, lin_x: float, ang_z: float):
        """
        Helper to publish a TwistStamped command in the base_link frame.
        lin_x: forward velocity in m/s
        ang_z: yaw angular velocity in rad/s
        """
        now = self.get_clock().now()
        ts = TwistStamped()
        ts.header.stamp = now.to_msg()
        ts.header.frame_id = 'base_link'

        ts.twist.linear.x = float(lin_x)
        ts.twist.linear.y = 0.0
        ts.twist.linear.z = 0.0
        ts.twist.angular.x = 0.0
        ts.twist.angular.y = 0.0
        ts.twist.angular.z = float(ang_z)

        self.cmd_pub.publish(ts)

    # =================================== Main control loop ===================================
    def control_loop(self):
        now = self.get_clock().now()

        # Check if the target message is fresh enough
        fresh_target = (
            self.have_target
            and self.last_target_time is not None
            and (now - self.last_target_time) <= Duration(seconds=self.timeout_s)
        )

        # Check if the size_ratio message is fresh enough
        fresh_size = (
            self.last_size_ratio is not None
            and self.last_size_time is not None
            and (now - self.last_size_time) <= Duration(seconds=self.timeout_s)
        )

        if not fresh_target or not fresh_size:
            # No reliable information → stop the robot for safety
            self.publish_twist(0.0, 0.0)
            return

        # ---------- Yaw control (horizontal alignment) ----------
        # nx: normalized x in [0, 1], where 0.5 is the image center
        nx = float(self.last_target.point.x)
        err = nx - 0.5

        if abs(err) < self.deadband:
            # Inside deadband → do not rotate
            omega = 0.0
        else:
            # Simple proportional controller around the image center
            omega = self.kp * (0.5 - nx)
            # Saturate angular velocity
            omega = max(-self.max_w, min(self.max_w, omega))

        # ---------- Distance control using size_ratio (forward-only) ----------
        sr = float(self.last_size_ratio)
        vx = 0.0
        state = "hold"

        # Rule:
        #   - While size_ratio < far_threshold → move forward.
        #   - When size_ratio >= far_threshold → stop (we are "close enough").
        if sr < self.far_threshold:
            vx = self.linear_speed
            state = "forward"
        else:
            vx = 0.0
            state = "hold"

        # Throttled debug log (1 Hz) for monitoring behavior
        if (now - self.last_log_time) > Duration(seconds=1.0):
            self.last_log_time = now
            self.get_logger().info(
                f'size_ratio={sr:.3f} | state={state} | lin_x={vx:.3f} | ang_z={omega:.3f}'
            )

        # Publish final velocity command
        self.publish_twist(vx, omega)

    # ============================== Dynamic parameter handling ==============================
    def parameters_callback(self, params):
        """
        Dynamic reconfigure callback.
        Allows updating gains and thresholds at runtime via ROS2 parameters.
        """
        for p in params:
            if p.name == 'kp' and p.type_ == Parameter.Type.DOUBLE:
                self.kp = p.value
                self.get_logger().info(f'kp updated to {self.kp}')
            elif p.name == 'deadband' and p.type_ == Parameter.Type.DOUBLE:
                self.deadband = p.value
                self.get_logger().info(f'deadband updated to {self.deadband}')
            elif p.name == 'max_w' and p.type_ == Parameter.Type.DOUBLE:
                self.max_w = p.value
                self.get_logger().info(f'max_w updated to {self.max_w}')
            elif p.name == 'timeout_s' and p.type_ == Parameter.Type.DOUBLE:
                self.timeout_s = p.value
                self.get_logger().info(f'timeout_s updated to {self.timeout_s}')
            elif p.name == 'near_threshold' and p.type_ == Parameter.Type.DOUBLE:
                # Kept for compatibility; not used in control_loop anymore.
                self.near_threshold = p.value
                self.get_logger().info(
                    f'near_threshold (unused in control loop) updated to {self.near_threshold}'
                )
            elif p.name == 'far_threshold' and p.type_ == Parameter.Type.DOUBLE:
                self.far_threshold = p.value
                self.get_logger().info(f'far_threshold updated to {self.far_threshold}')
            elif p.name == 'linear_speed' and p.type_ == Parameter.Type.DOUBLE:
                self.linear_speed = p.value
                self.get_logger().info(f'linear_speed updated to {self.linear_speed}')
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = MotorsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
