#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import TwistStamped, PointStamped
from std_msgs.msg import Float32


class MotorsNode(Node):
    def __init__(self):
        super().__init__('motors_node')

        # Parámetros básicos (giro)
        self.kp = self.declare_parameter('kp', 1.5).get_parameter_value().double_value
        self.deadband = self.declare_parameter('deadband', 0.05).get_parameter_value().double_value
        self.max_w = self.declare_parameter('max_w', 1.2).get_parameter_value().double_value
        self.timeout_s = self.declare_parameter('timeout_s', 1.0).get_parameter_value().double_value
        self.search_on_no_target = self.declare_parameter(
            'search_on_no_target', False
        ).get_parameter_value().bool_value

        # Parámetros de distancia con size_ratio
        # near_threshold se deja por compatibilidad pero YA NO se usa.
        self.near_threshold = self.declare_parameter(
            'near_threshold', 0.60
        ).get_parameter_value().double_value

        # Si size_ratio < far_threshold (0.65) → avanzar
        # Si size_ratio >= far_threshold → detenerse
        self.far_threshold = self.declare_parameter(
            'far_threshold', 0.65
        ).get_parameter_value().double_value

        self.linear_speed = self.declare_parameter(
            'linear_speed', 0.15  # m/s
        ).get_parameter_value().double_value

        # Subs
        self.sub_target = self.create_subscription(
            PointStamped,
            'tracking_person/target',
            self.target_cb,
            10,
        )
        self.sub_size = self.create_subscription(
            Float32,
            'tracking_person/size_ratio',
            self.size_cb,
            10,
        )

        # Pub /cmd_vel
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            qos_profile_sensor_data,
        )

        self.last_target = None
        self.last_target_time = None
        self.have_target = False

        self.last_size_ratio = None
        self.last_size_time = None

        # Log throttle
        self.last_log_time = self.get_clock().now()

        # Timer @ 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)

        # Dynamic params
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(
            '🛞 motors_node online — yaw + distancia (solo forward) con size_ratio /cmd_vel (TwistStamped).'
        )

    # =====================
    # Callbacks de subs
    # =====================
    def target_cb(self, msg: PointStamped):
        self.last_target = msg
        self.last_target_time = self.get_clock().now()
        self.have_target = True

    def size_cb(self, msg: Float32):
        self.last_size_ratio = float(msg.data)
        self.last_size_time = self.get_clock().now()

    # =====================
    # Publish helper
    # =====================
    def publish_twist(self, lin_x: float, ang_z: float):
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

    # =====================
    # Control loop
    # =====================
    def control_loop(self):
        now = self.get_clock().now()

        # Fresh target?
        fresh_target = (
            self.have_target
            and self.last_target_time is not None
            and (now - self.last_target_time) <= Duration(seconds=self.timeout_s)
        )

        # Fresh size_ratio?
        fresh_size = (
            self.last_size_ratio is not None
            and self.last_size_time is not None
            and (now - self.last_size_time) <= Duration(seconds=self.timeout_s)
        )

        if not fresh_target or not fresh_size:
            # No info confiable → no moverse
            self.publish_twist(0.0, 0.0)
            return

        # ---------- Control yaw ----------
        nx = float(self.last_target.point.x)  # 0..1
        err = nx - 0.5

        if abs(err) < self.deadband:
            omega = 0.0
        else:
            omega = self.kp * (0.5 - nx)
            omega = max(-self.max_w, min(self.max_w, omega))

        # ---------- Control distancia con size_ratio (solo forward) ----------
        sr = float(self.last_size_ratio)
        vx = 0.0
        state = "hold"

        # Regla: mientras size_ratio sea menor a far_threshold (0.65) → avanza
        if sr < self.far_threshold:
            vx = self.linear_speed
            state = "forward"
        else:
            vx = 0.0
            state = "hold"

        # Log suave para debug
        if (now - self.last_log_time) > Duration(seconds=1.0):
            self.last_log_time = now
            self.get_logger().info(
                f'size_ratio={sr:.3f} | state={state} | lin_x={vx:.3f} | ang_z={omega:.3f}'
            )

        # Publish comando final
        self.publish_twist(vx, omega)

    # =====================
    # Dynamic params
    # =====================
    def parameters_callback(self, params):
        for p in params:
            if p.name == 'kp' and p.type_ == Parameter.Type.DOUBLE:
                self.kp = p.value
                self.get_logger().info(f'✅ kp actualizado a {self.kp}')
            elif p.name == 'deadband' and p.type_ == Parameter.Type.DOUBLE:
                self.deadband = p.value
                self.get_logger().info(f'✅ deadband actualizado a {self.deadband}')
            elif p.name == 'max_w' and p.type_ == Parameter.Type.DOUBLE:
                self.max_w = p.value
                self.get_logger().info(f'✅ max_w actualizado a {self.max_w}')
            elif p.name == 'timeout_s' and p.type_ == Parameter.Type.DOUBLE:
                self.timeout_s = p.value
                self.get_logger().info(f'✅ timeout_s actualizado a {self.timeout_s}')
            elif p.name == 'near_threshold' and p.type_ == Parameter.Type.DOUBLE:
                # Se mantiene solo para compat; no se usa en control_loop
                self.near_threshold = p.value
                self.get_logger().info(f'✅ near_threshold (sin uso) actualizado a {self.near_threshold}')
            elif p.name == 'far_threshold' and p.type_ == Parameter.Type.DOUBLE:
                self.far_threshold = p.value
                self.get_logger().info(f'✅ far_threshold actualizado a {self.far_threshold}')
            elif p.name == 'linear_speed' and p.type_ == Parameter.Type.DOUBLE:
                self.linear_speed = p.value
                self.get_logger().info(f'✅ linear_speed actualizado a {self.linear_speed}')
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = MotorsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
