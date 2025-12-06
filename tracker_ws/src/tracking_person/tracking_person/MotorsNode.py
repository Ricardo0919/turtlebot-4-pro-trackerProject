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

        # =====================
        # Parámetros (iniciales)
        # =====================
        self.kp = self.declare_parameter('kp', 1.5).get_parameter_value().double_value
        self.deadband = self.declare_parameter('deadband', 0.05).get_parameter_value().double_value
        self.max_w = self.declare_parameter('max_w', 1.2).get_parameter_value().double_value
        self.timeout_s = self.declare_parameter('timeout_s', 1.0).get_parameter_value().double_value
        # sin búsqueda por defecto
        self.search_on_no_target = self.declare_parameter(
            'search_on_no_target', False
        ).get_parameter_value().bool_value

        # Umbrales para movimiento lineal basado en size_ratio
        # ratio ∈ [0,1]; 1.0 = bbox ocupa toda la imagen
        self.near_threshold = self.declare_parameter(
            'near_threshold', 0.70  # >= 70% → retroceder
        ).get_parameter_value().double_value

        self.far_threshold = self.declare_parameter(
            'far_threshold', 0.50  # <= 50% → avanzar
        ).get_parameter_value().double_value

        self.linear_speed = self.declare_parameter(
            'linear_speed', 0.15  # m/s hacia adelante/atrás
        ).get_parameter_value().double_value

        # =====================
        # Subs / Pubs
        # =====================
        # Target horizontal (x normalizado 0..1)
        self.sub = self.create_subscription(
            PointStamped,
            'tracking_person/target',
            self.target_cb,
            10
        )

        # Size ratio del bounding box (Float32, 0..1)
        self.size_sub = self.create_subscription(
            Float32,
            'tracking_person/size_ratio',
            self.size_cb,
            10
        )

        # Publisher /cmd_vel (TwistStamped) QoS BEST_EFFORT
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            qos_profile_sensor_data
        )

        self.last_target = None
        self.last_target_time = None
        self.have_target = False

        # último size_ratio recibido
        self.last_size_ratio = None

        # Control loop @ 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)

        # callback para rqt_reconfigure / parámetros dinámicos
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(
            '🛞 motors_node online — TwistStamped a /cmd_vel (BEST_EFFORT), '
            'yaw PID + control lineal por size_ratio.'
        )

    # =====================
    # Callbacks
    # =====================
    def target_cb(self, msg: PointStamped):
        self.last_target = msg
        self.last_target_time = self.get_clock().now()
        self.have_target = True

    def size_cb(self, msg: Float32):
        # Guardamos directamente el último ratio (0..1)
        self.last_size_ratio = float(msg.data)

    # =====================
    # Helpers
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

        # ¿Target fresco?
        fresh = (
            self.have_target
            and self.last_target_time is not None
            and (now - self.last_target_time) <= Duration(seconds=self.timeout_s)
        )

        if not fresh:
            self.have_target = False
            # sin búsqueda → todo en cero
            self.publish_twist(0.0, 0.0)
            return

        # -------- Giro (yaw) — NO LO TOCAMOS --------
        nx = float(self.last_target.point.x)  # 0..1
        err = nx - 0.5

        if abs(err) < self.deadband:
            omega = 0.0
        else:
            omega = self.kp * (0.5 - nx)
            # saturación
            omega = max(-self.max_w, min(self.max_w, omega))

        # -------- Movimiento lineal basado en size_ratio --------
        lin_x = 0.0

        if self.last_size_ratio is not None:
            r = self.last_size_ratio  # 0..1

            # >= near_threshold → muy cerca → retroceder
            if r >= self.near_threshold:
                lin_x = -self.linear_speed
            # <= far_threshold → muy lejos → avanzar
            elif r <= self.far_threshold:
                lin_x = self.linear_speed
            # entre far_threshold y near_threshold → stay
            else:
                lin_x = 0.0

        # Publicar resultante
        self.publish_twist(lin_x, omega)

    # =====================
    # Parámetros dinámicos
    # =====================
    def parameters_callback(self, params):
        for param in params:
            if param.name == 'kp' and param.type_ == Parameter.Type.DOUBLE:
                self.kp = param.value
                self.get_logger().info(f'✅ kp actualizado a {self.kp}')

            elif param.name == 'deadband' and param.type_ == Parameter.Type.DOUBLE:
                self.deadband = param.value
                self.get_logger().info(f'✅ deadband actualizado a {self.deadband}')

            elif param.name == 'max_w' and param.type_ == Parameter.Type.DOUBLE:
                self.max_w = param.value
                self.get_logger().info(f'✅ max_w actualizado a {self.max_w}')

            elif param.name == 'timeout_s' and param.type_ == Parameter.Type.DOUBLE:
                self.timeout_s = param.value
                self.get_logger().info(f'✅ timeout_s actualizado a {self.timeout_s}')

            elif param.name == 'search_on_no_target' and param.type_ == Parameter.Type.BOOL:
                self.search_on_no_target = param.value
                self.get_logger().info(
                    f'✅ search_on_no_target actualizado a {self.search_on_no_target}'
                )

            elif param.name == 'near_threshold' and param.type_ == Parameter.Type.DOUBLE:
                self.near_threshold = param.value
                self.get_logger().info(
                    f'✅ near_threshold actualizado a {self.near_threshold}'
                )

            elif param.name == 'far_threshold' and param.type_ == Parameter.Type.DOUBLE:
                self.far_threshold = param.value
                self.get_logger().info(
                    f'✅ far_threshold actualizado a {self.far_threshold}'
                )

            elif param.name == 'linear_speed' and param.type_ == Parameter.Type.DOUBLE:
                self.linear_speed = param.value
                self.get_logger().info(
                    f'✅ linear_speed actualizado a {self.linear_speed}'
                )

        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = MotorsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
