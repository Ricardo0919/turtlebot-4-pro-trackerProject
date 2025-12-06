#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped, PointStamped

class MotorsMovingNode(Node):
    def __init__(self):
        super().__init__('motors_node')

        # Parámetros yaw (giro)
        self.kp = self.declare_parameter('kp', 1.5).get_parameter_value().double_value
        self.deadband = self.declare_parameter('deadband', 0.05).get_parameter_value().double_value
        self.max_w = self.declare_parameter('max_w', 1.5).get_parameter_value().double_value
        self.timeout_s = self.declare_parameter('timeout_s', 1.0).get_parameter_value().double_value
        self.search_on_no_target = self.declare_parameter('search_on_no_target', False).get_parameter_value().bool_value

        # Parámetros distancia (adelante/atrás)
        # scale_target ≈ qué tan “grande” quieres ver a la persona (0..1)
        self.scale_target = self.declare_parameter('scale_target', 0.35).get_parameter_value().double_value
        # banda alrededor del target donde no se mueve
        self.scale_band = self.declare_parameter('scale_band', 0.05).get_parameter_value().double_value
        # ganancia y saturación de velocidad lineal
        self.kv = self.declare_parameter('kv', 0.8).get_parameter_value().double_value
        self.max_v = self.declare_parameter('max_v', 0.2).get_parameter_value().double_value

        # Subscripción al target
        self.sub = self.create_subscription(PointStamped, 'tracking_person/target', self.target_cb, 10)

        # Publisher /cmd_vel (TwistStamped) QoS BEST_EFFORT
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile_sensor_data)

        self.last_target = None
        self.last_target_time = None
        self.have_target = False

        # Control loop @ 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('🛞 motors_node online — yaw + distancia a /cmd_vel.')

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
            # sin target: detenerse (sin búsqueda por ahora)
            self.publish_twist(0.0, 0.0)
            return

        # 1) Control de YAW (giro)
        nx = float(self.last_target.point.x)  # 0..1
        err_yaw = nx - 0.5

        if abs(err_yaw) < self.deadband:
            omega = 0.0
        else:
            omega = self.kp * (0.5 - nx)
            omega = max(-self.max_w, min(self.max_w, omega))

        # 2) Control de DISTANCIA (adelante/atrás) usando point.z
        scale = float(self.last_target.point.z)  # 0..1: tamaño relativo de la persona

        # si por alguna razón viene 0.0, evita hacer tonterías
        if scale <= 0.0:
            v = 0.0
        else:
            # queremos que scale ≈ scale_target
            err_dist = self.scale_target - scale

            if abs(err_dist) < self.scale_band:
                v = 0.0
            else:
                v = self.kv * err_dist
                # saturación
                if v > self.max_v:
                    v = self.max_v
                elif v < -self.max_v:
                    v = -self.max_v

        # v > 0 → avanzar (persona lejos)
        # v < 0 → retroceder (persona muy cerca)

        self.publish_twist(v, omega)

def main():
    rclpy.init()
    node = MotorsMovingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
