#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


class TrackerNodeScale(Node):
    def __init__(self):
        super().__init__('tracking_person')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Parámetros
        self.rgb_topic = self.declare_parameter(
            'rgb_topic',
            '/oakd/rgb/preview/image_raw'
        ).get_parameter_value().string_value

        self.model_file = self.declare_parameter(
            'model_file',
            'TrackerPerson.pt'
        ).get_parameter_value().string_value

        self.conf_thres = self.declare_parameter(
            'conf',
            0.65
        ).get_parameter_value().double_value

        # Entrenamiento 160x120
        self.target_w = self.declare_parameter(
            'target_w',
            160
        ).get_parameter_value().integer_value

        self.target_h = self.declare_parameter(
            'target_h',
            120
        ).get_parameter_value().integer_value

        # Deadband horizontal (normalizado, ej 0.05 ≈ 5% del ancho)
        self.deadband = self.declare_parameter(
            'deadband',
            0.05
        ).get_parameter_value().double_value

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.rgb_topic, self.image_cb, qos)

        self.pub_annotated = self.create_publisher(Image, 'tracking_person/annotated', 10)
        self.pub_label = self.create_publisher(String, 'tracking_person/label', 10)
        self.pub_target = self.create_publisher(PointStamped, 'tracking_person/target', 10)
        self.pub_size_ratio = self.create_publisher(Float32, 'tracking_person/size_ratio', 10)

        share_dir = Path(get_package_share_directory('tracking_person'))
        model_path = share_dir / 'models' / self.model_file
        if not model_path.exists():
            self.get_logger().error(f'No encontré el modelo: {model_path}')
            raise FileNotFoundError(model_path)

        self.get_logger().info(f'Cargando modelo: {model_path}')
        self.model = YOLO(str(model_path))
        self.names = self.model.names

        # Callback para actualizar parámetros en runtime
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info('✅ tracking_person listo — modelo cargado y tópicos configurados.')

    def image_cb(self, msg: Image):
        # 1) ROS ➜ OpenCV
        try:
            orig = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        # 2) Resize a target_w × target_h
        try:
            frame = cv2.resize(orig, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            self.get_logger().warn(f'Resize falló: {e}')
            return

        # 3) Inferencia
        try:
            results = self.model(frame, conf=self.conf_thres, verbose=False)
        except Exception as e:
            self.get_logger().warn(f'YOLO inferencia falló: {e}')
            return

        annotated = frame.copy()
        top_label, top_conf = None, 0.0
        best_box = None  # (x1,y1,x2,y2) en espacio target_w x target_h

        try:
            r0 = results[0]
            if hasattr(r0, 'boxes') and r0.boxes is not None and len(r0.boxes):
                for b in r0.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0])

                    label = (
                        self.names[cls_id]
                        if (isinstance(self.names, list) and 0 <= cls_id < len(self.names))
                        else str(cls_id)
                    )
                    if conf > top_conf:
                        top_conf, top_label = conf, label
                        best_box = (x1, y1, x2, y2)

                # dibujar todas las cajas (opcional)
                for b in r0.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    label = (
                        self.names[cls_id]
                        if (isinstance(self.names, list) and 0 <= cls_id < len(self.names))
                        else str(cls_id)
                    )
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f'{label} {conf:.2f}',
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
            else:
                cv2.putText(
                    annotated,
                    'no detections',
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
        except Exception as e:
            self.get_logger().warn(f'Post-proceso falló: {e}')

        # 3.5) Overlays: línea central + deadband
        cx_img = int(self.target_w * 0.5)
        band = int(self.deadband * self.target_w)
        left_band = cx_img - band
        right_band = cx_img + band
        # Centro (blanco), deadband (amarillo)
        cv2.line(annotated, (cx_img, 0), (cx_img, self.target_h - 1), (255, 255, 255), 1)
        cv2.line(annotated, (left_band, 0), (left_band, self.target_h - 1), (0, 255, 255), 1)
        cv2.line(annotated, (right_band, 0), (right_band, self.target_h - 1), (0, 255, 255), 1)

        # 4) Publish label y target si hay best_box
        if top_label is not None and best_box is not None:
            self.pub_label.publish(String(data=top_label))

            x1, y1, x2, y2 = best_box
            bx = (x1 + x2) / 2.0
            by = (y1 + y2) / 2.0
            # normalizado 0..1
            nx = float(bx / self.target_w)
            ny = float(by / self.target_h)

            # puntito del centro de bbox
            cv2.circle(annotated, (int(bx), int(by)), 2, (0, 0, 255), -1)

            # ====== relación de tamaño del bounding box ======
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            img_area = float(self.target_w * self.target_h)
            box_area = float(box_w * box_h)
            size_ratio = box_area / img_area if img_area > 0 else 0.0  # 0..1

            # Solo porcentaje, esquina superior derecha
            percent_text = f'{size_ratio * 100:.1f}%'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                percent_text, font, font_scale, thickness
            )

            x = self.target_w - text_w - 4
            y = 10 + text_h // 2

            cv2.putText(
                annotated,
                percent_text,
                (x, y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )

            # Publicar tamaño relativo en tópico
            self.pub_size_ratio.publish(Float32(data=size_ratio))
            # ====== fin relación de tamaño ======

            pt = PointStamped()
            pt.header = msg.header  # timestamp sync con imagen
            pt.header.frame_id = 'camera'  # etiqueta lógica
            pt.point.x = nx
            pt.point.y = ny
            pt.point.z = 0.0
            self.pub_target.publish(pt)

        # 5) Publicar imagen anotada (target_w x target_h)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.pub_annotated.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f'Error publicando imagen: {e}')

    def parameters_callback(self, params):
        """
        Callback para actualizar parámetros en tiempo real:
        - deadband (DOUBLE)
        - conf (DOUBLE)
        - target_w (INTEGER)
        - target_h (INTEGER)
        """
        for param in params:
            if param.name == 'deadband' and param.type_ == Parameter.Type.DOUBLE:
                self.deadband = param.value
                self.get_logger().info(f'✅ deadband actualizado a {self.deadband}')
            elif param.name == 'conf' and param.type_ == Parameter.Type.DOUBLE:
                self.conf_thres = param.value
                self.get_logger().info(f'✅ conf_thres actualizado a {self.conf_thres}')
            elif param.name == 'target_w' and param.type_ == Parameter.Type.INTEGER:
                self.target_w = param.value
                self.get_logger().info(f'✅ target_w actualizado a {self.target_w}')
            elif param.name == 'target_h' and param.type_ == Parameter.Type.INTEGER:
                self.target_h = param.value
                self.get_logger().info(f'✅ target_h actualizado a {self.target_h}')

        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = TrackerNodeScale()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
