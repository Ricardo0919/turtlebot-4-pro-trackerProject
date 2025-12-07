#!/usr/bin/env python3
# ===============================================================================
# Project: Turtlebot 4 Pro - Person Tracker - Tracker Node (Scale + YOLO .pt)
# Student Project
# Date: December 7th, 2025
# Students:
#   - Sergi Fernandez Mendez
#   - Ricardo Sierra Roa
# ===============================================================================

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
    """
    TrackerNodeScale
    ----------------
    ROS2 node for person detection and tracking using a YOLO .pt model,
    including estimation of the relative bounding box area (size_ratio).

    Responsibilities:
      - Subscribes to:
          * rgb_topic (sensor_msgs/Image) - RGB image from the camera.
      - Publishes:
          * tracking_person/annotated (sensor_msgs/Image)
              Annotated image resized to (target_w, target_h).
          * tracking_person/label (std_msgs/String)
              Label of the best detection.
          * tracking_person/target (geometry_msgs/PointStamped)
              Normalized center (x, y in [0,1]) of the best bounding box.
          * tracking_person/size_ratio (std_msgs/Float32)
              Relative area of the best bounding box with respect to the resized image.

    Behavior:
      - Resizes the input image to (target_w, target_h).
      - Runs YOLO inference with a .pt model.
      - Selects the highest-confidence bounding box as the tracking target.
      - Computes:
          * Normalized center of the bounding box.
          * Relative area (size_ratio = box_area / image_area).
      - Draws:
          * Bounding boxes and labels.
          * Center line and horizontal deadband.
          * Size ratio percentage on the annotated image.
    """

    def __init__(self):
        super().__init__('tracking_person')

        # ================================ QoS configuration ================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ================================== Parameters ==================================
        # Topic for the RGB input image (camera feed)
        self.rgb_topic = self.declare_parameter(
            'rgb_topic',
            '/oakd/rgb/preview/image_raw'
        ).get_parameter_value().string_value

        # YOLO .pt model file name (located in the 'models' directory of the package)
        self.model_file = self.declare_parameter(
            'model_file',
            'TrackerPerson.pt'
        ).get_parameter_value().string_value

        # Minimum confidence threshold for detections
        self.conf_thres = self.declare_parameter(
            'conf',
            0.65
        ).get_parameter_value().double_value

        # Training / inference resolution: 160 x 120 (can be adjusted)
        self.target_w = self.declare_parameter(
            'target_w',
            160
        ).get_parameter_value().integer_value

        self.target_h = self.declare_parameter(
            'target_h',
            120
        ).get_parameter_value().integer_value

        # Horizontal deadband in normalized coordinates (e.g. 0.05 ≈ 5% of the image width)
        self.deadband = self.declare_parameter(
            'deadband',
            0.05
        ).get_parameter_value().double_value

        # ================================== Interfaces ==================================
        self.bridge = CvBridge()
        # Subscriber for RGB images
        self.sub = self.create_subscription(Image, self.rgb_topic, self.image_cb, qos)

        # Publisher for annotated image (resized)
        self.pub_annotated = self.create_publisher(Image, 'tracking_person/annotated', 10)
        # Publisher for best detection label
        self.pub_label = self.create_publisher(String, 'tracking_person/label', 10)
        # Publisher for normalized target position
        self.pub_target = self.create_publisher(PointStamped, 'tracking_person/target', 10)
        # Publisher for relative bounding box area
        self.pub_size_ratio = self.create_publisher(Float32, 'tracking_person/size_ratio', 10)

        # =============================== Model loading ===============================
        share_dir = Path(get_package_share_directory('tracking_person'))
        model_path = share_dir / 'models' / self.model_file
        if not model_path.exists():
            self.get_logger().error(f'Could not find model file at: {model_path}')
            raise FileNotFoundError(model_path)

        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(str(model_path))
        self.names = self.model.names

        # Dynamic parameter callback for runtime tuning
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info(
            'tracking_person node ready: model loaded and topics configured.'
        )

    # ==================================== Callbacks ====================================
    def image_cb(self, msg: Image):
        """
        Main callback for incoming images.
        Steps:
          1. Convert ROS Image → OpenCV.
          2. Resize to (target_w, target_h).
          3. Run YOLO inference.
          4. Select best detection and compute normalized target and size_ratio.
          5. Draw overlays and publish outputs.
        """
        # 1) ROS Image → OpenCV (BGR)
        try:
            orig = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion error: {e}')
            return

        # 2) Resize to target resolution (target_w × target_h)
        try:
            frame = cv2.resize(orig, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            self.get_logger().warn(f'Resize failed: {e}')
            return

        # 3) YOLO inference
        try:
            results = self.model(frame, conf=self.conf_thres, verbose=False)
        except Exception as e:
            self.get_logger().warn(f'YOLO inference failed: {e}')
            return

        annotated = frame.copy()
        top_label, top_conf = None, 0.0
        best_box = None  # (x1, y1, x2, y2) in target_w x target_h coordinates

        # 4) Post-processing: find best detection and draw boxes
        try:
            r0 = results[0]
            if hasattr(r0, 'boxes') and r0.boxes is not None and len(r0.boxes):
                # Select highest-confidence detection
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

                # Draw all bounding boxes (for visualization)
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
                # No detections in this frame
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
            self.get_logger().warn(f'Post-processing failed: {e}')

        # 4.5) Overlays: center line and deadband region
        cx_img = int(self.target_w * 0.5)
        band = int(self.deadband * self.target_w)
        left_band = cx_img - band
        right_band = cx_img + band
        # Center line (white) and deadband bounds (yellow)
        cv2.line(annotated, (cx_img, 0), (cx_img, self.target_h - 1), (255, 255, 255), 1)
        cv2.line(annotated, (left_band, 0), (left_band, self.target_h - 1), (0, 255, 255), 1)
        cv2.line(annotated, (right_band, 0), (right_band, self.target_h - 1), (0, 255, 255), 1)

        # 5) Publish label, target and size_ratio if a best_box is available
        if top_label is not None and best_box is not None:
            # Publish best label
            self.pub_label.publish(String(data=top_label))

            x1, y1, x2, y2 = best_box
            bx = (x1 + x2) / 2.0
            by = (y1 + y2) / 2.0

            # Normalized center coordinates in [0, 1]
            nx = float(bx / self.target_w)
            ny = float(by / self.target_h)

            # Draw center point of bounding box
            cv2.circle(annotated, (int(bx), int(by)), 2, (0, 0, 255), -1)

            # ====== Bounding box relative size (size_ratio) ======
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            img_area = float(self.target_w * self.target_h)
            box_area = float(box_w * box_h)
            size_ratio = box_area / img_area if img_area > 0 else 0.0  # 0..1

            # Draw percentage in the top-right corner
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

            # Publish relative size on topic
            self.pub_size_ratio.publish(Float32(data=size_ratio))
            # ====== end of size_ratio computation ======

            # Publish normalized target position
            pt = PointStamped()
            pt.header = msg.header  # keep original timestamp
            pt.header.frame_id = 'camera'  # logical frame name
            pt.point.x = nx
            pt.point.y = ny
            pt.point.z = 0.0
            self.pub_target.publish(pt)

        # 6) Publish annotated image (target_w x target_h)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.pub_annotated.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f'Error publishing annotated image: {e}')

    # ========================== Dynamic parameter handling ==========================
    def parameters_callback(self, params):
        """
        Dynamic parameter callback.
        Allows updating the following parameters at runtime:
          - deadband   (DOUBLE)
          - conf       (DOUBLE)
          - target_w   (INTEGER)
          - target_h   (INTEGER)
        """
        for param in params:
            if param.name == 'deadband' and param.type_ == Parameter.Type.DOUBLE:
                self.deadband = param.value
                self.get_logger().info(f'deadband updated to {self.deadband}')
            elif param.name == 'conf' and param.type_ == Parameter.Type.DOUBLE:
                self.conf_thres = param.value
                self.get_logger().info(f'conf_thres updated to {self.conf_thres}')
            elif param.name == 'target_w' and param.type_ == Parameter.Type.INTEGER:
                self.target_w = param.value
                self.get_logger().info(f'target_w updated to {self.target_w}')
            elif param.name == 'target_h' and param.type_ == Parameter.Type.INTEGER:
                self.target_h = param.value
                self.get_logger().info(f'target_h updated to {self.target_h}')

        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    node = TrackerNodeScale()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
