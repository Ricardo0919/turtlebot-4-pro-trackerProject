#!/usr/bin/env python3
# ===============================================================================
# Project: Turtlebot 4 Pro - Person Tracker - Tracker Node (YOLO .pt)
# Student Project
# Date: December 7th, 2025
# Students:
#   - Sergi Fernandez Mendez
#   - Ricardo Sierra Roa
# ===============================================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


class TrackerNode(Node):
    """
    TrackerNode
    -----------
    ROS2 node for person detection and tracking using a YOLO .pt model.

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

    Behavior:
      - Resizes the input image to (target_w, target_h).
      - Runs YOLO inference with a .pt model.
      - Selects the highest-confidence bounding box as the tracking target.
      - Publishes:
          * The best label and its normalized center as a target.
          * The annotated image with bounding boxes, labels,
            and a central vertical line with horizontal deadband.
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

        # Training resolution 160x120 (resize input to this size before inference)
        self.target_w = self.declare_parameter(
            'target_w',
            160
        ).get_parameter_value().integer_value

        self.target_h = self.declare_parameter(
            'target_h',
            120
        ).get_parameter_value().integer_value

        # Horizontal deadband in normalized coordinates (e.g. 0.05 ≈ 5% of image width)
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

        # =============================== Model loading ===============================
        share_dir = Path(get_package_share_directory('tracking_person'))
        model_path = share_dir / 'models' / self.model_file
        if not model_path.exists():
            self.get_logger().error(f'Could not find model file at: {model_path}')
            raise FileNotFoundError(model_path)

        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(str(model_path))
        self.names = self.model.names

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
          4. Select best detection and compute normalized target.
          5. Draw overlays and publish outputs.
        """
        # 1) ROS Image → OpenCV (BGR)
        try:
            orig = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion error: {e}')
            return

        # 2) Resize to target resolution (e.g. 160×120)
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

                # Draw all bounding boxes (optional, for visualization)
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
                    0.7,
                    (0, 0, 255),
                    2,
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

        # 5) Publish label and target if a best_box is available
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

            # Publish center as target
            pt = PointStamped()
            pt.header = msg.header  # keep original timestamp
            pt.header.frame_id = 'camera'  # logical frame name
            pt.point.x = nx
            pt.point.y = ny
            pt.point.z = 0.0
            self.pub_target.publish(pt)

        # 6) Publish annotated image (resized)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.pub_annotated.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f'Error publishing annotated image: {e}')


def main():
    rclpy.init()
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
