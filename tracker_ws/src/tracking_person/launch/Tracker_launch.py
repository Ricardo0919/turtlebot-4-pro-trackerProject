#!/usr/bin/env python3
# ===============================================================================
# Project: Turtlebot 4 Pro - Person Tracker - Launch File
# Student Project
# Date: December 7th, 2025
# Students:
#   - Sergi Fernandez Mendez
#   - Ricardo Sierra Roa
# ===============================================================================

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # ================================== TrackerNodeOnnx ==================================
    # Node responsible for person detection and tracking using an ONNX model.
    # Subscribes to:
    #   - /oakd/rgb/preview/image_raw  (RGB image from the OAK-D camera)
    # Publishes (within tracking_person package, typical topics):
    #   - tracking_person/target       (PointStamped with normalized coordinates)
    #   - tracking_person/size_ratio   (Float32 with relative size of the person)
    #   - tracking_person/annotated    (Annotated image for visualization)
    tracker_onnx = Node(
        package='tracking_person',
        executable='TrackerNodeOnnx',
        name='tracking_person',
        output='screen',
        parameters=[{
            'rgb_topic': '/oakd/rgb/preview/image_raw',
            'model_file': 'TrackerPerson.onnx',
            'conf': 0.60,      # Minimum confidence threshold for detections
            'deadband': 0.20,  # Horizontal deadband for selecting the target
            'target_w': 160,   # Internal processing width
            'target_h': 120,   # Internal processing height
        }],
    )

    # ===================================== MotorsNode =====================================
    # Node responsible for generating velocity commands based on the tracker output.
    # Subscribes to:
    #   - tracking_person/target       (PointStamped)
    #   - tracking_person/size_ratio   (Float32)
    # Publishes:
    #   - /cmd_vel (TwistStamped)
    # Parameters:
    #   - kp, deadband, max_w: yaw controller configuration.
    #   - timeout_s: maximum age for tracking messages.
    #   - near_threshold, far_threshold, linear_speed: distance control.
    #   - search_on_no_target: kept for compatibility (not used in the current controller).
    motors = Node(
        package='tracking_person',
        executable='MotorsNode',
        name='motors_node',
        output='screen',
        parameters=[{
            'kp': 1.5,
            'deadband': 0.05,
            'max_w': 1.5,
            'timeout_s': 0.5,
            'near_threshold': 0.60,
            'far_threshold': 0.65,
            'linear_speed': 0.15,
            'search_on_no_target': False,
        }],
    )

    # ================================ rqt_image_view (input) ================================
    # rqt_image_view instance for visualizing the raw RGB image from the OAK-D camera.
    rqt_in = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_in',
        output='screen',
        remappings=[('image', '/oakd/rgb/preview/image_raw')],
    )

    # ============================== rqt_image_view (annotated) ==============================
    # rqt_image_view instance for visualizing the annotated output from the tracker node.
    rqt_out = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_out',
        output='screen',
        remappings=[('image', 'tracking_person/annotated')],
    )

    # ================================ Launch description ====================================
    # Launch all nodes required for person tracking and visualization.
    return LaunchDescription([tracker_onnx, motors, rqt_in, rqt_out])
