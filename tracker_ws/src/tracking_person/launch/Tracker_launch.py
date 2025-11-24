# launch/Tracker_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    tracker = Node(
        package='tracking_person',
        executable='TrackerNode',
        name='tracking_person',
        output='screen',
        parameters=[{
            'rgb_topic': '/oakd/rgb/preview/image_raw',
            'model_file': 'TrackerPerson.pt',
            'conf': 0.60,
            'deadband': 0.20,
            'target_w': 120,
            'target_h': 160,
        }],
    )

    motors = Node(
        package='tracking_person',
        executable='MotorsNode',
        name='MotorsNode',
        output='screen',
        parameters=[{
            'kp': 1.5,
            'deadband': 0.05,
            'max_w': 1.5,
            'timeout_s': 0.5,
        }],
    )

    rqt_in = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_in',
        output='screen',
        remappings=[('image', '/oakd/rgb/preview/image_raw')],
    )

    rqt_out = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_out',
        output='screen',
        remappings=[('image', 'tracking_person/annotated')],
    )

    return LaunchDescription([tracker, motors, rqt_in, rqt_out])
