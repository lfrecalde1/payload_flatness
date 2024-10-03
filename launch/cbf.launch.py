from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    launch_args = [
        DeclareLaunchArgument('name', default_value='quadrotor'),
        DeclareLaunchArgument('world_frame_id', default_value='world'),
        DeclareLaunchArgument('camera_frame', default_value='camera'),
        DeclareLaunchArgument('camera_position.x', default_value='0.0'),
        DeclareLaunchArgument('camera_position.y', default_value='0.0'),
        DeclareLaunchArgument('camera_position.z', default_value='-0.075'),
        DeclareLaunchArgument('camera_ori.x', default_value='-0.707107'),
        DeclareLaunchArgument('camera_ori.y', default_value='0.707107'),
        DeclareLaunchArgument('camera_ori.z', default_value='0.0'),
        DeclareLaunchArgument('camera_ori.w', default_value='0.0'),
    ]

    # Use the LaunchConfiguration for each argument
    name = LaunchConfiguration('name')
    world_frame_id = LaunchConfiguration('world_frame_id')
    camera_frame = LaunchConfiguration('camera_frame')

    camera_position_x = LaunchConfiguration('camera_position.x')
    camera_position_y = LaunchConfiguration('camera_position.y')
    camera_position_z = LaunchConfiguration('camera_position.z')

    camera_ori_x = LaunchConfiguration('camera_ori.x')
    camera_ori_y = LaunchConfiguration('camera_ori.y')
    camera_ori_z = LaunchConfiguration('camera_ori.z')
    camera_ori_w = LaunchConfiguration('camera_ori.w')

    package_name = 'payload_flatness'  # Replace with your package name

    # Define the node and include the parameter
    quadrotor_node = Node(
        package=package_name,  # Replace with your package name
        executable='cbf_node',  # Replace with your node's executable name
        name='cbf_node',
        parameters=[
            {'world_frame_id': world_frame_id},
            {'quadrotor_name': name},
            {'camera_frame': camera_frame},
            {'camera_position.x': camera_position_x},
            {'camera_position.y': camera_position_y},
            {'camera_position.z': camera_position_z},
            {'camera_ori.x': camera_ori_x},
            {'camera_ori.y': camera_ori_y},
            {'camera_ori.z': camera_ori_z},
            {'camera_ori.w': camera_ori_w},
        ],
        output='screen'
    )

    # Create the launch description and include the launch arguments and nodes
    return LaunchDescription(launch_args + [quadrotor_node])