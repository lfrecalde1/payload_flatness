from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package source path
    package_name = 'payload_flatness'  # Replace with your package name
    share_dir = get_package_share_directory(package_name)
    # Construct the path to the src directory assuming typical workspace structure
    package_src_path = os.path.join(share_dir, 'results')
    print(package_src_path)

    # Declare the 'path' argument for the launch file
    path_arg = DeclareLaunchArgument(
        'path',
        default_value=package_src_path,
        description='Path to the package source directory'
    )

    # Define the node and include the parameter
    quadrotor_node = Node(
        package=package_name,  # Replace with your package name
        executable='quadrotor_flatness_node',  # Replace with your node's executable name
        name='quadrotor_flatness_node',
        parameters=[{'path': LaunchConfiguration('path')}],
    )

    # Return the LaunchDescription with the arguments and node
    return LaunchDescription([
        path_arg,
        quadrotor_node
    ])