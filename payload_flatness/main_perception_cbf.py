#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PointStamped
from payload_flatness import rotation_casadi, rotation_inverse_casadi, quaternion_multiplication_casadi
import casadi as ca


# Functions From Casadi
rot = rotation_casadi()
inverse_rot = rotation_inverse_casadi()
quat_multi = quaternion_multiplication_casadi()

class PerceptionCBFNode(Node):
    def __init__(self):
        super().__init__('QUADROTOR_FLATNESS')

        # Declare parameters
        self.declare_parameter('path', '')
        self.declare_parameter('world_frame_id', 'world')
        self.declare_parameter('quadrotor_name', 'quadrotor')
        self.declare_parameter('camera_frame', 'camera')
        self.declare_parameter('camera_position.x', 0.0)
        self.declare_parameter('camera_position.y', 0.0)
        self.declare_parameter('camera_position.z', -0.075)
        self.declare_parameter('camera_ori.x', -0.707107)
        self.declare_parameter('camera_ori.y', 0.707107)
        self.declare_parameter('camera_ori.z', 0.0)
        self.declare_parameter('camera_ori.w', 0.0)

        # Get the parameter values
        self.package_path = self.get_parameter('path').get_parameter_value().string_value
        self.world_frame_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        self.quadrotor_name = self.get_parameter('quadrotor_name').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        self.camera_position_x = self.get_parameter('camera_position.x').get_parameter_value().double_value
        self.camera_position_y = self.get_parameter('camera_position.y').get_parameter_value().double_value
        self.camera_position_z = self.get_parameter('camera_position.z').get_parameter_value().double_value

        self.camera_ori_x = self.get_parameter('camera_ori.x').get_parameter_value().double_value
        self.camera_ori_y = self.get_parameter('camera_ori.y').get_parameter_value().double_value
        self.camera_ori_z = self.get_parameter('camera_ori.z').get_parameter_value().double_value
        self.camera_ori_w = self.get_parameter('camera_ori.w').get_parameter_value().double_value

        # Log the parameters to verify
        self.get_logger().info(f'Package path: {self.package_path}')
        self.get_logger().info(f'World frame ID: {self.world_frame_id}')
        self.get_logger().info(f'Quadrotor name: {self.quadrotor_name}')
        self.get_logger().info(f'Camera frame: {self.camera_frame}')
        self.get_logger().info(f'Camera position: ({self.camera_position_x}, {self.camera_position_y}, {self.camera_position_z})')
        self.get_logger().info(f'Camera orientation: ({self.camera_ori_x}, {self.camera_ori_y}, {self.camera_ori_z}, {self.camera_ori_w})')

        # Max norm for the perception staff
        self.r_max = 0.4
        self.r_max_square = self.r_max*self.r_max

        # Additional setup can be done here, using the retrieved parameters

        # Define Camera Position
        self.b_t_bc = np.array([self.camera_position_x, self.camera_position_y, self.camera_position_z])
        self.q_bc = np.array([self.camera_ori_w, self.camera_ori_x, self.camera_ori_y, self.camera_ori_z])

        # Subscriptions to topics
        self.quadrotor_odom_subscriber = self.create_subscription(Odometry, '/quadrotor/odom', self.quadrotor_odom_callback, 10)

        self.payload_odom_subscriber = self.create_subscription(Odometry, '/quadrotor/payload/odom', self.payload_odom_callback, 10)

        self.payload_projection_publisher = self.create_publisher(PointStamped, '/quadrotor/payload/point_python', 10)

        self.cbf_publisher = self.create_publisher(PointStamped, '/quadrotor/payload/classic', 10)

        # Init Pose ofr quadrtor and payload 
        self.quadrotor_pose = Pose()
        self.payload_pose = Pose()

        # Timer to call the projection function at 100 Hz
        self.timer = self.create_timer(0.01, self.projection)

    def quadrotor_odom_callback(self, msg):
        # Update the quadrotor pose with data from the Odometry message
        self.quadrotor_pose.position = msg.pose.pose.position
        self.quadrotor_pose.orientation = msg.pose.pose.orientation
        #self.get_logger().info(f'Updated quadrotor pose: {self.quadrotor_pose}')
        return None

    def payload_odom_callback(self, msg):
        # Update the payload pose with data from the Odometry message
        self.payload_pose.position = msg.pose.pose.position
        self.payload_pose.orientation = msg.pose.pose.orientation
        #self.get_logger().info(f'Updated payload pose: {self.payload_pose}')
        return None
    
    def projection(self):
        # Get values of the system
        # Internal states of the quadrotor
        q_wb = np.array([self.quadrotor_pose.orientation.w, self.quadrotor_pose.orientation.x, self.quadrotor_pose.orientation.y, self.quadrotor_pose.orientation.z])
        w_t_wb = np.array([self.quadrotor_pose.position.x, self.quadrotor_pose.position.y, self.quadrotor_pose.position.z])

        # Internal states of the payload
        w_t = np.array([self.payload_pose.position.x, self.payload_pose.position.y, self.payload_pose.position.z])

        q_wc = quat_multi(q_wb, self.q_bc)
        first_part = inverse_rot(q_wc, w_t - w_t_wb)
        second_part = inverse_rot(self.q_bc, self.b_t_bc)
        projection_value = first_part - second_part

        # Publish the current position of the quadrotor at 100 Hz using PointStamped
        point_stamped = PointStamped()
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.header.frame_id = self.camera_frame
        point_stamped.point.x = float(projection_value[0,0])
        point_stamped.point.y = float(projection_value[1,0])
        point_stamped.point.z = float(projection_value[2,0])
        # Publish the quadrotor position
        self.payload_projection_publisher.publish(point_stamped)

        # Compute distance to the center of the camera frame
        distance = ca.norm_2(projection_value[0:2])
        # CBF Information
        cbf_stamped = PointStamped()
        cbf_stamped.header.stamp = self.get_clock().now().to_msg()
        cbf_stamped.header.frame_id = self.camera_frame
        cbf_stamped.point.x = float(distance)
        cbf_stamped.point.y = 0.0
        cbf_stamped.point.z = -float(distance) + float(self.r_max)
        self.cbf_publisher.publish(cbf_stamped)
        return None


def main(args=None):
    rclpy.init(args=args)
    cbf_node = PerceptionCBFNode()
    try:
        rclpy.spin(cbf_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        cbf_node.get_logger().info('Simulation stopped manually.')
    finally:
        cbf_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()