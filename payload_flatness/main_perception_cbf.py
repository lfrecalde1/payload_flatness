#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PointStamped, Twist
from quadrotor_msgs.msg import TRPYCommand, PositionCommand
from payload_flatness import rotation_casadi, rotation_inverse_casadi, quaternion_multiplication_casadi
import casadi as ca
from payload_flatness import export_model


# Functions From Casadi
rot = rotation_casadi()
inverse_rot = rotation_inverse_casadi()
quat_multi = quaternion_multiplication_casadi()
model, f_x, g_x, h_f, distance_f, dh_dx_f, ddistance_dx_f, lyapunov_f = export_model()

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
        self.r_max = 0.3
        self.r_max_square = self.r_max*self.r_max

        # Additional setup can be done here, using the retrieved parameters

        # Define Camera Position
        self.b_t_bc = np.array([self.camera_position_x, self.camera_position_y, self.camera_position_z])
        self.q_bc = np.array([self.camera_ori_w, self.camera_ori_x, self.camera_ori_y, self.camera_ori_z])

        # Subscriptions to topics
        self.quadrotor_odom_subscriber = self.create_subscription(Odometry, f'/{self.quadrotor_name}/odom', self.quadrotor_odom_callback, 10)

        self.payload_odom_subscriber = self.create_subscription(Odometry, f'/{self.quadrotor_name}/payload/odom', self.payload_odom_callback, 10)

        self.payload_projection_publisher = self.create_publisher(PointStamped, f'/{self.quadrotor_name}/payload/point_python', 10)

        self.cbf_publisher = self.create_publisher(PointStamped, f'/{self.quadrotor_name}/payload/cbf', 10)

        self.quadrotor_desired_subscriber = self.create_subscription(PositionCommand, f'/{self.quadrotor_name}/position_cmd', self.desired_callback, 10)

        # Init Pose ofr quadrtor and payload 
        self.quadrotor_pose = Pose()
        self.quadrotor_twist = Twist()
        self.payload_pose = Pose()
        self.payload_twist = Twist()

        self.desired_values = np.zeros((24, ))

        # Timer to call the projection function at 100 Hz
        self.timer = self.create_timer(0.01, self.projection)

    def quadrotor_odom_callback(self, msg):
        # Update the quadrotor pose with data from the Odometry message
        self.quadrotor_pose.position = msg.pose.pose.position
        self.quadrotor_pose.orientation = msg.pose.pose.orientation

        self.quadrotor_twist.linear = msg.twist.twist.linear
        self.quadrotor_twist.angular = msg.twist.twist.angular
        #self.get_logger().info(f'Updated quadrotor pose: {self.quadrotor_pose}')
        return None

    def desired_callback(self, msg):
        # Update the quadrotor pose with data from the Odometry message
        self.desired_values[0] = msg.points[0].position.x
        self.desired_values[1] = msg.points[0].position.y
        self.desired_values[2] = msg.points[0].position.z

        self.desired_values[3] = msg.points[0].velocity.x
        self.desired_values[4] = msg.points[0].velocity.y
        self.desired_values[5] = msg.points[0].velocity.z

        self.desired_values[6] = msg.points[0].position_quad.x
        self.desired_values[7] = msg.points[0].position_quad.y
        self.desired_values[8] = msg.points[0].position_quad.z

        self.desired_values[9] = msg.points[0].velocity_quad.x
        self.desired_values[10] = msg.points[0].velocity_quad.y
        self.desired_values[11] = msg.points[0].velocity_quad.z

        self.desired_values[12] = msg.points[0].quaternion.w
        self.desired_values[13] = msg.points[0].quaternion.x
        self.desired_values[14] = msg.points[0].quaternion.y
        self.desired_values[15] = msg.points[0].quaternion.z

        self.desired_values[16] = msg.points[0].angular_velocity.x
        self.desired_values[17] = msg.points[0].angular_velocity.y
        self.desired_values[18] = msg.points[0].angular_velocity.z
        #self.get_logger().info(f'Updated quadrotor pose: {self.quadrotor_pose}')
        return None

    def payload_odom_callback(self, msg):
        # Update the payload pose with data from the Odometry message
        self.payload_pose.position = msg.pose.pose.position
        self.payload_pose.orientation = msg.pose.pose.orientation

        self.payload_twist.linear = msg.twist.twist.linear
        self.payload_twist.angular = msg.twist.twist.angular
        #self.get_logger().info(f'Updated payload pose: {self.payload_pose}')
        return None
    
    def projection(self):
        # Get values of the system
        # Casadi Function
        X = np.array([self.payload_pose.position.x, self.payload_pose.position.y, self.payload_pose.position.z,
                    self.payload_twist.linear.x, self.payload_twist.linear.y, self.payload_twist.linear.z,
                    self.quadrotor_pose.position.x, self.quadrotor_pose.position.y, self.quadrotor_pose.position.z,
                    self.quadrotor_twist.linear.x, self.quadrotor_twist.linear.y, self.quadrotor_twist.linear.z,
                    self.quadrotor_pose.orientation.w, self.quadrotor_pose.orientation.x, self.quadrotor_pose.orientation.y, self.quadrotor_pose.orientation.z,
                    self.quadrotor_twist.angular.x, self.quadrotor_twist.angular.y, self.quadrotor_twist.angular.z])
        X_desired = self.desired_values
        
        # External parameters for trajectory and slack or tau 
        param = np.zeros((X.shape[0] + 4 + 1, ))
        param[-1] = 1.0

        # Set Control Actions to zero
        U = np.zeros((4, ))

        # CBF Information
        cbf_stamped = PointStamped()
        cbf_stamped.header.stamp = self.get_clock().now().to_msg()
        cbf_stamped.header.frame_id = self.camera_frame

        cbf_stamped.point.x = float(h_f(X))
        cbf_stamped.point.y = 0.0
        cbf_stamped.point.z = float(ddistance_dx_f(X)@f_x(X, U, param))
        self.cbf_publisher.publish(cbf_stamped)

        # Publish the current position of the quadrotor at 100 Hz using PointStamped
        point_stamped = PointStamped()
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.header.frame_id = self.camera_frame
        point_stamped.point.x = float(ddistance_dx_f(X)@f_x(X, U, param)) - float(h_f(X))
        point_stamped.point.y = 0.0
        point_stamped.point.z = float(lyapunov_f(X, X_desired))
        # Publish the quadrotor position
        self.payload_projection_publisher.publish(point_stamped)
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