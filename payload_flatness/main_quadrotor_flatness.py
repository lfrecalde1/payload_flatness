#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import threading
import os
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from payload_flatness import compute_flatness_quadrotor
from payload_flatness import compute_flatness_quadrotor_old
from payload_flatness import compute_flatness_payload
from payload_flatness import fancy_plots_3, plot_states_position
from payload_flatness import fancy_plots_4, plot_control_actions_reference, plot_states_quaternion
import time

class FlatnessQuadrotorNode(Node):
    def __init__(self):
        super().__init__('QUADROTOR_FLATNESS')

        # Declare the 'path' parameter with an empty string as the default value
        self.declare_parameter('path', '')

        # Get the parameter value
        self.package_path = self.get_parameter('path').get_parameter_value().string_value

        # Check if the path is provided
        if not self.package_path:
            self.get_logger().error('The "path" parameter was not provided in the launch file.')
            raise RuntimeError('The "path" parameter is required but was not provided.')

        # Check if the path exists
        if not os.path.exists(self.package_path):
            self.get_logger().error(f'The provided path "{self.package_path}" does not exist.')
            raise RuntimeError(f'The provided path "{self.package_path}" does not exist.')

        # Other initialization code...
        self.get_logger().info(f'Using path: {self.package_path}')

        # Mass and Inertia
        self.g = 9.81
        self.mQ = 1.272

        # Inertia Matrix
        self.Jxx = 0.00304475
        self.Jyy = 0.00454981
        self.Jzz = 0.00281995
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])

        self.dx = 0.1
        self.dy = 0.2
        self.dz = 0.5
        self.D = np.array([[self.dx, 0.0, 0.0], [0.0, self.dy, 0.0], [0.0, 0.0, self.dz]])

        self.Lq = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g, self.dx, self.dy, self.dz]

        # Payload parameters
        self.mL = 0.103
        self.l = 0.5
        self.dlx = 0.0
        self.dly = 0.0
        self.dlz = 0.0
        self.Dl = np.array([[self.dlx, 0.0, 0.0], [0.0, self.dly, 0.0], [0.0, 0.0, self.dlz]])
        self.Ll = [self.mL, self.l, self.g, self.dlx, self.dly, self.dlz]

        # Time of the system
        self.t_f = 10.0
        self.ts = 0.01
        self.t = np.arange(0, self.t_f + self.ts, self.ts)

        # Desired Trajectory parameters
        self.p = 2.0  # Radius
        self.w_c = 1 # Angular velocity
        self.c = np.array([0.0, 0.0, 0.5])  # Center

        print("Computing Path")
        self.hd, self.hd_p, self.hd_pp, self.hd_ppp, self.hd_pppp, self.qd, self.fd, self.wd = compute_flatness_quadrotor(self.Lq, self.t, self.p, self.w_c, self.c)
        self.hd_old, self.hd_p_old, self.hd_pp_old, self.hd_ppp_old, self.hd_pppp_old, self.qd_old, self.fd_old, self.wd_old = compute_flatness_quadrotor_old(self.Lq, self.t, self.p, self.w_c, self.c)
        compute_flatness_payload(self.Lq, self.Ll, self.t, self.p, self.w_c, self.c)
        print("Path Computed")

        # Define odometry publisher for the desired path
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "desired_frame", 10)

        # Publisher for the desired trajectory using PathStamped
        self.path_msg = Path()
        self.path_msg.header.frame_id = "world"
        self.publisher_ref_trajectory_ = self.create_publisher(Path, 'desired_path', 10)

        # Create a thread to run the simulation and viewer
        self.simulation_thread = threading.Thread(target=self.run)
        self.simulation_thread.start()

    def send_ref(self, h, q):
        self.ref_msg.header.frame_id = "world"
        self.ref_msg.header.stamp = self.get_clock().now().to_msg()

        self.ref_msg.pose.pose.position.x = h[0]
        self.ref_msg.pose.pose.position.y = h[1]
        self.ref_msg.pose.pose.position.z = h[2]

        self.ref_msg.pose.pose.orientation.x = q[1]
        self.ref_msg.pose.pose.orientation.y = q[2]
        self.ref_msg.pose.pose.orientation.z = q[3]
        self.ref_msg.pose.pose.orientation.w = q[0]

        # Send Message
        self.publisher_ref_.publish(self.ref_msg)
        return None 

    def add_path_point(self, h, q):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = h[0]
        pose.pose.position.y = h[1]
        pose.pose.position.z = h[2]

        pose.pose.orientation.x = q[1]
        pose.pose.orientation.y = q[2]
        pose.pose.orientation.z = q[3]
        pose.pose.orientation.w = q[0]

        self.path_msg.poses.append(pose)
        self.publisher_ref_trajectory_.publish(self.path_msg)

    def run(self):
        # Simulation loop
        for k in range(0, self.t.shape[0]):
            # Get model
            tic = time.time()

            # Add current desired state to the path
            self.add_path_point(self.hd[:, k], self.qd[:, k])

            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            self.get_logger().info("FLATNESS QUADROTOR")
        
        # Plot Orientation of the system
        fig11, ax11, ax12, ax13 = fancy_plots_3()
        plot_states_position(fig11, ax11, ax12, ax13, self.hd[0:3, :], self.hd_old[0:3, :], self.t, "Desired Position Trajectory ", self.package_path)
        plt.show()

        fig21, ax21, ax22, ax23, ax24 = fancy_plots_4()
        plot_control_actions_reference(fig21, ax21, ax22, ax23, ax24, self.fd, self.wd, self.fd_old, self.wd_old, self.t, "Desired Control Actions ", self.package_path)
        plt.show()

        fig31, ax31, ax32, ax33, ax34 = fancy_plots_4()
        plot_states_quaternion(fig31, ax31, ax32, ax33, ax34, self.qd, self.qd_old, self.t, "Desired Orientations  ", self.package_path)
        plt.show()

        None

def main(args=None):
    rclpy.init(args=args)
    planning_node = FlatnessQuadrotorNode()
    try:
        rclpy.spin(planning_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        planning_node.get_logger().info('Simulation stopped manually.')
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()